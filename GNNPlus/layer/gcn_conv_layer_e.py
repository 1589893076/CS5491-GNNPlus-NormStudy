import torch.nn as nn
from torch_geometric.graphgym import cfg
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pyg_nn
from torch_scatter import scatter
import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from .norm import build_norm
import torch
from torch import Tensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)


class GCNConvWithEdges(GCNConv):
    def __init__(self, in_channels, out_channels, edge_dim=None, bias=True):
        super(GCNConvWithEdges, self).__init__(in_channels, out_channels, bias, add_self_loops=False, normalize=False)
        self.edge_dim = edge_dim
        # self.lin_edge = torch.nn.Linear(edge_dim, out_channels) if edge_dim is not None else None
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        
    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        # if self.lin_edge is None and x_j.size(-1) != edge_attr.size(-1):
        #     raise ValueError("Node and edge feature dimensionalities do not "
        #                      "match. Consider setting the 'edge_dim' "
        #                      "attribute of 'GCNConvWithEdges'")

        # Apply linear transformation to edge features if necessary
        # if self.lin_edge is not None:
        #     edge_attr = self.lin_edge(edge_attr)
        # print(edge_attr.shape,x_j.shape)
        # Modify the message passing to incorporate edge features (e.g., addition + ReLU)
        return (x_j + edge_attr).relu()
        # return (x_j).relu()

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None,
                edge_weight: OptTensor = None) -> Tensor:
        # CS5491 PyG-compatibility patch.
        #
        # Upstream called
        #     self.propagate(edge_index, x=x, edge_attr=edge_attr,
        #                    edge_weight=edge_weight, size=None)
        # Newer PyG releases generate a jittable `propagate` template from the
        # message() signature.  On Colab this raised
        #     TypeError: propagate() got an unexpected keyword argument
        #     'edge_attr'
        # before the first training step could finish.  Reproduce the same
        # computation with plain PyTorch so the layer no longer depends on
        # that template:
        #     x_j = x[edge_index[0]] ; msg = relu(x_j + edge_attr)
        #     out[i] = sum_{j -> i} msg_j   (GCNConv's default add/sum aggr)
        # __init__ passes add_self_loops=False and normalize=False, so there
        # is no adjacency normalisation to reproduce.  edge_weight was passed
        # to propagate() but never consumed by message(), so it stays unused.
        x = self.lin(x)
        if (not isinstance(edge_index, Tensor) or edge_index.dim() != 2
                or edge_index.size(0) != 2):
            raise TypeError('GCNConvWithEdges expects a [2, E] LongTensor '
                            'edge_index; got {!r}'.format(type(edge_index)))
        src, dst = edge_index[0], edge_index[1]
        x_j = x.index_select(0, src)
        msg = x_j.relu() if edge_attr is None else (x_j + edge_attr).relu()
        out = torch.zeros_like(x).index_add_(0, dst, msg)
        if self.bias is not None:
            out = out + self.bias
        return out
    
class GCNConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dropout, residual, ffn):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        # CS5491: configurable normalization (default = upstream BatchNorm).
        self.norm_name = getattr(cfg.gnn, 'norm', 'batch')
        self.batch_norm = self.norm_name != 'none'
        self.ffn = ffn
        self.bn_node_x = build_norm(self.norm_name, dim_out)
        self.act = nn.Sequential(
            register.act_dict[cfg.gnn.act](),
            nn.Dropout(self.dropout),
        )
        edge_dim = dim_in
        if edge_dim is not None:
            self.model = GCNConvWithEdges(dim_in, dim_out, edge_dim, bias=True)
        else:
            self.model = GCNConvWithEdges(dim_in, dim_out, bias=True)
        
        if self.ffn:
            # Feed Forward block.
            # NOTE: upstream uses dim_in here; preserved to avoid any change
            # in the default configuration (dim_in == dim_out for ZINC).
            self.norm1_local = build_norm(self.norm_name, dim_in)
            self.ff_linear1 = nn.Linear(dim_in, dim_in*2)
            self.ff_linear2 = nn.Linear(dim_in*2, dim_in)
            self.act_fn_ff = register.act_dict[cfg.gnn.act]()
            self.norm2 = build_norm(self.norm_name, dim_in)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)
        
    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def forward(self, batch):
        x_in = batch.x
        e_in = batch.edge_attr
        b_idx = getattr(batch, 'batch', None)  # CS5491: graph index for GraphNorm
        
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        batch.x = self.bn_node_x(batch.x, b_idx)
        batch.x = self.act(batch.x)
        
        if self.residual:
            batch.x = x_in + batch.x  # Residual connection.
        
        if self.ffn:
            batch.x = self.norm1_local(batch.x, b_idx)
            
            batch.x = batch.x + self._ff_block(batch.x)

            batch.x = self.norm2(batch.x, b_idx)

        return batch
    
