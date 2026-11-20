from torch_geometric.graphgym.register import register_config


@register_config('custom_gnn')
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    # Use residual connections between the GNN layers.
    cfg.gnn.residual = True
    cfg.gnn.ffn = True

    # CS5491 project: make normalization a first-class configurable component.
    # Default 'batch' reproduces the published GNN+ setup exactly.
    # Choices: batch | layer | graph | pair | none
    cfg.gnn.norm = 'batch'
