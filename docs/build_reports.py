#!/usr/bin/env python3
"""Build the CS5491 milestone and final report PDFs.

Format follows the course rules: A4, Times New Roman 12 pt, 2.5 cm margins,
single spacing.  Result tables are read from CSVs produced by the experiment
scripts; if a CSV is absent the cell is marked as pending rather than filled
with invented numbers.

Usage:
    python docs/build_reports.py
"""

import csv
import os
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle,
                                PageBreak, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = Path(os.environ.get('WINDIR', r'C:\Windows')) / 'Fonts'


def register_fonts():
    for name, filename in [('TNR', 'times.ttf'),
                           ('TNR-Bold', 'timesbd.ttf'),
                           ('TNR-Italic', 'timesi.ttf'),
                           ('TNR-BoldItalic', 'timesbi.ttf')]:
        path = FONT_DIR / filename
        if path.exists():
            pdfmetrics.registerFont(TTFont(name, str(path)))
        else:
            pdfmetrics.registerFont(TTFont(name, 'Times-Roman'))
    pdfmetrics.registerFontFamily('TNR', normal='TNR', bold='TNR-Bold',
                                  italic='TNR-Italic',
                                  boldItalic='TNR-BoldItalic')


register_fonts()

BODY = ParagraphStyle('body', fontName='TNR', fontSize=12, leading=14.5,
                      alignment=TA_JUSTIFY, spaceAfter=5)
TITLE = ParagraphStyle('title', fontName='TNR-Bold', fontSize=14, leading=17,
                       alignment=TA_CENTER, spaceAfter=5)
SUB = ParagraphStyle('sub', fontName='TNR', fontSize=11, leading=13,
                     alignment=TA_CENTER, spaceAfter=10)
HEAD = ParagraphStyle('head', fontName='TNR-Bold', fontSize=12, leading=14.5,
                      spaceBefore=9, spaceAfter=4)
CAP = ParagraphStyle('cap', fontName='TNR', fontSize=10.5, leading=12.5,
                     spaceBefore=3, spaceAfter=7)
REF = ParagraphStyle('ref', fontName='TNR', fontSize=10.5, leading=12.5,
                     spaceAfter=2)
BULLET = ParagraphStyle('bullet', parent=BODY, leftIndent=14,
                        bulletIndent=4, spaceAfter=3)


def P(text, style=BODY):
    return Paragraph(text, style)


def bullet(text):
    return Paragraph('&bull;&nbsp;&nbsp;' + text, BULLET)


def make_table(data, col_widths, font_size=10):
    style = ParagraphStyle('cell', fontName='TNR', fontSize=font_size,
                           leading=font_size + 2)
    head_style = ParagraphStyle('cellh', fontName='TNR-Bold',
                                fontSize=font_size, leading=font_size + 2)
    rows = []
    for r, row in enumerate(data):
        rows.append([Paragraph(str(c), head_style if r == 0 else style)
                     for c in row])
    t = Table(rows, colWidths=col_widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#777777')),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8e8e8')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return t


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def result_table(rows, depths, note):
    norms = ['batch', 'layer', 'graph', 'pair', 'none']
    lookup = {(r['norm'], int(r['depth'])): r for r in rows}
    data = [['Norm'] + ['d={}'.format(d) for d in depths]]
    for norm in norms:
        cells = []
        for d in depths:
            row = lookup.get((norm, d))
            if row is None:
                cells.append('pending')
            elif float(row.get('test_mae_mean', 0)) == 0 and note == 'zinc':
                cells.append('pending')
            else:
                cells.append('{:.4f}±{:.3f}'.format(
                    float(row['test_mae_mean']), float(row['test_mae_std'])))
        data.append([norm] + cells)
    return make_table(data, [2.2 * cm] + [2.75 * cm] * len(depths), 10)


def chart_from_pilot(rows):
    try:
        from reportlab.graphics.shapes import Drawing, Line, Circle, String
    except Exception:
        return P('[Figure omitted: chart library unavailable]', CAP)
    depths = [4, 8, 12, 20]
    norms = ['batch', 'layer', 'graph', 'pair', 'none']
    lookup = {(r['norm'], int(r['depth'])): float(r['test_mae_mean'])
              for r in rows}
    series = [[lookup.get((n, d), 0) for d in depths] for n in norms]
    nonzero = [v for s in series for v in s if v > 0]
    if not nonzero:
        return P('[Figure pending experimental results]', CAP)
    ymin, ymax = max(0, min(nonzero) - 0.02), max(nonzero) + 0.02
    x0, x1, y0, y1 = 45, 320, 35, 148
    palette = [colors.black, colors.HexColor('#2b6cb0'),
               colors.HexColor('#c53030'), colors.HexColor('#2f855a'),
               colors.HexColor('#805ad5')]

    def px(i):
        return x0 + (x1 - x0) * i / (len(depths) - 1)

    def py(v):
        return y0 + (y1 - y0) * (v - ymin) / (ymax - ymin)

    drawing = Drawing(430, 175)
    drawing.add(Line(x0, y0, x1, y0, strokeColor=colors.black))
    drawing.add(Line(x0, y0, x0, y1, strokeColor=colors.black))
    for i, d in enumerate(depths):
        drawing.add(String(px(i) - 4, y0 - 11, str(d),
                           fontName='TNR', fontSize=8))
    for frac in (0, 0.5, 1):
        value = ymin + (ymax - ymin) * frac
        drawing.add(Line(x0 - 3, py(value), x0, py(value),
                         strokeColor=colors.black))
        drawing.add(String(x0 - 28, py(value) - 3, '{:.2f}'.format(value),
                           fontName='TNR', fontSize=7.5))
    for si, values in enumerate(series):
        pts = [(px(i), py(v)) for i, v in enumerate(values)]
        for i in range(len(pts) - 1):
            drawing.add(Line(pts[i][0], pts[i][1], pts[i + 1][0],
                             pts[i + 1][1], strokeColor=palette[si],
                             strokeWidth=1.2))
        for x, y in pts:
            drawing.add(Circle(x, y, 2.2, fillColor=palette[si],
                               strokeColor=palette[si]))
    drawing.add(String((x0 + x1) / 2 - 18, 12, 'Network depth',
                       fontName='TNR', fontSize=8))
    for i, (name, color) in enumerate(zip(norms, palette)):
        yy = 133 - i * 13
        drawing.add(Line(335, yy, 350, yy, strokeColor=color, strokeWidth=1.4))
        drawing.add(Circle(342.5, yy, 2, fillColor=color, strokeColor=color))
        drawing.add(String(356, yy - 3, name, fontName='TNR', fontSize=8))
    return KeepTogether([drawing,
                         P('Figure 1. Synthetic-pilot test MAE versus depth '
                           '(lower is better). Not ZINC results.', CAP)])


def page_decorator(canvas, doc):
    canvas.saveState()
    canvas.setFont('TNR', 9)
    canvas.drawString(2.5 * cm, 1.3 * cm,
                      'CS5491 Project — Xinsen Yuan (72656002), '
                      'Zanqun Li (72655183)')
    canvas.drawRightString(A4[0] - 2.5 * cm, 1.3 * cm, str(doc.page))
    canvas.restoreState()


def build_doc(path, story):
    doc = BaseDocTemplate(str(path), pagesize=A4,
                          leftMargin=2.5 * cm, rightMargin=2.5 * cm,
                          topMargin=2.2 * cm, bottomMargin=2.2 * cm,
                          title='CS5491 Project', author='Xinsen Yuan')
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id='body')
    doc.addPageTemplates([PageTemplate(id='main', frames=[frame],
                                       onPage=page_decorator)])
    doc.build(story)


def milestone_story(pilot, zinc):
    s = []
    s += [P('Project Milestone Report', TITLE),
          P('Which Normalization Makes Classic GNNs Strong? '
            'A Study of the Normalization Component in GNN+', SUB),
          P('CS5491 &nbsp;|&nbsp; Xinsen Yuan (72656002), '
            'Zanqun Li (72655183)', SUB)]

    s += [P('1. Problem description and motivation', HEAD)]
    s += [P('Graph Transformers have become the default choice for graph-level '
            'prediction, but Luo, Shi and Wu (ICML 2025) showed that classic '
            'GCN, GIN and GatedGCN can match or beat them once six standard '
            'ingredients are applied: edge features, normalization, dropout, '
            'residual connections, feed-forward networks and positional '
            'encoding. Their GNN+ recipe is evaluated on 14 datasets and its '
            'most striking claim is that residual connections allow classic '
            'GNNs to be trained as deep as 20 layers.')]
    s += [P('Normalization is the one ingredient that is both essential to '
            'deep-network stability and under-studied in that work. The '
            'released code sets <font name="TNR-Bold">self.batch_norm = True'
            '</font> in every convolution layer and never varies the '
            'normalization type; the published ablation only asks whether '
            'normalization is present. Consequently the depth claim rests on '
            'an unexamined assumption: that BatchNorm remains suitable at 20 '
            'layers when a GraphGym batch contains roughly 32 small molecular '
            'graphs. Our project turns that implicit assumption into a '
            'measurable question.')]

    s += [P('2. Research question and hypotheses', HEAD)]
    s += [P('In GNN+, does the choice of normalization affect graph-level '
            'performance, and how does it interact with depth?')]
    s += [bullet('Removing normalization degrades performance, reproducing '
                 'the published ablation.'),
          bullet('BatchNorm is strong at shallow depth but degrades at 20 '
                 'layers because its batch statistics are noisy.'),
          bullet('Graph-aware alternatives remain comparatively stable at '
                 'depth.')]

    s += [P('3. Design of the method / approach', HEAD)]
    s += [P('We introduce a single factory, <font name="TNR-Bold">build_norm'
            '(name, dim)</font>, in a new module <font name="TNR">'
            'GNNPlus/layer/norm.py</font>. It exposes five options behind one '
            'config key, <font name="TNR-Bold">cfg.gnn.norm</font>: BatchNorm, '
            'LayerNorm, GraphNorm, PairNorm and an identity control. All '
            'options implement the same <font name="TNR">forward(x, batch)'
            '</font> signature, where the PyG graph-index vector is used only '
            'by GraphNorm. The default remains <font name="TNR">batch'
            '</font>, so the published configuration is numerically '
            'unchanged. Four convolution-layer files read the new key instead '
            'of the hard-coded flag; the edit is about thirty lines and is '
            'recorded in <font name="TNR">MYPROJECT_CHANGES.diff</font>.')]
    s += [P('The experiment is a 5 × 4 grid: normalization in {batch, layer, '
            'graph, pair, none} crossed with depth in {4, 8, 12, 20}, three '
            'seeds per cell. We use the ZINC subset (about 12,000 molecular '
            'graphs, downloaded automatically by PyG), the GCN+ backbone and '
            'test MAE (lower is better) selected at the best validation epoch. '
            'Each run uses 300 epochs to keep the grid affordable; key cells '
            'will be re-run at the paper’s 2000 epochs. A sweep script '
            'generates one YAML per cell and an aggregator produces the '
            'mean ± standard-deviation table, so every reported number can be '
            'recomputed from the committed code.')]

    s += [P('4. Preliminary results', HEAD)]
    s += [P('The patch is complete and passes a PyTorch-only smoke test: the '
            '<font name="TNR">batch</font> option is numerically identical to '
            'the original <font name="TNR">nn.BatchNorm1d</font>, and all five '
            'options return finite, correctly shaped tensors. To obtain real '
            'numbers without a GPU, we also ran a CPU pilot on a synthetic '
            'graph-regression task with the same normalization module '
            '(3 seeds, 120 epochs):')]
    depths = sorted({int(r['depth']) for r in pilot}) or [4, 8, 12, 20]
    s += [result_table(pilot, depths, 'pilot'),
          P('Table 1. Synthetic-pilot test MAE (mean ± std). This is a sanity '
            'check of the mechanism, not an evaluation of GNN+ on ZINC.', CAP)]
    if pilot:
        s += [chart_from_pilot(pilot)]
    s += [P('The pilot is informative precisely because it does not reproduce '
            'the paper’s setting: <font name="TNR">none</font> is competitive '
            'on this simple synthetic target, while BatchNorm degrades from '
            '4 to 20 layers. That contrast supports the premise that the '
            'effect of normalization depends on the data and the depth '
            'regime, and it underlines why the ZINC grid is necessary.')]
    s += [P('Status of the main experiment: the code and configs are ready '
            'and the exact command is <font name="TNR">python '
            'experiments/run_norm_study.py --repeat 3 --epochs 300</font>. '
            'The ZINC table will be filled from '
            '<font name="TNR">results/norm_study_summary.csv</font> once the '
            'GPU run completes; no values are fabricated here.')]

    s += [P('5. Work remaining', HEAD)]
    s += [bullet('Run the full ZINC grid and aggregate the main result table.'),
          bullet('Re-run key cells at 2000 epochs for the final table.'),
          bullet('Complete the optional positional-encoding substitution and '
                 'write the four-page final report.')]

    s += [P('References', HEAD)]
    s += [P('[1] Y. Luo, L. Shi, X.-M. Wu. Can Classic GNNs Be Strong '
            'Baselines for Graph-level Tasks? ICML 2025.', REF),
          P('[2] T. Cai et al. GraphNorm: A Principled Approach to '
            'Accelerating Graph Neural Network Training. ICML 2021.', REF),
          P('[3] L. Zhao, L. Akoglu. PairNorm: Tackling Oversmoothing in '
            'GNNs. ICLR 2020.', REF)]
    return s


def final_story(pilot, zinc):
    s = []
    s += [P('Which Normalization Makes Classic GNNs Strong?', TITLE),
          P('A normalization × depth study in GNN+  |  CS5491 course project',
            SUB),
          P('Xinsen Yuan (72656002) and Zanqun Li (72655183)', SUB)]

    s += [P('Abstract', HEAD)]
    s += [P('The GNN+ recipe shows that classic graph neural networks are '
            'strong baselines for graph-level tasks when a small set of '
            'generic techniques is applied. Normalization is one of those '
            'techniques, yet the released implementation fixes it to '
            'BatchNorm and the paper never asks whether another normalization '
            'would be better, especially at the depth of 20 layers that its '
            'residual connections enable. We make normalization configurable '
            'in the official codebase and study five choices across four '
            'depths on ZINC, with three seeds per cell. We report a '
            'synthetic-data pilot that validates the mechanism and a '
            'single-command reproduction protocol for the main experiment.')]

    s += [P('1. Motivation and problem', HEAD)]
    s += [P('Graph Transformers dominate many graph-level benchmarks, but '
            'their attention mechanism is expensive. Luo et al. (ICML 2025) '
            'challenged this trend by showing that GCN, GIN and GatedGCN, '
            'equipped with edge features, normalization, dropout, residual '
            'connections, feed-forward blocks and positional encoding, match '
            'or exceed Transformers on 14 datasets. The ingredient that most '
            'directly governs whether a deep network can be optimized is '
            'normalization, and it is also the only ingredient hard-coded in '
            'the public implementation: every layer constructs '
            '<font name="TNR">nn.BatchNorm1d</font> unconditionally. The '
            'depth result (20 message-passing layers) was therefore obtained '
            'with one particular normalization choice and one particular '
            'batch size (32 graphs). We investigate whether the depth claim '
            'is normalization-dependent.')]

    s += [P('2. Method: a configurable normalization component', HEAD)]
    s += [P('Our patch is intentionally small so that the published baseline '
            'remains reproducible. A new factory returns a uniform module '
            'with the call signature <font name="TNR">forward(x, batch)</font>:')]
    s += [make_table([
        ['Option', 'Implementation', 'Parameters', 'Uses graph index'],
        ['batch', 'nn.BatchNorm1d', '2 × dim', 'No'],
        ['layer', 'nn.LayerNorm', '2 × dim', 'No'],
        ['graph', 'GraphNorm with learnable mean mixture', '2 × dim + 1', 'Yes'],
        ['pair', 'PairNorm (center and rescale)', '0', 'No'],
        ['none', 'Identity control', '0', 'No'],
    ], [2.0 * cm, 7.0 * cm, 3.0 * cm, 3.6 * cm], 10)]
    s += [P('Table 2. Normalization options exposed by '
            '<font name="TNR">build_norm</font>.', CAP)]
    s += [P('GraphNorm follows Cai et al. (ICML 2021) with a learnable scalar '
            'that interpolates between a global mean and per-graph means; '
            'PairNorm follows Zhao and Akoglu (ICLR 2020) and has no '
            'parameters, making it a useful efficiency control. The config '
            'key <font name="TNR">cfg.gnn.norm</font> is registered with '
            'default <font name="TNR">batch</font>, and the four convolution '
            'layers replace their hard-coded BatchNorm construction with the '
            'factory. Because the default path is the original module, the '
            'unmodified <font name="TNR">configs/gcn/zinc.yaml</font> still '
            'reproduces the paper exactly. This is appropriate for a course '
            'project: the scientific change is isolated, auditable in '
            '<font name="TNR">MYPROJECT_CHANGES.diff</font>, and does not '
            'require reimplementing the backbone.')]

    s += [P('2.1 Integration with the GNN+ layers', HEAD)]
    s += [P('Each convolution layer previously created two or three '
            '<font name="TNR">nn.BatchNorm1d</font> objects: one for the '
            'message-passing output and, when the feed-forward block is '
            'enabled, two more around it. We replaced every one of these '
            'sites with a factory call and pass the graph-index vector '
            '<font name="TNR">data.batch</font> into the unified forward '
            'method. BatchNorm, LayerNorm, PairNorm and the identity ignore '
            'that vector; GraphNorm uses it to interpolate between global '
            'and per-graph statistics. GatedGCN also normalizes edge '
            'features, for which no graph-index vector exists, so those calls '
            'use the global fallback.')]
    s += [P('The change is confined to configuration and layer construction; '
            'the data loader, positional encoders, loss, optimizer and '
            'training loop are untouched. This matters for validity: any '
            'difference between cells in the result table can be attributed '
            'to the normalization type and depth rather than to incidental '
            'changes in preprocessing or scheduling.')]

    s += [P('3. Experimental setup', HEAD)]
    s += [P('Dataset: ZINC subset, approximately 12,000 small molecular '
            'graphs with regression labels, automatically loaded by PyG. '
            'Model: <font name="TNR">custom_gnn</font> with '
            '<font name="TNR">layer_type: gcne</font> (GCN+). Factors: '
            'normalization in {batch, layer, graph, pair, none} and depth in '
            '{4, 8, 12, 20}. Three seeds per cell; test MAE (lower is better) '
            'is read at the epoch with best validation MAE and summarized as '
            'mean ± standard deviation. Each run trains for 300 epochs, with '
            'key cells scheduled for 2000 epochs to match the paper. '
            'Baselines are the paper’s reported GCN+ number and the '
            '<font name="TNR">none</font> control, which reproduces the '
            'remove-normalization ablation.')]

    s += [P('4. Results and insights', HEAD)]
    if zinc:
        zinc_depths = sorted({int(r['depth']) for r in zinc})
        s += [result_table(zinc, zinc_depths, 'zinc'),
              P('Table 3. ZINC test MAE, mean ± std over three seeds '
                '(lower is better).', CAP)]
    else:
        s += [P('Table 3 (ZINC test MAE) is generated automatically by '
                '<font name="TNR">experiments/aggregate_norm_study.py</font> '
                'from the training logs. It is marked pending here because '
                'the GPU run has not been executed; the repository contains '
                'the complete commands to produce it, and no numbers have '
                'been substituted by hand.', BODY)]
    s += [P('As a mechanism check that runs on any CPU, we applied the same '
            'normalization factory to a small synthetic graph-regression '
            'task (240 graphs, 12-dimensional hidden size, 3 seeds, 120 '
            'epochs).')]
    pilot_depths = sorted({int(r['depth']) for r in pilot}) or [4, 8, 12, 20]
    s += [result_table(pilot, pilot_depths, 'pilot'),
          P('Table 4. Synthetic-pilot test MAE (mean ± std). Not ZINC '
            'results; included only to demonstrate that the knob changes '
            'training behaviour.', CAP)]
    if pilot:
        s += [chart_from_pilot(pilot)]
    s += [P('The pilot yields a nuanced picture. BatchNorm’s error rises '
            'from the 4-layer to the 20-layer setting, consistent with the '
            'hypothesis that batch statistics over many small graphs become '
            'unreliable at depth. Conversely, the unnormalized control is '
            'competitive on this simple synthetic target, and GraphNorm does '
            'not dominate at depth. We therefore do not claim that the '
            'synthetic result settles the research question: it shows the '
            'question is empirically non-trivial and dataset-dependent, '
            'which is exactly why the ZINC grid matters. The final claim '
            'will be stated only after the ZINC table is generated from the '
            'logged runs.')]

    s += [P('5. Code, reproducibility and presentation', HEAD)]
    s += [P('The submission is a self-contained Git repository. '
            '<font name="TNR">README.md</font> records the conda commands, '
            'the exact Python version and package list, and one command per '
            'experiment. <font name="TNR">experiments/run_norm_study.py'
            '</font> creates a separate YAML and output directory for each '
            'cell, launches GraphGym with three seeds, writes logs and a '
            'manifest, and supports resuming. '
            '<font name="TNR">experiments/aggregate_norm_study.py</font> '
            'produces both a CSV and a Markdown table. New modules and '
            'functions have docstrings and every edited layer carries a '
            '<font name="TNR"># CS5491</font> comment, so the data pipeline, '
            'model definition and train/evaluation loop remain traceable. '
            'No private API keys or external services are used; ZINC is '
            'downloaded on first run and weights & biases logging is disabled '
            'in the study configs.')]
    s += [make_table([
        ['Purpose', 'Command (from repository root)'],
        ['Check the patch',
         'python experiments/test_norm_smoke.py'],
        ['Reproduce original setting',
         'python main.py --cfg configs/gcn/zinc.yaml --repeat 3 seed 0'],
        ['Run the full grid',
         'python experiments/run_norm_study.py --repeat 3 --epochs 300'],
        ['Build result table',
         'python experiments/aggregate_norm_study.py'],
        ['Rebuild this PDF',
         'python docs/build_reports.py'],
    ], [4.0 * cm, 11.6 * cm], 9),
        P('Table 5. Exact commands for every stage of reproduction.', CAP)]

    s += [P('6. Limitations and conclusion', HEAD)]
    s += [P('The 300-epoch budget is shorter than the paper’s 2000 epochs, '
            'so absolute MAE values are not directly comparable to the '
            'published table; the comparison of interest is relative across '
            'normalization choices under an identical budget. The '
            'synthetic pilot uses a smaller model and an artificial target, '
            'and is reported only as a mechanism check. Within these limits, '
            'the work contributes (i) an open patch that makes the GNN+ '
            'normalization slot configurable, (ii) a reproducible '
            'normalization × depth protocol, and (iii) preliminary evidence '
            'that depth-related conclusions cannot be separated from the '
            'normalization choice.')]

    s += [P('References', HEAD)]
    for ref in [
        '[1] Y. Luo, L. Shi, X.-M. Wu. Can Classic GNNs Be Strong Baselines '
        'for Graph-level Tasks? ICML 2025.',
        '[2] T. Cai et al. GraphNorm: A Principled Approach to Accelerating '
        'Graph Neural Network Training. ICML 2021.',
        '[3] L. Zhao, L. Akoglu. PairNorm: Tackling Oversmoothing in GNNs. '
        'ICLR 2020.',
    ]:
        s += [P(ref, REF)]
    return s


def main():
    pilot = read_csv(ROOT / 'results' / 'pilot_norm_study.csv')
    zinc = read_csv(ROOT / 'results' / 'norm_study_summary.csv')
    out_dir = ROOT / 'docs'
    out_dir.mkdir(parents=True, exist_ok=True)
    build_doc(out_dir / 'CS5491_Milestone_Report.pdf',
              milestone_story(pilot, zinc))
    build_doc(out_dir / 'CS5491_Final_Report.pdf',
              final_story(pilot, zinc))
    print('Wrote:', out_dir / 'CS5491_Milestone_Report.pdf')
    print('Wrote:', out_dir / 'CS5491_Final_Report.pdf')
    print('pilot rows: {}, ZINC rows: {}'.format(len(pilot), len(zinc)))


if __name__ == '__main__':
    main()
