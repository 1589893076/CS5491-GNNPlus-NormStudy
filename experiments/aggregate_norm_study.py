#!/usr/bin/env python3
"""Aggregate the normalization x depth sweep into a result table.

For each seed we select the epoch with the best *validation* MAE (argmin) and
report the *test* MAE at that epoch, exactly as GraphGym's ``agg_runs`` does.
Cells are summarized as mean +/- standard deviation over seeds.

Example
-------
    python experiments/aggregate_norm_study.py
"""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r'^zinc_norm_(batch|layer|graph|pair|none)_d(\d+)$')


def label(path):
    """Show a path relative to the repo when possible (absolute may be on another drive)."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_stats(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def best_test_mae(result_dir):
    val = read_stats(result_dir / 'val' / 'stats.json')
    test = read_stats(result_dir / 'test' / 'stats.json')
    if not val or not test:
        return None
    best_epoch = min(val, key=lambda r: r['mae'])['epoch']
    matches = [r for r in test if r['epoch'] == best_epoch]
    if not matches:
        return None
    return float(matches[0]['mae'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', default='results/norm_study')
    p.add_argument('--output', default='results/norm_study_summary')
    args = p.parse_args()

    root = ROOT / args.results_dir
    cells = {}
    for item in sorted(root.iterdir()) if root.exists() else []:
        match = PATTERN.match(item.name)
        if not match or not item.is_dir():
            continue
        norm, depth = match.group(1), int(match.group(2))
        values = []
        for seed_dir in sorted(item.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.isdigit():
                continue
            val = best_test_mae(seed_dir)
            if val is not None:
                values.append(val)
        if values:
            cells[(norm, depth)] = values

    rows = []
    for (norm, depth), values in sorted(cells.items(),
                                        key=lambda kv: (kv[0][1], kv[0][0])):
        rows.append({
            'norm': norm,
            'depth': depth,
            'seeds': len(values),
            'test_mae_mean': round(statistics.mean(values), 4),
            'test_mae_std': round(statistics.pstdev(values), 4)
                            if len(values) > 1 else 0.0,
            'all': '|'.join('{:.4f}'.format(v) for v in values),
        })

    out_path = ROOT / (args.output + '.csv')
    md_path = ROOT / (args.output + '.md')
    if not rows:
        for stale in (out_path, md_path):
            if stale.exists():
                stale.unlink()
                print('Removed empty summary:', label(stale))
        print('\nERROR: no completed runs found under {}.'.format(
            args.results_dir))
        print('Each cell needs val/stats.json and test/stats.json; none were')
        print('found, so no summary was written. Fix the sweep, re-run it,')
        print('then aggregate again. Never download an empty result table.')
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['norm', 'depth', 'seeds',
                                                'test_mae_mean',
                                                'test_mae_std', 'all'])
        writer.writeheader()
        writer.writerows(rows)

    depths = sorted({r['depth'] for r in rows})
    norms = sorted({r['norm'] for r in rows})
    lookup = {(r['norm'], r['depth']): r for r in rows}
    lines = ['| Normalization | ' + ' | '.join('Depth {}'.format(d)
                                               for d in depths) + ' |',
             '| --- | ' + ' | '.join('---' for _ in depths) + ' |']
    for norm in norms:
        cells_md = []
        for depth in depths:
            row = lookup.get((norm, depth))
            cells_md.append('n/a' if row is None else
                            '{:.4f} ± {:.4f}'.format(row['test_mae_mean'],
                                                    row['test_mae_std']))
        lines.append('| {} | {} |'.format(norm, ' | '.join(cells_md)))
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print('\n'.join(lines))
    print('\nSaved:\n  {}\n  {}'.format(label(out_path), label(md_path)))
    if len(rows) < 20:
        print('\nWARNING: only {} of 20 planned grid cells produced results;'
              ' the report will label this as a partial grid.'.format(len(rows)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
