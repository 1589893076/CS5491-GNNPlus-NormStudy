#!/usr/bin/env python3
"""Sweep normalization types x network depth on ZINC using GNN+.

Every cell gets its own generated YAML file and therefore its own output
directory ``results/norm_study/<config-stem>/<seed>/``.  This makes the sweep
resumable and lets :mod:`aggregate_norm_study` find all runs deterministically.

Example
-------
    python experiments/run_norm_study.py --repeat 3 --epochs 300 --dry-run
    python experiments/run_norm_study.py --repeat 3 --epochs 300
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NORMS = ('batch', 'layer', 'graph', 'pair', 'none')
DEPTHS = (4, 8, 12, 20)


def make_config(norm, depth, epochs, template, tag=''):
    """Create a per-cell YAML based on the official zinc.yaml."""
    text = (ROOT / template).read_text(encoding='utf-8')
    text = text.replace('out_dir: results', 'out_dir: results/norm_study', 1)
    text = re.sub(r'(?m)^  max_epoch: .*$', '  max_epoch: {}'.format(epochs),
                  text, count=1)
    # A short budget (smoke check) must not be shorter than the configured
    # warmup: otherwise the LR stays in warmup for the whole run and the cell
    # "trains" without ever reaching the normal schedule.
    warm = re.search(r'(?m)^  num_warmup_epochs: (\d+)\s*$', text)
    if warm and int(warm.group(1)) >= epochs:
        text = re.sub(r'(?m)^  num_warmup_epochs: .*$',
                      '  num_warmup_epochs: {}'.format(max(1, epochs // 10)),
                      text, count=1)
    text = re.sub(r'(?m)^  layers_mp: .*$', '  layers_mp: {}'.format(depth),
                  text, count=1)
    if re.search(r'(?m)^  norm:', text):
        text = re.sub(r'(?m)^  norm: .*$', '  norm: {}'.format(norm),
                      text, count=1)
    else:
        text = text.replace('  residual: True\n',
                            '  residual: True\n  norm: {}\n'.format(norm), 1)
    out_dir = ROOT / 'configs' / 'generated'
    out_dir.mkdir(parents=True, exist_ok=True)
    # A tagged run (e.g. the one-cell smoke test) uses its own result dir,
    # which the aggregator's strict naming pattern deliberately ignores.
    suffix = '_{}'.format(tag) if tag else ''
    path = out_dir / 'zinc_norm_{}_d{}{}.yaml'.format(norm, depth, suffix)
    path.write_text(text, encoding='utf-8')
    return path


def read_stats(path):
    """Read GraphGym's line-delimited stats.json."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def cell_complete(stem, seeds):
    run_root = ROOT / 'results' / 'norm_study' / stem
    for seed in range(seeds):
        stats = read_stats(run_root / str(seed) / 'test' / 'stats.json')
        if not stats:
            return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--norms', nargs='+', default=list(NORMS))
    p.add_argument('--depths', nargs='+', type=int, default=list(DEPTHS))
    p.add_argument('--repeat', type=int, default=3,
                   help='Number of seeds per cell (base seed 0 -> 0..N-1).')
    p.add_argument('--seed', type=int, default=0, help='Base random seed.')
    p.add_argument('--epochs', type=int, default=300,
                   help='optim.max_epoch; original paper uses 2000.')
    p.add_argument('--template', default='configs/gcn/zinc.yaml')
    p.add_argument('--tag', default='',
                   help='Suffix for config/result dirs; tagged runs are '
                        'excluded from the aggregated ZINC table.')
    p.add_argument('--dry-run', action='store_true',
                   help='Only print commands, do not train.')
    p.add_argument('--skip-existing', action='store_true',
                   help='Skip cells whose test stats for all seeds exist.')
    args = p.parse_args()

    log_dir = ROOT / 'logs' / 'norm_study'
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def save_manifest():
        (ROOT / 'experiments' / 'norm_study_manifest.json').write_text(
            json.dumps(manifest, indent=2), encoding='utf-8')

    for norm in args.norms:
        for depth in args.depths:
            cfg_path = make_config(norm, depth, args.epochs, args.template,
                                   args.tag)
            # Store a portable relative command in the manifest; on this
            # machine subprocess still runs the absolute interpreter path.
            cmd = [sys.executable, str(ROOT / 'main.py'),
                   '--cfg', str(cfg_path.relative_to(ROOT)),
                   '--repeat', str(args.repeat), 'seed', str(args.seed)]
            portable_cmd = ' '.join([
                'python', 'main.py', '--cfg',
                str(cfg_path.relative_to(ROOT)).replace('\\', '/'),
                '--repeat', str(args.repeat), 'seed', str(args.seed)])
            stem = cfg_path.stem
            manifest.append({
                'norm': norm, 'depth': depth, 'seeds': args.repeat,
                'config': str(cfg_path.relative_to(ROOT)),
                'result_dir': 'results/norm_study/' + stem,
                'command': portable_cmd,
            })
            if cell_complete(stem, args.repeat) and args.skip_existing:
                print('[skip] {} d={}'.format(norm, depth))
                continue
            print('[run ] {} d={} -> {}'.format(norm, depth, stem))
            print('       ' + ' '.join(cmd))
            if args.dry_run:
                continue
            log_path = log_dir / (stem + '.log')
            started = time.time()
            with log_path.open('w', encoding='utf-8') as fh:
                proc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh,
                                      stderr=subprocess.STDOUT)
            print('       exit={} elapsed={:.1f}s log={}'.format(
                proc.returncode, time.time() - started,
                log_path.relative_to(ROOT)))
            if proc.returncode != 0:
                # Surface the real traceback in the notebook.  Hiding it in a
                # log file is how a failed sweep once produced an empty CSV
                # without anybody noticing.
                tail = log_path.read_text(
                    encoding='utf-8', errors='replace').splitlines()[-40:]
                print('\n----- last {} lines of {} -----'.format(
                    len(tail), log_path.relative_to(ROOT)), file=sys.stderr)
                print('\n'.join(tail), file=sys.stderr)
                print('----- end of log -----\n', file=sys.stderr)
                print('FAILED - {} d={}; aborting sweep.'.format(norm, depth),
                      file=sys.stderr)
                save_manifest()
                return proc.returncode or 1

    save_manifest()
    print('\nWrote experiments/norm_study_manifest.json '
          '({} cells)'.format(len(manifest)))
    if args.dry_run:
        print('Dry run: no training started.')
        return 0
    missing = ['{} d={}'.format(m['norm'], m['depth']) for m in manifest
               if not cell_complete(Path(m['result_dir']).name, m['seeds'])]
    if missing:
        print('\nINCOMPLETE: {}/{} cells produced no test stats:'.format(
            len(missing), len(manifest)), file=sys.stderr)
        for name in missing:
            print('  - ' + name, file=sys.stderr)
        print('Do not aggregate yet: the summary would be empty.',
              file=sys.stderr)
        return 1
    print('All {} cells produced test stats.'.format(len(manifest)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
