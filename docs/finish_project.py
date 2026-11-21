#!/usr/bin/env python3
"""One-command finalization for the CS5491 project.

After the ZINC grid has run on Colab and the aggregated CSV has been copied
into ``results/``, this script validates the results, rebuilds both report
PDFs, checks the generated PDFs, and optionally points the local git remote
at the public repository and commits everything.

Usage (from the repository root)::

    # origin already points at the public repository; no URL needed
    python docs/finish_project.py
    python docs/finish_project.py --push
    python docs/finish_project.py --repo-url https://github.com/<user>/<repo>.git
    python docs/finish_project.py --skip-git
    python docs/finish_project.py --csv path/to.csv --out-dir tmp --skip-git

The last form is useful for validating a result file without touching the
official stage folders or the git history. ``--repo-url`` is only needed when
the local ``origin`` still points somewhere else; omitting it keeps the
current remote.

Nothing is pushed unless ``--push`` is given, and no number is ever written
into a report by hand: the report builder reads the aggregated CSV.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent.parent          # course stage folders live here
NORMS = ('batch', 'layer', 'graph', 'pair', 'none')
DEPTHS = (4, 8, 12, 20)
SUMMARY = ROOT / 'results' / 'norm_study_summary.csv'


def check_csv(path):
    """Report which grid cells are present and flag suspicious values."""
    if not path.exists():
        print('ERROR: missing', path)
        print('Copy norm_study_summary.csv from Colab into results/ first.')
        return None
    with path.open(newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    found = set()
    bad = []
    for r in rows:
        try:
            found.add((r['norm'], int(r['depth'])))
            value = float(r['test_mae_mean'])
            if not (value > 0) or value != value:   # non-positive or NaN
                bad.append(r)
        except (KeyError, TypeError, ValueError):
            bad.append(r)
    missing = [(n, d) for n in NORMS for d in DEPTHS if (n, d) not in found]
    seeds = sorted({int(r.get('seeds') or 0) for r in rows})
    print('ZINC summary: {} rows / {} expected cells; seeds per cell: {}'.format(
        len(rows), len(NORMS) * len(DEPTHS), seeds))
    if missing:
        print('  WARNING missing cells (shown as pending in Table 3):',
              ', '.join('{}/d{}'.format(n, d) for n, d in missing))
    if bad:
        print('  WARNING suspicious values in {} rows'.format(len(bad)))
    if not missing and not bad:
        print('  OK: full grid present with positive, finite MAE values.')
    return rows


def run(cmd):
    """Run a command from the repository root, streaming its output."""
    print('\n$', ' '.join(cmd))
    return subprocess.run(cmd, cwd=str(ROOT)).returncode == 0


def check_pdfs(out_dir=None):
    """Verify page counts and that the real results reached the final PDF."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print('\npypdf not installed (pip install pypdf) -> skipping PDF checks.')
        return True
    if out_dir:
        base = Path(out_dir).resolve()
        targets = [(base / 'CS5491_Final_Report.pdf', 4),
                   (base / 'CS5491_Milestone_Report.pdf', 3)]
    else:
        targets = [
            (WORKSPACE / 'Final project' / 'CS5491_Final_Report.pdf', 4),
            (WORKSPACE / 'project milestone' / 'CS5491_Milestone_Report.pdf', 3),
        ]
    ok = True
    for path, expected in targets:
        if not path.exists():
            print('MISSING', path)
            ok = False
            continue
        reader = PdfReader(str(path))
        text = ' '.join((page.extract_text() or '') for page in reader.pages)
        print('\n{}: {} pages (target {})'.format(
            path.name, len(reader.pages), expected))
        if len(reader.pages) > expected + 1:
            print('  WARNING: report is longer than expected')
            ok = False
        if path.name.startswith('CS5491_Final'):
            checks = [
                ('Table 3 has no pending marker', 'pending' not in text.lower()),
                ('Figure 1 is the ZINC curve', 'Figure 1. ZINC test MAE' in text),
                ('data-derived conclusions present', 'At the deepest setting' in text),
            ]
            for label, passed in checks:
                print('  {} {}'.format('OK:' if passed else 'CHECK FAILED:', label))
                ok = ok and passed
    return ok


def git_steps(repo_url, push):
    """Set the remote, commit results, and print (or run) the push command."""
    if repo_url and not run(['git', 'remote', 'set-url', 'origin', repo_url]):
        return False
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=str(ROOT),
                            capture_output=True, text=True)
    if status.stdout.strip():
        if not run(['git', 'add', '-A']):
            return False
        if not run(['git', 'commit', '-m',
                    'add ZINC results and rebuild reports']):
            return False
    else:
        print('\nNo uncommitted changes.')
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT),
                          capture_output=True, text=True).stdout.strip()
    print('\nCommit hash for Canvas:', head)
    if push:
        return run(['git', 'push', '-u', 'origin', 'HEAD'])
    print('Next: git push -u origin HEAD  (then paste URL + hash into Canvas)')
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-url', default=None,
                   help='Public GitHub repository URL; sets git origin.')
    p.add_argument('--push', action='store_true',
                   help='Also push (requires your own git credentials).')
    p.add_argument('--skip-git', action='store_true',
                   help='Only validate results and rebuild the reports.')
    p.add_argument('--csv', default=None,
                   help='Override results/norm_study_summary.csv (for testing).')
    p.add_argument('--out-dir', default=None,
                   help='Write PDFs here instead of the course stage folders.')
    args = p.parse_args()

    csv_path = Path(args.csv).resolve() if args.csv else SUMMARY
    if check_csv(csv_path) is None:
        return 1
    build_cmd = [sys.executable, 'docs/build_reports.py']
    if args.csv:
        build_cmd += ['--zinc-csv', str(csv_path)]
    if args.out_dir:
        build_cmd += ['--out-dir', str(Path(args.out_dir).resolve())]
    if not run(build_cmd):
        return 1
    if not check_pdfs(args.out_dir):
        print('\nPDF checks reported warnings — read them before submitting.')
    if not args.skip_git:
        git_steps(args.repo_url, args.push)
    print('\nDone.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
