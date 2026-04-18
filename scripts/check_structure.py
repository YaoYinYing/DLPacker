#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

from dlpacker_pytorch.structure_checker import (
    check_structure,
    compare_reports,
    format_delta,
    format_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run geometry checks on a PDB file.')
    parser.add_argument('pdb_file', help='Input PDB file path.')
    parser.add_argument(
        '--clash-threshold',
        type=float,
        default=1.0,
        help='Inter-residue heavy-atom distance threshold (A) reported as severe clash.',
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='Max number of severe clashes to print.',
    )
    parser.add_argument(
        '--fail-on-severe',
        action='store_true',
        help='Return non-zero exit code when severe clashes are detected.',
    )
    parser.add_argument(
        '--reference-pdb',
        default='',
        help='Optional reference PDB to compare against and print delta report.',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdb_file = str(Path(args.pdb_file).expanduser().resolve())
    report = check_structure(
        pdb_file,
        clash_threshold=float(args.clash_threshold),
        top_n_clashes=int(args.top_n),
    )
    print(format_report(report), flush=True)
    if args.reference_pdb:
        ref = str(Path(args.reference_pdb).expanduser().resolve())
        before = check_structure(
            ref,
            clash_threshold=float(args.clash_threshold),
            top_n_clashes=int(args.top_n),
        )
        delta = compare_reports(
            before=before,
            after=report,
            clash_threshold=float(args.clash_threshold),
            top_n=int(args.top_n),
        )
        print(format_delta(delta), flush=True)
    if args.fail_on_severe and report.severe_clashes:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
