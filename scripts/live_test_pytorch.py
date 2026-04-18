#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

from dlpacker_pytorch import DLPacker
from dlpacker_pytorch.utils import DLPModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Live smoke test for PyTorch DLPacker reconstruction.'
    )
    parser.add_argument(
        '--pdb-id',
        default='4p0d',
        help='RCSB PDB id to download (ignored if --pdb-file is provided).',
    )
    parser.add_argument(
        '--pdb-file',
        default='',
        help='Optional local PDB file path to use instead of downloading.',
    )
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'mps', 'cuda'],
        help='Torch device passed to DLPModel.',
    )
    parser.add_argument(
        '--order',
        default='sequence',
        choices=['sequence', 'natoms', 'score'],
        help='Reconstruction order heuristic.',
    )
    parser.add_argument(
        '--out',
        default='',
        help='Optional output file path. Default: <input_stem>_live_test.pdb',
    )
    return parser.parse_args()


def resolve_input_pdb(args: argparse.Namespace) -> Path:
    if args.pdb_file:
        in_path = Path(args.pdb_file).expanduser().resolve()
        if not in_path.exists():
            raise FileNotFoundError(f'Input file not found: {in_path}')
        return in_path

    pdb_id = args.pdb_id.lower()
    out = Path(f'{pdb_id}.pdb').resolve()
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    print(f'Downloading {pdb_id} from {url}', flush=True)
    urllib.request.urlretrieve(url, str(out))
    return out


def main() -> int:
    # Make logs visible promptly even when stdout is buffered (CI, wrappers).
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    t0 = time.time()

    input_pdb = resolve_input_pdb(args)
    output_pdb = (
        Path(args.out).expanduser().resolve()
        if args.out
        else input_pdb.with_name(f'{input_pdb.stem}_live_test.pdb')
    )

    print(f'Input:  {input_pdb}', flush=True)
    print(f'Output: {output_pdb}', flush=True)
    print(f'Device: {args.device}', flush=True)

    model = DLPModel(device=args.device)
    dlp = DLPacker(str(input_pdb), model=model)
    dlp.reconstruct_protein(order=args.order, output_filename=str(output_pdb))

    elapsed = time.time() - t0
    print(f'Live test passed in {elapsed:.2f}s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
