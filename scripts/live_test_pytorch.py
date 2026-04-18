#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from dlpacker_pytorch import DLPacker
from dlpacker_pytorch.dlpacker import DEFAULT_WEIGHTS
from dlpacker_pytorch.structure_checker import (
    check_structure,
    compare_reports,
    format_delta,
    format_report,
)
from dlpacker_pytorch.tf_parity import (
    load_tf_model_from_h5,
    postprocess_prediction,
    rank_rotamers,
    tensorflow_available,
)
from dlpacker_pytorch.utils import (
    DLPModel,
    THE20,
    checkpoint_info,
    ensure_pretrained_weights,
    file_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Live smoke test for PyTorch DLPacker reconstruction.'
    )
    parser.add_argument(
        '--pdb-id',
        default='1ubq',
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
    parser.add_argument(
        '--parity-target',
        default='',
        help='Optional residue tuple for TF parity check: resid,chain,resname (e.g. 314,A,PHE).',
    )
    parser.add_argument(
        '--parity-topk',
        type=int,
        default=5,
        help='Top-k size for TF/PT rotamer overlap when --parity-target is set.',
    )
    parser.add_argument(
        '--run-structure-check',
        action='store_true',
        help='Run geometry checker on output PDB and print a report.',
    )
    parser.add_argument(
        '--check-clash-threshold',
        type=float,
        default=1.0,
        help='Severe clash threshold (A) for structure checker.',
    )
    parser.add_argument(
        '--check-fail-on-severe',
        action='store_true',
        help='Exit with non-zero code if severe clashes are found.',
    )
    parser.add_argument(
        '--check-compare-input',
        action='store_true',
        help='Also compare checker report against original input structure.',
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

    pt_path = ensure_pretrained_weights(DEFAULT_WEIGHTS)
    h5_path = f'{DEFAULT_WEIGHTS}.h5'
    info = checkpoint_info(pt_path)
    print('Weight source info:', flush=True)
    print(f'  h5: {h5_path}', flush=True)
    print(f'  h5_sha256: {file_sha256(h5_path)}', flush=True)
    print(f'  pt: {info["path"]}', flush=True)
    print(f'  pt_sha256: {info["sha256"]}', flush=True)
    meta = info.get('meta', {})
    print(f'  converter_version: {meta.get("converter_version", "n/a")}', flush=True)

    dlp.reconstruct_protein(order=args.order, output_filename=str(output_pdb))

    if args.run_structure_check:
        report = check_structure(
            str(output_pdb),
            clash_threshold=float(args.check_clash_threshold),
            top_n_clashes=20,
        )
        print(format_report(report), flush=True)
        if args.check_compare_input:
            before = check_structure(
                str(input_pdb),
                clash_threshold=float(args.check_clash_threshold),
                top_n_clashes=20,
            )
            delta = compare_reports(
                before=before,
                after=report,
                clash_threshold=float(args.check_clash_threshold),
                top_n=20,
            )
            print(format_delta(delta), flush=True)
        if args.check_fail_on_severe and report.severe_clashes:
            raise RuntimeError(
                f'Structure checker found {len(report.severe_clashes)} severe clashes.'
            )

    if args.parity_target:
        resid_s, chain, label = [x.strip() for x in args.parity_target.split(',')]
        resid = int(resid_s)
        target = (resid, chain, label)
        residue = None
        for r in dlp.structure.get_residues():
            if dlp._get_residue_tuple(r) == target:
                residue = r
                break
        if residue is None:
            raise ValueError(f'Parity target residue not found: {target}')
        if not tensorflow_available():
            raise RuntimeError('TensorFlow is not available for parity check.')

        box = dlp._genetare_input_box(residue, True)
        if not box:
            raise RuntimeError(f'Failed to generate input box for target {target}')
        x, _ = dlp.input_reader(box)
        labels = np.zeros((1, 20), dtype=np.float32)
        labels[0, THE20[label]] = 1.0

        tf_model = load_tf_model_from_h5(h5_path)
        y_pt = dlp.model.model([x[None, ...], labels]).numpy()[0]
        y_tf = tf_model([x[None, ...], labels]).numpy()[0]
        mae = float(np.mean(np.abs(y_pt - y_tf)))
        maxae = float(np.max(np.abs(y_pt - y_tf)))

        pp_pt = postprocess_prediction(y_pt, x, label)
        pp_tf = postprocess_prediction(y_tf, x, label)
        pp_mae = float(np.mean(np.abs(pp_pt - pp_tf)))

        r_pt = rank_rotamers(dlp.library['grids'][label], pp_pt, top_k=args.parity_topk)
        r_tf = rank_rotamers(dlp.library['grids'][label], pp_tf, top_k=args.parity_topk)
        overlap = len(set(r_pt.topk_idx).intersection(r_tf.topk_idx))

        print('TF parity probe:', flush=True)
        print(f'  target: {target}', flush=True)
        print(f'  output_mae: {mae:.6e}', flush=True)
        print(f'  output_maxae: {maxae:.6e}', flush=True)
        print(f'  postprocess_mae: {pp_mae:.6e}', flush=True)
        print(f'  best_idx_pt: {r_pt.best_idx}', flush=True)
        print(f'  best_idx_tf: {r_tf.best_idx}', flush=True)
        print(f'  top{args.parity_topk}_overlap: {overlap}', flush=True)

    elapsed = time.time() - t0
    print(f'Live test passed in {elapsed:.2f}s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
