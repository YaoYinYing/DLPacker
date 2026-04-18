#!/usr/bin/env python

from __future__ import annotations

import argparse
import contextlib
import io
import statistics
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
from Bio.PDB import Selection

from dlpacker_pytorch import DLPacker
from dlpacker_pytorch.utils import DLPModel, THE20


# 10-mutation live-test plan for 1UBQ (resid -> new amino acid)
MUTATION_PLAN = [
    (3, 'VAL'),
    (7, 'SER'),
    (14, 'ASP'),
    (22, 'ALA'),
    (28, 'GLU'),
    (35, 'LEU'),
    (44, 'TYR'),
    (52, 'ASN'),
    (63, 'MET'),
    (71, 'PHE'),
]


@dataclass
class BenchResult:
    scenario: str
    device: str
    status: str
    elapsed_s: float
    output_pdb: str
    details: str
    run_index: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Live mutation benchmark on 1UBQ across CPU/MPS.'
    )
    parser.add_argument('--pdb-id', default='1ubq', help='RCSB PDB id to use.')
    parser.add_argument('--chain', default='A', help='Chain id used for mutations.')
    parser.add_argument(
        '--devices',
        default='cpu,mps',
        help='Comma-separated devices to benchmark (e.g. cpu,mps).',
    )
    parser.add_argument(
        '--order',
        default='sequence',
        choices=['sequence', 'natoms', 'score'],
        help='Reconstruction order heuristic.',
    )
    parser.add_argument(
        '--benchmark-mode',
        default='both',
        choices=['full', 'partial', 'both'],
        help='Benchmark full reconstruction, partial radius sweep, or both.',
    )
    parser.add_argument(
        '--partial-radius-min',
        type=float,
        default=0.0,
        help='Minimum radius for partial reconstruction benchmark (angstrom).',
    )
    parser.add_argument(
        '--partial-radius-max',
        type=float,
        default=10.0,
        help='Maximum radius for partial reconstruction benchmark (angstrom).',
    )
    parser.add_argument(
        '--partial-radius-step',
        type=float,
        default=1.0,
        help='Step for partial radius sweep (angstrom).',
    )
    parser.add_argument(
        '--outdir',
        default='.',
        help='Directory for benchmark output PDB files.',
    )
    parser.add_argument(
        '--repeats',
        type=int,
        default=1,
        help='Number of measured runs per scenario/device.',
    )
    parser.add_argument(
        '--warmup',
        action='store_true',
        help='Run an extra warmup pass before measured repeats.',
    )
    return parser.parse_args()


def ensure_line_buffered() -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)


def download_pdb(pdb_id: str, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    pdb_id = pdb_id.lower()
    out = outdir / f'{pdb_id}.pdb'
    if out.exists():
        print(f'Using cached PDB: {out}', flush=True)
        return out

    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    print(f'Downloading {pdb_id} from {url}', flush=True)
    urllib.request.urlretrieve(url, str(out))
    return out


def device_available(device: str) -> bool:
    if device == 'cpu':
        return True
    if device == 'mps':
        return torch.backends.mps.is_available()
    if device == 'cuda':
        return torch.cuda.is_available()
    return False


def build_targets(dlp: DLPacker, chain: str) -> list[tuple[tuple[int, str, str], str]]:
    residues = {}
    for residue in Selection.unfold_entities(dlp.structure, 'R'):
        resid = residue.get_id()[1]
        segid = residue.get_full_id()[2]
        resname = residue.get_resname()
        residues[(resid, segid)] = resname

    targets = []
    for resid, new_label in MUTATION_PLAN:
        key = (resid, chain)
        if key not in residues:
            raise ValueError(f'Residue {key} is not present in structure.')
        old_label = residues[key]
        if old_label not in THE20:
            raise ValueError(f'Residue {key} has non-canonical type: {old_label}')
        if new_label not in THE20:
            raise ValueError(f'Unsupported mutation target: {new_label}')
        targets.append(((resid, chain, old_label), new_label))
    return targets


def apply_mutations(dlp: DLPacker, chain: str) -> list[tuple[tuple[int, str, str], str]]:
    targets = build_targets(dlp, chain=chain)
    print(f'Applying {len(targets)} mutations on chain {chain}...', flush=True)
    mutated_targets: list[tuple[tuple[int, str, str], str]] = []
    for target, new_label in targets:
        residue = dlp.mutate_sequence(target, new_label)
        if residue is None:
            raise RuntimeError(f'Failed to mutate residue {target} -> {new_label}')
        mutated_targets.append(((target[0], target[1], new_label), new_label))
    return mutated_targets


def verify_mutations(dlp: DLPacker, mutated_targets: list[tuple[tuple[int, str, str], str]]) -> None:
    reconstructed = {}
    for residue in Selection.unfold_entities(dlp.reconstructed, 'R'):
        reconstructed[(residue.get_id()[1], residue.get_full_id()[2])] = residue.get_resname()

    for target, new_label in mutated_targets:
        key = (target[0], target[1])
        if reconstructed.get(key) != new_label:
            raise RuntimeError(
                f'Post-check failed for {key}: expected {new_label}, got {reconstructed.get(key)}'
            )


def run_full_reconstruction(
    dlp: DLPacker,
    mutated_targets: list[tuple[tuple[int, str, str], str]],
    order: str,
    output_pdb: Path,
) -> str:
    with contextlib.redirect_stdout(io.StringIO()):
        dlp.reconstruct_protein(order=order, output_filename=str(output_pdb))
    verify_mutations(dlp, mutated_targets)
    return f'{len(mutated_targets)} mutations completed (full)'


def run_partial_reconstruction(
    dlp: DLPacker,
    mutated_targets: list[tuple[tuple[int, str, str], str]],
    order: str,
    output_pdb: Path,
    radius: float,
) -> str:
    # Reconstruct only residues within radius around each mutated residue.
    # Always include mutated residues themselves so radius=0 remains meaningful.
    region = set(t for t, _ in mutated_targets)
    for target, _ in mutated_targets:
        local = dlp.get_targets(target=target, radius=radius)
        region.update(local)

    with contextlib.redirect_stdout(io.StringIO()):
        dlp.reconstruct_region(
            targets=sorted(region),
            order=order,
            output_filename=str(output_pdb),
        )
    verify_mutations(dlp, mutated_targets)
    return f'{len(mutated_targets)} mutations, region_size={len(region)}, radius={radius:g}'


def run_one(
    pdb_path: Path,
    chain: str,
    device: str,
    order: str,
    outdir: Path,
    scenario: str,
    radius: float | None,
) -> BenchResult:
    scenario_label = scenario if radius is None else f'partial_r{radius:g}'

    if not device_available(device):
        return BenchResult(
            scenario=scenario_label,
            device=device,
            status='SKIPPED',
            elapsed_s=0.0,
            output_pdb='',
            details=f"Device '{device}' unavailable in this runtime.",
        )

    if scenario == 'full':
        output_pdb = outdir / f'{pdb_path.stem}_10mut_full_{device}.pdb'
    elif scenario == 'partial':
        if radius is None:
            raise ValueError('radius must be provided for partial scenario')
        output_pdb = outdir / f'{pdb_path.stem}_10mut_partial_r{radius:g}_{device}.pdb'
    else:
        raise ValueError(f'Unknown scenario: {scenario}')

    t0 = time.time()
    model = DLPModel(device=device)
    dlp = DLPacker(str(pdb_path), model=model)

    print(f'[{device}][{scenario}] setup & mutations...', flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        mutated_targets = apply_mutations(dlp, chain=chain)

    if scenario == 'full':
        print(f'[{device}][full] running reconstruction ({order})...', flush=True)
        details = run_full_reconstruction(
            dlp=dlp,
            mutated_targets=mutated_targets,
            order=order,
            output_pdb=output_pdb,
        )
    else:
        print(
            f'[{device}][partial r={radius:g}] running region reconstruction ({order})...',
            flush=True,
        )
        details = run_partial_reconstruction(
            dlp=dlp,
            mutated_targets=mutated_targets,
            order=order,
            output_pdb=output_pdb,
            radius=float(radius),
        )

    elapsed = time.time() - t0
    return BenchResult(
        scenario=scenario_label,
        device=device,
        status='OK',
        elapsed_s=elapsed,
        output_pdb=str(output_pdb),
        details=details,
    )


def _frange(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError('--partial-radius-step must be > 0')
    vals = []
    cur = start
    while cur <= stop + 1e-9:
        vals.append(round(cur, 6))
        cur += step
    return vals


def print_summary(results: list[BenchResult]) -> None:
    print('\nBenchmark summary', flush=True)
    print('scenario\tdevice\trun\tstatus\telapsed_s\toutput\tdetails', flush=True)
    for r in results:
        print(
            f'{r.scenario}\t{r.device}\t{r.run_index}\t{r.status}\t{r.elapsed_s:.3f}\t{r.output_pdb}\t{r.details}',
            flush=True,
        )

    # Aggregate measured runs (median) for each scenario/device.
    grouped: dict[tuple[str, str], list[BenchResult]] = {}
    for r in results:
        grouped.setdefault((r.scenario, r.device), []).append(r)
    print('\nMedian summary', flush=True)
    print('scenario\tdevice\tstatus\tmedian_elapsed_s\tnruns', flush=True)
    for (scenario, device), rows in sorted(grouped.items()):
        oks = [x.elapsed_s for x in rows if x.status == 'OK']
        if oks:
            med = statistics.median(oks)
            print(f'{scenario}\t{device}\tOK\t{med:.3f}\t{len(oks)}', flush=True)
        else:
            status = rows[0].status if rows else 'SKIPPED'
            print(f'{scenario}\t{device}\t{status}\t0.000\t0', flush=True)

    # CPU/MPS speedup by scenario when both succeeded.
    scenarios = sorted({r.scenario for r in results})
    for scenario in scenarios:
        cpu_vals = [
            r.elapsed_s
            for r in results
            if r.scenario == scenario and r.device == 'cpu' and r.status == 'OK'
        ]
        mps_vals = [
            r.elapsed_s
            for r in results
            if r.scenario == scenario and r.device == 'mps' and r.status == 'OK'
        ]
        if cpu_vals and mps_vals:
            cpu_med = statistics.median(cpu_vals)
            mps_med = statistics.median(mps_vals)
            if mps_med <= 0:
                continue
            speedup = cpu_med / mps_med
            print(f'{scenario}: CPU/MPS speedup = {speedup:.2f}x', flush=True)

    print(
        f"MPS diagnostics: built={torch.backends.mps.is_built()} available={torch.backends.mps.is_available()}",
        flush=True,
    )


def main() -> int:
    ensure_line_buffered()
    args = parse_args()

    devices = [d.strip().lower() for d in args.devices.split(',') if d.strip()]
    if not devices:
        raise ValueError('No devices requested. Use --devices cpu,mps')
    if args.repeats < 1:
        raise ValueError('--repeats must be >= 1')

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    pdb_path = download_pdb(args.pdb_id, outdir=outdir)

    tasks: list[tuple[str, float | None]] = []
    if args.benchmark_mode in ('full', 'both'):
        tasks.append(('full', None))
    if args.benchmark_mode in ('partial', 'both'):
        radii = _frange(args.partial_radius_min, args.partial_radius_max, args.partial_radius_step)
        for r in radii:
            tasks.append(('partial', r))

    results: list[BenchResult] = []
    for scenario, radius in tasks:
        scenario_name = scenario if radius is None else f'partial r={radius:g}'
        for device in devices:
            total_runs = args.repeats + (1 if args.warmup else 0)
            for run_idx in range(1, total_runs + 1):
                is_warmup = args.warmup and run_idx == 1
                label = 'warmup' if is_warmup else f'run {run_idx - (1 if args.warmup else 0)}'
                print(
                    f'\n=== Benchmark {scenario_name} on device: {device} ({label}) ===',
                    flush=True,
                )
                try:
                    result = run_one(
                        pdb_path=pdb_path,
                        chain=args.chain,
                        device=device,
                        order=args.order,
                        outdir=outdir,
                        scenario=scenario,
                        radius=radius,
                    )
                except Exception as exc:
                    result = BenchResult(
                        scenario=scenario_name.replace(' ', '_'),
                        device=device,
                        status='FAILED',
                        elapsed_s=0.0,
                        output_pdb='',
                        details=str(exc),
                    )
                # Warmup results are intentionally not included in final stats.
                if is_warmup:
                    continue
                result.run_index = run_idx - (1 if args.warmup else 0)
                results.append(result)

    print_summary(results)

    if any(r.status == 'FAILED' for r in results):
        return 1
    if all(r.status == 'SKIPPED' for r in results):
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
