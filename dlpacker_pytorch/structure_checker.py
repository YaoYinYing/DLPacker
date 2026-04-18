from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from Bio.PDB import PDBParser, Selection

from dlpacker_pytorch.utils import SIDE_CHAINS, THE20

EXPECTED_BONDS: Dict[str, List[Tuple[str, str]]] = {
    'ALA': [('CA', 'CB')],
    'ARG': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'NE'), ('NE', 'CZ'), ('CZ', 'NH1'), ('CZ', 'NH2')],
    'ASN': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'OD1'), ('CG', 'ND2')],
    'ASP': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'OD1'), ('CG', 'OD2')],
    'CYS': [('CA', 'CB'), ('CB', 'SG')],
    'GLN': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'OE1'), ('CD', 'NE2')],
    'GLU': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'OE1'), ('CD', 'OE2')],
    'HIS': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'ND1'), ('CG', 'CD2'), ('ND1', 'CE1'), ('CD2', 'NE2')],
    'ILE': [('CA', 'CB'), ('CB', 'CG1'), ('CB', 'CG2'), ('CG1', 'CD1')],
    'LEU': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2')],
    'LYS': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD'), ('CD', 'CE'), ('CE', 'NZ')],
    'MET': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'SD'), ('SD', 'CE')],
    'PHE': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2'), ('CD1', 'CE1'), ('CD2', 'CE2'), ('CE1', 'CZ'), ('CE2', 'CZ')],
    'PRO': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD')],
    'SER': [('CA', 'CB'), ('CB', 'OG')],
    'THR': [('CA', 'CB'), ('CB', 'OG1'), ('CB', 'CG2')],
    'TRP': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2'), ('CD1', 'NE1'), ('CD2', 'CE2'), ('CD2', 'CE3'), ('CE2', 'CZ2'), ('CE3', 'CZ3'), ('CZ2', 'CH2'), ('CZ3', 'CH2')],
    'TYR': [('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2'), ('CD1', 'CE1'), ('CD2', 'CE2'), ('CE1', 'CZ'), ('CE2', 'CZ'), ('CZ', 'OH')],
    'VAL': [('CA', 'CB'), ('CB', 'CG1'), ('CB', 'CG2')],
}


@dataclass
class ClashRecord:
    distance: float
    atom_a: Tuple[str, int, str, str]
    atom_b: Tuple[str, int, str, str]


@dataclass
class StructureCheckReport:
    pdb_path: str
    heavy_atom_count: int
    min_inter_residue_distance: float
    p1_inter_residue_distance: float
    p5_inter_residue_distance: float
    missing_sidechain_atoms: List[Tuple[str, int, str, str]]
    bond_length_outliers: List[Tuple[str, int, str, str, str, float]]
    severe_clashes: List[ClashRecord]


@dataclass
class StructureCheckDelta:
    before_min_inter_residue_distance: float
    after_min_inter_residue_distance: float
    worsened_clash_count: int
    new_severe_clashes: List[ClashRecord]


def _load_structure(pdb_path: str):
    parser = PDBParser(QUIET=True)
    return parser.get_structure('checked', pdb_path)


def _heavy_atoms(structure):
    return [a for a in Selection.unfold_entities(structure, 'A') if a.element != 'H']


def _nonbonded_neighbor_stats(atoms) -> Tuple[np.ndarray, np.ndarray]:
    n = len(atoms)
    coords = np.array([a.coord for a in atoms], dtype=np.float32)
    residue_ids = np.array([id(a.get_parent()) for a in atoms])

    nearest = np.full(n, np.inf, dtype=np.float32)
    nearest_idx = np.full(n, -1, dtype=np.int32)

    block = 512
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        d = np.sqrt(np.sum((coords[i0:i1, None, :] - coords[None, :, :]) ** 2, axis=-1), dtype=np.float32)
        same = residue_ids[i0:i1, None] == residue_ids[None, :]
        d[same] = np.inf
        nearest[i0:i1] = d.min(axis=1)
        nearest_idx[i0:i1] = d.argmin(axis=1)

    return nearest, nearest_idx


def _missing_sidechain_atoms(structure) -> List[Tuple[str, int, str, str]]:
    out: List[Tuple[str, int, str, str]] = []
    for residue in Selection.unfold_entities(structure, 'R'):
        name = residue.get_resname()
        if name not in THE20 or name == 'GLY':
            continue
        chain = residue.get_full_id()[2]
        resid = residue.get_id()[1]
        for atom_name in SIDE_CHAINS[name]:
            if not residue.has_id(atom_name):
                out.append((chain, resid, name, atom_name))
    return out


def _bond_length_outliers(
    structure,
    *,
    lower: float = 1.0,
    upper: float = 2.2,
) -> List[Tuple[str, int, str, str, str, float]]:
    out: List[Tuple[str, int, str, str, str, float]] = []
    for residue in Selection.unfold_entities(structure, 'R'):
        name = residue.get_resname()
        if name not in THE20:
            continue
        chain = residue.get_full_id()[2]
        resid = residue.get_id()[1]

        for a1, a2 in [('N', 'CA'), ('CA', 'C')]:
            if residue.has_id(a1) and residue.has_id(a2):
                d = float(np.linalg.norm(residue[a1].coord - residue[a2].coord))
                if d < lower or d > upper:
                    out.append((chain, resid, name, a1, a2, d))

        if name in EXPECTED_BONDS:
            for a1, a2 in EXPECTED_BONDS[name]:
                if residue.has_id(a1) and residue.has_id(a2):
                    d = float(np.linalg.norm(residue[a1].coord - residue[a2].coord))
                    if d < lower or d > upper:
                        out.append((chain, resid, name, a1, a2, d))
    return out


def check_structure(
    pdb_path: str | Path,
    *,
    clash_threshold: float = 1.0,
    top_n_clashes: int = 20,
) -> StructureCheckReport:
    path = str(Path(pdb_path).expanduser().resolve())
    structure = _load_structure(path)
    atoms = _heavy_atoms(structure)
    if not atoms:
        raise ValueError(f'No heavy atoms found in {path}')

    nearest, nearest_idx = _nonbonded_neighbor_stats(atoms)
    min_idx = int(np.argmin(nearest))
    min_d = float(nearest[min_idx])
    p1 = float(np.percentile(nearest, 1))
    p5 = float(np.percentile(nearest, 5))

    severe: List[ClashRecord] = []
    for i, d in enumerate(nearest):
        if float(d) >= clash_threshold:
            continue
        j = int(nearest_idx[i])
        if j < 0:
            continue
        a = atoms[i]
        b = atoms[j]
        if a.get_parent() == b.get_parent():
            continue
        severe.append(
            ClashRecord(
                distance=float(d),
                atom_a=(a.get_full_id()[2], a.get_parent().get_id()[1], a.get_parent().get_resname(), a.get_name()),
                atom_b=(b.get_full_id()[2], b.get_parent().get_id()[1], b.get_parent().get_resname(), b.get_name()),
            )
        )

    severe.sort(key=lambda x: x.distance)
    severe = severe[: max(0, int(top_n_clashes))]

    return StructureCheckReport(
        pdb_path=path,
        heavy_atom_count=len(atoms),
        min_inter_residue_distance=min_d,
        p1_inter_residue_distance=p1,
        p5_inter_residue_distance=p5,
        missing_sidechain_atoms=_missing_sidechain_atoms(structure),
        bond_length_outliers=_bond_length_outliers(structure),
        severe_clashes=severe,
    )


def format_report(report: StructureCheckReport) -> str:
    lines = [
        f'Structure check: {report.pdb_path}',
        f'Heavy atoms: {report.heavy_atom_count}',
        f'Inter-residue nearest distances: min={report.min_inter_residue_distance:.3f}A p1={report.p1_inter_residue_distance:.3f}A p5={report.p5_inter_residue_distance:.3f}A',
        f'Missing sidechain atoms: {len(report.missing_sidechain_atoms)}',
        f'Bond-length outliers: {len(report.bond_length_outliers)}',
        f'Severe clashes (< threshold): {len(report.severe_clashes)}',
    ]
    for chain, resid, resname, atom in report.missing_sidechain_atoms[:20]:
        lines.append(f'  missing: {chain}/{resid}/{resname}/{atom}')
    for chain, resid, resname, a1, a2, d in report.bond_length_outliers[:20]:
        lines.append(
            f'  bond outlier {d:.3f}A: {chain}/{resid}/{resname}/{a1}-{a2}'
        )
    for c in report.severe_clashes[:20]:
        a = c.atom_a
        b = c.atom_b
        lines.append(
            f'  clash {c.distance:.3f}A: {a[0]}/{a[1]}/{a[2]}/{a[3]} -- {b[0]}/{b[1]}/{b[2]}/{b[3]}'
        )
    return '\n'.join(lines)


def compare_reports(
    before: StructureCheckReport,
    after: StructureCheckReport,
    *,
    clash_threshold: float = 1.0,
    top_n: int = 20,
) -> StructureCheckDelta:
    before_pairs = {
        tuple(sorted([c.atom_a, c.atom_b])): c.distance for c in before.severe_clashes
    }
    new = []
    for c in after.severe_clashes:
        key = tuple(sorted([c.atom_a, c.atom_b]))
        old = before_pairs.get(key, np.inf)
        if c.distance < min(float(old), clash_threshold):
            new.append(c)
    new.sort(key=lambda x: x.distance)
    return StructureCheckDelta(
        before_min_inter_residue_distance=before.min_inter_residue_distance,
        after_min_inter_residue_distance=after.min_inter_residue_distance,
        worsened_clash_count=len(new),
        new_severe_clashes=new[: max(0, int(top_n))],
    )


def format_delta(delta: StructureCheckDelta) -> str:
    lines = [
        'Structure check delta:',
        f'  min inter-residue distance: {delta.before_min_inter_residue_distance:.3f}A -> {delta.after_min_inter_residue_distance:.3f}A',
        f'  worsened severe clashes: {delta.worsened_clash_count}',
    ]
    for c in delta.new_severe_clashes:
        a = c.atom_a
        b = c.atom_b
        lines.append(
            f'  new/worse clash {c.distance:.3f}A: {a[0]}/{a[1]}/{a[2]}/{a[3]} -- {b[0]}/{b[1]}/{b[2]}/{b[3]}'
        )
    return '\n'.join(lines)
