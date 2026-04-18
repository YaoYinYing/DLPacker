from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
import torch
from Bio.PDB import Selection

from dlpacker_pytorch.dlpacker import DLPacker


class _IdentityPredictor:
    def __call__(self, inputs):
        x, _labels = inputs
        x = np.asarray(x, dtype=np.float32)
        return torch.from_numpy(x[..., :4].copy())


class _DummyModel:
    def __init__(self):
        self.model = _IdentityPredictor()


def _find_residue(dlp: DLPacker, target):
    for residue in Selection.unfold_entities(dlp.structure, 'R'):
        if dlp._get_residue_tuple(residue) == target:
            return residue
    raise AssertionError(f'Residue not found: {target}')


def _find_residue_in_structure(dlp: DLPacker, structure, target):
    for residue in Selection.unfold_entities(structure, 'R'):
        if dlp._get_residue_tuple(residue) == target:
            return residue
    raise AssertionError(f'Residue not found: {target}')


@pytest.fixture
def dlp_phe(data_dir: Path) -> DLPacker:
    fixture = data_dir.parents[1] / 'tests' / 'fixtures' / 'phe_region.pdb'
    return DLPacker(
        str_pdb=str(fixture),
        model=_DummyModel(),
        charges_filename=str(data_dir / 'charges.rtp'),
    )


def test_phe_reconstruction_has_reasonable_bond_lengths(dlp_phe: DLPacker):
    target = (314, 'A', 'PHE')
    residue = _find_residue(dlp_phe, target)
    dlp_phe._remove_sidechain(residue)
    dlp_phe.reconstruct_residue(residue, refine_only=False)

    bonded_pairs = [
        ('CB', 'CG'),
        ('CG', 'CD1'),
        ('CG', 'CD2'),
        ('CD1', 'CE1'),
        ('CD2', 'CE2'),
        ('CE1', 'CZ'),
        ('CE2', 'CZ'),
    ]
    for a1, a2 in bonded_pairs:
        d = np.linalg.norm(residue[a1].coord - residue[a2].coord)
        assert 1.0 < d < 2.1, f'Unreasonable bond length {a1}-{a2}: {d:.3f}'


def test_phe_neighborhood_nonbonded_min_distance_is_sane(dlp_phe: DLPacker):
    dlp_phe.reconstruct_protein(order='sequence')
    target = (314, 'A', 'PHE')
    residue = _find_residue_in_structure(dlp_phe, dlp_phe.reconstructed, target)
    center = np.mean(np.array([atom.coord for atom in residue]), axis=0)

    nearby = []
    for atom in Selection.unfold_entities(dlp_phe.reconstructed, 'A'):
        if atom.element == 'H':
            continue
        if np.linalg.norm(atom.coord - center) <= 6.0:
            nearby.append(atom)

    min_nonbonded = float('inf')
    for a1, a2 in itertools.combinations(nearby, 2):
        # Ignore pairs from same residue that are directly bonded by name.
        if a1.get_parent() == a2.get_parent():
            continue
        d = float(np.linalg.norm(a1.coord - a2.coord))
        min_nonbonded = min(min_nonbonded, d)

    assert min_nonbonded > 0.8
