from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from Bio.PDB import Selection

from DLPacker.dlpacker import DLPacker
from DLPacker.utils import InputBoxReader


class _IdentityPredictor:
    def __call__(self, inputs):
        x, _labels = inputs
        x = np.asarray(x, dtype=np.float32)
        return torch.from_numpy(x[..., :4].copy())


class DummyModel:
    def __init__(self):
        self.model = _IdentityPredictor()


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / 'DLPacker' / 'data'


@pytest.fixture
def sample_pdb(tmp_path: Path) -> Path:
    pdb = tmp_path / 'sample.pdb'
    pdb.write_text(
        """
ATOM      1  N   ALA A   1      -3.274  -1.500  -1.169  1.00 27.30           N
ATOM      2  CA  ALA A   1      -2.468  -1.500   0.000  1.00 27.66           C
ATOM      3  C   ALA A   1      -3.379  -1.500   1.222  1.00 28.10           C
ATOM      4  O   ALA A   1      -4.000  -1.500   2.100  1.00 28.10           O
ATOM      5  CB  ALA A   1      -1.300  -2.300  -0.200  1.00 20.00           C
ATOM      6  N   GLY A   2      -3.400  -1.500   2.500  1.00 27.30           N
ATOM      7  CA  GLY A   2      -4.100  -1.500   3.700  1.00 27.66           C
ATOM      8  C   GLY A   2      -3.300  -1.500   4.900  1.00 28.10           C
ATOM      9  O   GLY A   2      -2.200  -1.500   4.900  1.00 28.10           O
ATOM     10  N   SER A   3      -3.900  -1.500   6.100  1.00 27.30           N
ATOM     11  CA  SER A   3      -3.300  -1.500   7.400  1.00 27.66           C
ATOM     12  C   SER A   3      -4.100  -1.500   8.600  1.00 28.10           C
ATOM     13  O   SER A   3      -5.300  -1.500   8.600  1.00 28.10           O
ATOM     14  CB  SER A   3      -2.100  -2.300   7.300  1.00 20.00           C
ATOM     15  OG  SER A   3      -1.300  -1.700   6.200  1.00 20.00           O
""".strip()
        + "\n"
    )
    return pdb


@pytest.fixture
def dlp(sample_pdb: Path, data_dir: Path) -> DLPacker:
    return DLPacker(
        str_pdb=str(sample_pdb),
        model=DummyModel(),
        charges_filename=str(data_dir / 'charges.rtp'),
    )


@pytest.fixture
def residues(dlp: DLPacker):
    out = {}
    for residue in Selection.unfold_entities(dlp.structure, 'R'):
        out[dlp._get_residue_tuple(residue)] = residue
    return out
