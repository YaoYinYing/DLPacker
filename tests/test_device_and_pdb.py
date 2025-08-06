import urllib.request
from pathlib import Path

from DLPacker.utils import get_available_device
from DLPacker.dlpacker import DLPacker
from Bio.PDB import Selection


def test_get_available_device_returns_valid():
    device = get_available_device()
    assert device in {"/GPU:0", "/CPU:0"}


def test_dlpacker_reads_1ubq(tmp_path: Path):
    pdb_file = tmp_path / "1UBQ.pdb"
    urllib.request.urlretrieve("https://files.rcsb.org/download/1UBQ.pdb", pdb_file)

    class DummyModel:
        pass

    packer = DLPacker(str_pdb=str(pdb_file), model=DummyModel())
    residues = list(Selection.unfold_entities(packer.structure, "R"))
    assert len(residues) == 76

