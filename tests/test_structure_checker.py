from __future__ import annotations

from pathlib import Path

from dlpacker_pytorch.structure_checker import check_structure, compare_reports


def test_structure_checker_reports_clean_fixture(data_dir: Path):
    fixture = data_dir.parents[1] / 'tests' / 'fixtures' / 'phe_region.pdb'
    report = check_structure(str(fixture), clash_threshold=0.8, top_n_clashes=10)
    assert report.heavy_atom_count > 0
    assert report.min_inter_residue_distance > 0.8
    assert report.bond_length_outliers == []
    assert report.severe_clashes == []


def test_structure_checker_detects_severe_overlap(tmp_path: Path):
    pdb = tmp_path / 'overlap.pdb'
    pdb.write_text(
        (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
            "ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 20.00           C\n"
            "ATOM      3  C   ALA A   1       2.900   0.000   0.000  1.00 20.00           C\n"
            "ATOM      4  O   ALA A   1       3.900   0.000   0.000  1.00 20.00           O\n"
            "ATOM      5  N   GLY A   2       1.460   0.010   0.000  1.00 20.00           N\n"
            "ATOM      6  CA  GLY A   2       2.910   0.010   0.000  1.00 20.00           C\n"
            "ATOM      7  C   GLY A   2       4.360   0.010   0.000  1.00 20.00           C\n"
            "ATOM      8  O   GLY A   2       5.360   0.010   0.000  1.00 20.00           O\n"
            "END\n"
        )
    )
    report = check_structure(str(pdb), clash_threshold=1.0, top_n_clashes=10)
    assert report.min_inter_residue_distance < 0.1
    assert len(report.severe_clashes) > 0


def test_compare_reports_tracks_worsened_clashes(tmp_path: Path):
    before_pdb = tmp_path / 'before.pdb'
    after_pdb = tmp_path / 'after.pdb'
    before_pdb.write_text(
        (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
            "ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 20.00           C\n"
            "ATOM      3  N   GLY A   2       4.000   0.000   0.000  1.00 20.00           N\n"
            "ATOM      4  CA  GLY A   2       5.450   0.000   0.000  1.00 20.00           C\n"
            "END\n"
        )
    )
    after_pdb.write_text(
        (
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
            "ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 20.00           C\n"
            "ATOM      3  N   GLY A   2       1.520   0.030   0.000  1.00 20.00           N\n"
            "ATOM      4  CA  GLY A   2       2.970   0.030   0.000  1.00 20.00           C\n"
            "END\n"
        )
    )
    before = check_structure(str(before_pdb), clash_threshold=1.0, top_n_clashes=10)
    after = check_structure(str(after_pdb), clash_threshold=1.0, top_n_clashes=10)
    delta = compare_reports(before, after, clash_threshold=1.0, top_n=10)
    assert delta.after_min_inter_residue_distance < delta.before_min_inter_residue_distance
    assert delta.worsened_clash_count > 0


def test_structure_checker_detects_bond_outlier(tmp_path: Path):
    pdb = tmp_path / 'bond_outlier.pdb'
    pdb.write_text(
        (
            "ATOM      1  N   SER A   1       0.000   0.000   0.000  1.00 20.00           N\n"
            "ATOM      2  CA  SER A   1       1.450   0.000   0.000  1.00 20.00           C\n"
            "ATOM      3  C   SER A   1       2.900   0.000   0.000  1.00 20.00           C\n"
            "ATOM      4  CB  SER A   1       5.000   0.000   0.000  1.00 20.00           C\n"
            "ATOM      5  OG  SER A   1       6.450   0.000   0.000  1.00 20.00           O\n"
            "ATOM      6  N   GLY A   2       4.300   0.000   0.000  1.00 20.00           N\n"
            "ATOM      7  CA  GLY A   2       5.700   0.000   0.000  1.00 20.00           C\n"
            "END\n"
        )
    )
    report = check_structure(str(pdb), clash_threshold=0.8, top_n_clashes=10)
    assert any(x[3] == 'CA' and x[4] == 'CB' for x in report.bond_length_outliers)
