from __future__ import annotations

from Bio.PDB import Selection



def _find_residue(dlp, target):
    for residue in Selection.unfold_entities(dlp.structure, 'R'):
        if dlp._get_residue_tuple(residue) == target:
            return residue
    raise AssertionError(f'Residue not found: {target}')


def test_reconstruct_residue_restores_sidechain(dlp):
    target = (1, 'A', 'ALA')
    residue = _find_residue(dlp, target)

    dlp._remove_sidechain(residue)
    assert not residue.has_id('CB')

    dlp.reconstruct_residue(residue, refine_only=False)
    assert residue.has_id('CB')


def test_reconstruct_protein_smoke(dlp):
    dlp.reconstruct_protein(order='sequence')
    assert dlp.reconstructed is not None

    rebuilt = {}
    for residue in Selection.unfold_entities(dlp.reconstructed, 'R'):
        rebuilt[dlp._get_residue_tuple(residue)] = residue

    assert rebuilt[(1, 'A', 'ALA')].has_id('CB')


def test_mutation_and_repack(dlp):
    dlp.mutate_residue((1, 'A', 'ALA'), 'SER')

    residue = _find_residue(dlp, (1, 'A', 'SER'))
    assert residue.get_resname() == 'SER'
    assert residue.has_id('OG')


def test_get_targets_returns_local_region(dlp):
    targets = dlp.get_targets((1, 'A', 'ALA'), radius=2.5)
    assert (1, 'A', 'ALA') in targets
