from __future__ import annotations

import numpy as np
import pytest
from Bio.PDB import Selection

from dlpacker_pytorch.dlpacker import DLPacker, DEFAULT_WEIGHTS
from dlpacker_pytorch.tf_parity import (
    load_tf_model_from_h5,
    postprocess_prediction,
    rank_rotamers,
    tensorflow_available,
)
from dlpacker_pytorch.utils import DLPModel, THE20, ensure_pretrained_weights


pytestmark = pytest.mark.tf_parity


def _find_residue(dlp: DLPacker, target: tuple[int, str, str]):
    for residue in Selection.unfold_entities(dlp.structure, 'R'):
        if dlp._get_residue_tuple(residue) == target:
            return residue
    raise AssertionError(f'Residue not found: {target}')


@pytest.mark.skipif(not tensorflow_available(), reason='TensorFlow not available')
def test_tf_pytorch_volume_prediction_and_rotamer_parity(data_dir):
    ensure_pretrained_weights(DEFAULT_WEIGHTS)
    h5_path = f'{DEFAULT_WEIGHTS}.h5'

    model = DLPModel(device='cpu')
    dlp = DLPacker(
        str_pdb=str(data_dir.parents[1] / 'tests' / 'fixtures' / 'phe_region.pdb'),
        model=model,
    )
    tf_model = load_tf_model_from_h5(h5_path)

    target = (314, 'A', 'PHE')
    residue = _find_residue(dlp, target)
    box = dlp._genetare_input_box(residue, True)
    assert box
    x, _ = dlp.input_reader(box)
    labels = np.zeros((1, 20), dtype=np.float32)
    labels[0, THE20['PHE']] = 1.0

    y_pt = dlp.model.model([x[None, ...], labels]).numpy()[0]
    y_tf = tf_model([x[None, ...], labels]).numpy()[0]

    mae = float(np.mean(np.abs(y_pt - y_tf)))
    maxae = float(np.max(np.abs(y_pt - y_tf)))
    assert mae < 3e-4
    assert maxae < 5e-3

    pp_pt = postprocess_prediction(y_pt, x, 'PHE')
    pp_tf = postprocess_prediction(y_tf, x, 'PHE')
    pp_mae = float(np.mean(np.abs(pp_pt - pp_tf)))
    assert pp_mae < 3e-4

    rank_pt = rank_rotamers(dlp.library['grids']['PHE'], pp_pt, top_k=5)
    rank_tf = rank_rotamers(dlp.library['grids']['PHE'], pp_tf, top_k=5)
    assert rank_pt.best_idx == rank_tf.best_idx
    assert len(set(rank_pt.topk_idx).intersection(rank_tf.topk_idx)) >= 4
