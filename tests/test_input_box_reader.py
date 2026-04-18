from __future__ import annotations

import numpy as np

from dlpacker_pytorch.utils import InputBoxReader, THE20


def test_input_box_reader_shapes_and_channels(data_dir):
    reader = InputBoxReader(charges_filename=str(data_dir / 'charges.rtp'))

    box = {
        'target': {'id': 1, 'segid': 'A', 'name': 'ALA', 'atomids': [1]},
        'types': np.array(['C', 'C', 'N'], dtype=object),
        'resnames': np.array(['ALA', 'ALA', 'GLY'], dtype=object),
        'segids': np.array(['A', 'A', 'A'], dtype=object),
        'positions': np.array([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        'names': np.array(['C', 'CB', 'N'], dtype=object),
        'resids': np.array([1, 1, 2], dtype=np.int32),
    }

    x, y = reader(box)

    assert x.shape == (40, 40, 40, 27)
    assert y.shape == (40, 40, 40, 4)
    assert x.dtype == np.float32
    assert y.dtype == np.float32

    assert float(x[..., 0].sum()) > 0.0  # carbon input
    assert float(x[..., 1].sum()) > 0.0  # nitrogen input
    assert float(y[..., 0].sum()) > 0.0  # target carbon contribution

    ala_bb_channel = 6 + THE20['ALA']
    assert float(x[..., ala_bb_channel].sum()) > 0.0
