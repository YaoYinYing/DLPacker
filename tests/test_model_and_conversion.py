from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from DLPacker.utils import DLPModel, Generator3D, convert_keras_h5_to_pt


def _write_fake_keras_h5(path: Path, model: Generator3D) -> None:
    state = model.state_dict()
    with h5py.File(path, 'w') as f:
        root = f.create_group('model_weights')

        dense = root.create_group('dense')
        dense.create_dataset('kernel:0', data=state['label_fc.weight'].cpu().numpy().T)
        dense.create_dataset('bias:0', data=state['label_fc.bias'].cpu().numpy())

        for i, layer_name in enumerate(model.ordered_conv_layers()):
            g = root.create_group(f'conv3d_{i}')
            w = state[f'{layer_name}.weight'].cpu().numpy()
            b = state[f'{layer_name}.bias'].cpu().numpy()
            g.create_dataset('kernel:0', data=np.transpose(w, (2, 3, 4, 1, 0)))
            g.create_dataset('bias:0', data=b)


def test_model_forward_cpu_shape():
    model = DLPModel(grid_size=8, nres=1, num_channels=27, width=4, device='cpu')
    x = np.random.rand(2, 8, 8, 8, 27).astype(np.float32)
    labels = np.zeros((2, 20), dtype=np.float32)
    labels[:, 0] = 1.0

    out = model.model([x, labels]).numpy()
    assert out.shape == (2, 8, 8, 8, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_model_forward_cuda_shape():
    model = DLPModel(grid_size=8, nres=1, num_channels=27, width=4, device='cuda')
    x = np.random.rand(1, 8, 8, 8, 27).astype(np.float32)
    labels = np.zeros((1, 20), dtype=np.float32)
    labels[:, 1] = 1.0

    out = model.model([x, labels]).numpy()
    assert out.shape == (1, 8, 8, 8, 4)


def test_weight_conversion_roundtrip(tmp_path: Path):
    width = 2
    nres = 1
    grid_size = 4
    num_channels = 3

    source = Generator3D(width=width, nres=nres, grid_size=grid_size, num_channels=num_channels)
    for p in source.parameters():
        torch.nn.init.uniform_(p, a=-0.2, b=0.2)

    h5_path = tmp_path / 'weights.h5'
    pt_path = tmp_path / 'weights.pt'
    _write_fake_keras_h5(h5_path, source)

    convert_keras_h5_to_pt(
        keras_h5_path=str(h5_path),
        out_pt_path=str(pt_path),
        width=width,
        nres=nres,
        grid_size=grid_size,
        num_channels=num_channels,
    )

    checkpoint = torch.load(pt_path, map_location='cpu')
    converted_state = checkpoint['state_dict']
    src_state = source.state_dict()

    assert set(converted_state.keys()) == set(src_state.keys())
    for key in src_state:
        assert converted_state[key].shape == src_state[key].shape
        assert torch.allclose(converted_state[key], src_state[key], atol=1e-6, rtol=1e-6)


def test_conversion_parity_output(tmp_path: Path):
    width = 2
    nres = 1
    grid_size = 8
    num_channels = 5

    baseline = Generator3D(width=width, nres=nres, grid_size=grid_size, num_channels=num_channels)
    for p in baseline.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=0.1)

    h5_path = tmp_path / 'weights.h5'
    pt_path = tmp_path / 'weights.pt'
    _write_fake_keras_h5(h5_path, baseline)

    convert_keras_h5_to_pt(
        keras_h5_path=str(h5_path),
        out_pt_path=str(pt_path),
        width=width,
        nres=nres,
        grid_size=grid_size,
        num_channels=num_channels,
    )

    converted = Generator3D(width=width, nres=nres, grid_size=grid_size, num_channels=num_channels)
    converted.load_state_dict(torch.load(pt_path, map_location='cpu')['state_dict'])

    x = torch.randn(2, num_channels, grid_size, grid_size, grid_size)
    labels = torch.zeros(2, 20)
    labels[:, 4] = 1.0

    with torch.no_grad():
        y_ref = baseline(x, labels)
        y_new = converted(x, labels)

    mae = torch.mean(torch.abs(y_ref - y_new)).item()
    assert mae < 1e-6


def test_import_without_tensorflow_dependency():
    assert importlib.util.find_spec('tensorflow') is None
    import DLPacker  # noqa: F401
