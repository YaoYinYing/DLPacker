# Copyright 2021 (c) Mikita Misiura
#
# This code is part of DLPacker. If you use it in your work, please cite the
# following paper:
#
# @article {Misiura2021.05.23.445347,
#     author = {Misiura, Mikita and Shroff, Raghav and Thyer, Ross and Kolomeisky, Anatoly},
#     title = {DLPacker: Deep Learning for Prediction of Amino Acid Side Chain Conformations in Proteins},
#     elocation-id = {2021.05.23.445347},
#     year = {2021},
#     doi = {10.1101/2021.05.23.445347},
#     publisher = {Cold Spring Harbor Laboratory},
#     URL = {https://www.biorxiv.org/content/early/2021/05/25/2021.05.23.445347},
#     eprint = {https://www.biorxiv.org/content/early/2021/05/25/2021.05.23.445347.full.pdf},
#     journal = {bioRxiv}
# }
#
# Licensed under the MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

from __future__ import annotations

import os
import pickle
import re
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset


# do not change any of these
THE20 = {
    'ALA': 0,
    'ARG': 1,
    'ASN': 2,
    'ASP': 3,
    'CYS': 4,
    'GLN': 5,
    'GLU': 6,
    'GLY': 7,
    'HIS': 8,
    'ILE': 9,
    'LEU': 10,
    'LYS': 11,
    'MET': 12,
    'PHE': 13,
    'PRO': 14,
    'SER': 15,
    'THR': 16,
    'TRP': 17,
    'TYR': 18,
    'VAL': 19,
}
SCH_ATOMS = {
    'ALA': 1,
    'ARG': 7,
    'ASN': 4,
    'ASP': 4,
    'CYS': 2,
    'GLN': 5,
    'GLU': 5,
    'GLY': 0,
    'HIS': 6,
    'ILE': 4,
    'LEU': 4,
    'LYS': 5,
    'MET': 4,
    'PHE': 7,
    'PRO': 3,
    'SER': 2,
    'THR': 3,
    'TRP': 10,
    'TYR': 8,
    'VAL': 3,
}
BB_ATOMS = ['C', 'CA', 'N', 'O']
SIDE_CHAINS = {
    'MET': ['CB', 'CE', 'CG', 'SD'],
    'ILE': ['CB', 'CD1', 'CG1', 'CG2'],
    'LEU': ['CB', 'CD1', 'CD2', 'CG'],
    'VAL': ['CB', 'CG1', 'CG2'],
    'THR': ['CB', 'CG2', 'OG1'],
    'ALA': ['CB'],
    'ARG': ['CB', 'CD', 'CG', 'CZ', 'NE', 'NH1', 'NH2'],
    'SER': ['CB', 'OG'],
    'LYS': ['CB', 'CD', 'CE', 'CG', 'NZ'],
    'HIS': ['CB', 'CD2', 'CE1', 'CG', 'ND1', 'NE2'],
    'GLU': ['CB', 'CD', 'CG', 'OE1', 'OE2'],
    'ASP': ['CB', 'CG', 'OD1', 'OD2'],
    'PRO': ['CB', 'CD', 'CG'],
    'GLN': ['CB', 'CD', 'CG', 'NE2', 'OE1'],
    'TYR': ['CB', 'CD1', 'CD2', 'CE1', 'CE2', 'CG', 'CZ', 'OH'],
    'TRP': [
        'CB',
        'CD1',
        'CD2',
        'CE2',
        'CE3',
        'CG',
        'CH2',
        'CZ2',
        'CZ3',
        'NE1',
    ],
    'CYS': ['CB', 'SG'],
    'ASN': ['CB', 'CG', 'ND2', 'OD1'],
    'PHE': ['CB', 'CD1', 'CD2', 'CE1', 'CE2', 'CG', 'CZ'],
}
BOX_SIZE = 10
GRID_SIZE = 40
SIGMA = 0.65

WEIGHT_URL = 'https://github.com/YaoYinYing/DLPacker/releases/download/v1.0-alpha/DLPacker_weights.7z'
WEIGHT_MD5 = 'md5:0a05db1e8a0468b570402efbd891102b'


class WeightBootstrapError(RuntimeError):
    """Raised when pretrained weight bootstrap cannot produce a valid checkpoint."""


def _validate_pt_checkpoint(path: str) -> None:
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state = checkpoint['state_dict']
            if not isinstance(state, dict) or not state:
                raise ValueError("'.pt' checkpoint has empty or invalid 'state_dict'.")
        elif not checkpoint:
            raise ValueError("'.pt' checkpoint is an empty dict.")
    else:
        raise ValueError("'.pt' checkpoint must be a dict-like object.")


def _validate_h5_weights(path: str) -> None:
    import h5py

    with h5py.File(path, 'r') as f:
        has_dataset = False

        def _visitor(_, obj):
            nonlocal has_dataset
            if isinstance(obj, h5py.Dataset):
                has_dataset = True

        f.visititems(_visitor)
        if not has_dataset:
            raise ValueError("'.h5' file does not contain any datasets.")


def _build_weight_error(
    weights_prefix: str,
    attempted_paths: List[str],
    last_error: Exception,
) -> WeightBootstrapError:
    output_dir = os.path.dirname(os.path.abspath(weights_prefix))
    h5_path = f'{weights_prefix}.h5'
    pt_path = f'{weights_prefix}.pt'
    msg = (
        'Failed to bootstrap pretrained DLPacker weights.\n'
        f'Attempted paths:\n  - {h5_path}\n  - {pt_path}\n'
        f'Weight URL: {WEIGHT_URL}\n'
        f'Output directory: {output_dir}\n'
        f'Root cause: {type(last_error).__name__}: {last_error}\n'
        'Manual remediation:\n'
        f'  1) Ensure the directory is writable: {output_dir}\n'
        f'  2) Download/extract archive so `{os.path.basename(h5_path)}` exists in that directory.\n'
        f'  3) Convert manually:\n'
        f'     python scripts/convert_keras_weights.py --weights-prefix {weights_prefix}\n'
        '  4) Optional override directory:\n'
        '     export DLPACKER_PRETRAINED_WEIGHT=/path/to/weights_dir\n'
        f'Failed attempts: {len(attempted_paths)}'
    )
    return WeightBootstrapError(msg)


def _fetch_and_extract_once(output_dir: str) -> List[str]:
    """Fetches pretrained archive and extracts into output_dir atomically."""
    os.makedirs(output_dir, exist_ok=True)

    import pooch
    import py7zr

    archive_path = pooch.retrieve(
        url=WEIGHT_URL,
        known_hash=WEIGHT_MD5,
        progressbar=False,
    )
    extracted_files: List[str] = []
    staging_dir = tempfile.mkdtemp(prefix='dlpacker_weights_extract_', dir=output_dir)
    try:
        with py7zr.SevenZipFile(archive_path, mode='r') as z:
            z.extractall(path=staging_dir)
        for name in os.listdir(staging_dir):
            src = os.path.join(staging_dir, name)
            dst = os.path.join(output_dir, name)
            os.replace(src, dst)
            extracted_files.append(name)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    return extracted_files


def ensure_pretrained_weights(
    weights_prefix: str,
    *,
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
    fetch_if_missing: bool = True,
) -> str:
    """Ensures a valid `<weights_prefix>.pt` exists, fetching/converting if needed."""
    if max_attempts < 1:
        raise ValueError('max_attempts must be >= 1')
    if backoff_seconds < 0:
        raise ValueError('backoff_seconds must be >= 0')

    h5_path = f'{weights_prefix}.h5'
    pt_path = f'{weights_prefix}.pt'
    out_dir = os.path.dirname(os.path.abspath(weights_prefix))
    os.makedirs(out_dir, exist_ok=True)

    attempted_paths: List[str] = []
    last_error: Exception | None = None

    def _try_validate_or_convert() -> str:
        if os.path.exists(pt_path):
            try:
                _validate_pt_checkpoint(pt_path)
                return pt_path
            except Exception:
                # Corrupt checkpoint should not be trusted.
                os.remove(pt_path)
        if os.path.exists(h5_path):
            _validate_h5_weights(h5_path)
            tmp_pt = f'{pt_path}.tmp'
            if os.path.exists(tmp_pt):
                os.remove(tmp_pt)
            try:
                convert_keras_h5_to_pt(keras_h5_path=h5_path, out_pt_path=tmp_pt)
                _validate_pt_checkpoint(tmp_pt)
                os.replace(tmp_pt, pt_path)
            except Exception:
                if os.path.exists(tmp_pt):
                    os.remove(tmp_pt)
                raise
            return pt_path
        raise FileNotFoundError(
            f'No weight file found. Expected either {pt_path} or {h5_path}.'
        )

    attempts = max_attempts if fetch_if_missing else 1
    for attempt in range(1, attempts + 1):
        try:
            return _try_validate_or_convert()
        except Exception as exc:
            last_error = exc
            attempted_paths.append(f'attempt={attempt}: {type(exc).__name__}: {exc}')
            if not fetch_if_missing:
                break
            # Clear stale artifacts before retrying fresh fetch/extract.
            for stale in (pt_path, h5_path, f'{pt_path}.tmp'):
                if os.path.exists(stale):
                    try:
                        os.remove(stale)
                    except Exception:
                        pass
            try:
                extracted = _fetch_and_extract_once(out_dir)
                print(f'Extracted files: {extracted}')
            except Exception as fetch_exc:
                last_error = fetch_exc
                attempted_paths.append(
                    f'attempt={attempt}: {type(fetch_exc).__name__}: {fetch_exc}'
                )
            if attempt < attempts and backoff_seconds > 0:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))

    assert last_error is not None
    raise _build_weight_error(weights_prefix, attempted_paths, last_error)


def fetch_and_unzip_weight(output_dir: str) -> List[str]:
    """Backward-compatible helper used by callers that only need fetch/extract."""
    try:
        return _fetch_and_extract_once(output_dir=output_dir)
    except Exception as exc:
        raise _build_weight_error(
            weights_prefix=os.path.join(output_dir, 'DLPacker_weights'),
            attempted_paths=[f'fetch_only: {type(exc).__name__}: {exc}'],
            last_error=exc,
        ) from exc


def _natural_sort_key(text: str) -> List[object]:
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r'(\d+)', text)]


def _collect_h5_datasets(group, prefix: str = '') -> Dict[str, np.ndarray]:
    import h5py

    out: Dict[str, np.ndarray] = {}
    for key, item in group.items():
        path = f'{prefix}/{key}' if prefix else key
        if isinstance(item, h5py.Dataset):
            out[path] = item[()]
        else:
            out.update(_collect_h5_datasets(item, path))
    return out


def _convert_keras_h5_to_state_dict(
    keras_h5_path: str,
    model: 'Generator3D',
) -> Dict[str, torch.Tensor]:
    import h5py

    with h5py.File(keras_h5_path, 'r') as f:
        arrays = _collect_h5_datasets(f)

    dense_kernel = None
    dense_bias = None
    conv_layers: Dict[str, Dict[str, np.ndarray]] = {}

    for path, arr in arrays.items():
        lpath = path.lower()
        if arr.ndim == 2 and 'kernel' in lpath and 'dense' in lpath:
            dense_kernel = arr
            continue
        if arr.ndim == 1 and 'bias' in lpath and 'dense' in lpath:
            dense_bias = arr
            continue

        if arr.ndim == 5 and 'kernel' in lpath:
            base = path.rsplit('/', 1)[0]
            conv_layers.setdefault(base, {})['kernel'] = arr
        elif arr.ndim == 1 and 'bias' in lpath:
            base = path.rsplit('/', 1)[0]
            conv_layers.setdefault(base, {})['bias'] = arr

    if dense_kernel is None or dense_bias is None:
        raise ValueError('Could not find Dense layer kernel/bias in keras .h5 weights.')

    conv_pairs = []
    for base, tensors in conv_layers.items():
        if 'kernel' in tensors and 'bias' in tensors:
            conv_pairs.append((base, tensors['kernel'], tensors['bias']))
    conv_pairs.sort(key=lambda x: _natural_sort_key(x[0]))

    target_convs = model.ordered_conv_layers()
    if len(conv_pairs) != len(target_convs):
        raise ValueError(
            f'Keras/PyTorch conv layer count mismatch: {len(conv_pairs)} != {len(target_convs)}'
        )

    state = model.state_dict()

    state['label_fc.weight'] = torch.from_numpy(dense_kernel.T.astype(np.float32))
    state['label_fc.bias'] = torch.from_numpy(dense_bias.astype(np.float32))

    for (_, k_kernel, k_bias), layer in zip(conv_pairs, target_convs):
        t_weight = torch.from_numpy(np.transpose(k_kernel, (4, 3, 0, 1, 2)).astype(np.float32))
        t_bias = torch.from_numpy(k_bias.astype(np.float32))

        if tuple(t_weight.shape) != tuple(state[f'{layer}.weight'].shape):
            raise ValueError(
                f'Weight shape mismatch for {layer}.weight: {tuple(t_weight.shape)} != {tuple(state[f"{layer}.weight"].shape)}'
            )
        if tuple(t_bias.shape) != tuple(state[f'{layer}.bias'].shape):
            raise ValueError(
                f'Bias shape mismatch for {layer}.bias: {tuple(t_bias.shape)} != {tuple(state[f"{layer}.bias"].shape)}'
            )

        state[f'{layer}.weight'] = t_weight
        state[f'{layer}.bias'] = t_bias

    return state


def convert_keras_h5_to_pt(
    keras_h5_path: str,
    out_pt_path: str,
    width: int = 128,
    nres: int = 6,
    grid_size: int = GRID_SIZE,
    num_channels: int = 27,
) -> str:
    model = Generator3D(width=width, nres=nres, grid_size=grid_size, num_channels=num_channels)
    state = _convert_keras_h5_to_state_dict(keras_h5_path, model)
    tmp_out = f'{out_pt_path}.tmp'
    if os.path.exists(tmp_out):
        os.remove(tmp_out)
    torch.save(
        {
            'state_dict': state,
            'meta': {
                'width': width,
                'nres': nres,
                'grid_size': grid_size,
                'num_channels': num_channels,
                'source': os.path.abspath(keras_h5_path),
            },
        },
        tmp_out,
    )
    os.replace(tmp_out, out_pt_path)
    return out_pt_path


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, bottleneck_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, bottleneck_channels, kernel_size=1)
        self.conv2 = nn.Conv3d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(bottleneck_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = F.relu(self.conv1(x), inplace=False)
        x = F.relu(self.conv2(x), inplace=False)
        x = self.conv3(x)
        x = x + identity
        return F.relu(x, inplace=False)


class Generator3D(nn.Module):
    def __init__(
        self,
        width: int,
        nres: int,
        grid_size: int = GRID_SIZE,
        num_channels: int = 27,
    ):
        super().__init__()
        self.width = width
        self.nres = nres
        self.grid_size = grid_size
        self.num_channels = num_channels

        self.label_fc = nn.Linear(20, grid_size * grid_size * grid_size)

        self.enc1 = nn.Conv3d(num_channels + 1, width, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Conv3d(width, 2 * width, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv3d(2 * width, 4 * width, kernel_size=3, stride=1, padding=1)

        self.res_blocks = nn.ModuleList(
            [ResidualBlock3D(in_channels=4 * width, bottleneck_channels=2 * width) for _ in range(nres)]
        )

        self.dec1 = nn.Conv3d(6 * width, 4 * width, kernel_size=3, stride=1, padding=1)
        self.dec2 = nn.Conv3d(5 * width, 2 * width, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv3d(2 * width + num_channels, 4, kernel_size=3, stride=1, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def ordered_conv_layers(self) -> List[str]:
        names = ['enc1', 'enc2', 'enc3']
        for i in range(self.nres):
            names.extend(
                [
                    f'res_blocks.{i}.conv1',
                    f'res_blocks.{i}.conv2',
                    f'res_blocks.{i}.conv3',
                ]
            )
        names.extend(['dec1', 'dec2', 'out_conv'])
        return names

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # input x expected in N,C,D,H,W
        fc = F.relu(self.label_fc(labels), inplace=False)
        fc = fc.view(-1, 1, self.grid_size, self.grid_size, self.grid_size)

        l0 = torch.cat([x, fc], dim=1)

        l1 = F.relu(self.enc1(l0), inplace=False)
        l2 = F.relu(self.enc2(l1), inplace=False)
        l3 = F.relu(self.enc3(l2), inplace=False)

        for block in self.res_blocks:
            l3 = block(l3)

        l = torch.cat([l3, l2], dim=1)
        l = self.leaky_relu(self.dec1(l))

        l = F.interpolate(l, scale_factor=2, mode='nearest')
        l = torch.cat([l, l1], dim=1)
        l = self.leaky_relu(self.dec2(l))

        l = F.interpolate(l, scale_factor=2, mode='nearest')
        l = torch.cat([l, x], dim=1)
        l = self.out_conv(l)

        l = l + x[:, :4, ...]
        return l


class _InferenceModelWrapper:
    """Keras-like callable wrapper to preserve downstream call sites."""

    def __init__(self, net: Generator3D, device: torch.device):
        self.net = net
        self.device = device

    def __call__(self, inputs: List[np.ndarray] | Tuple[np.ndarray, np.ndarray]):
        x, labels = inputs
        with torch.no_grad():
            x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            labels_t = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
            # NHWDC -> NCDHW
            x_t = x_t.permute(0, 4, 1, 2, 3).contiguous()
            out = self.net(x_t, labels_t)
            # NCDHW -> NHWDC
            out = out.permute(0, 2, 3, 4, 1).contiguous().cpu().detach()
            return out


class DLPModel:
    # This class represents DNN model we used in this work
    # If you just want to use the pre-trained weights that
    # we've published, then there is nothing you will ever
    # need to change here
    def __init__(
        self,
        grid_size: int = 40,
        nres: int = 6,
        num_channels: int = 27,
        batch_size: int = 32,
        lr: float = 1e-4,
        width: int = 128,
        device: str | None = None,
    ):
        self.width = width  # base number of channels
        self.lr = lr  # learning rate
        self.nres = nres  # number of residual layers
        self.grid_size = grid_size  # grid size
        self.num_channels = num_channels  # number of input channels
        self.batch_size = batch_size

        self.device = self._resolve_device(device)

        self.net = Generator3D(
            width=self.width,
            nres=self.nres,
            grid_size=self.grid_size,
            num_channels=self.num_channels,
        ).to(self.device)
        self.model = _InferenceModelWrapper(self.net, self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.data_gen = DataGenerator(self.batch_size, folder='./BOXES_TRAIN/')
        self.val_gen = DataGenerator(self.batch_size, folder='./BOXES_VAL/')

        self.loss_history = {'mae': [], 'roi': []}
        self.ema = 0.999  # for loss history smoothing
        self.weights_loaded = False

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        # Keep default behavior conservative and portable:
        # use CPU unless the user explicitly asks for an accelerator.
        if not device:
            return torch.device('cpu')

        requested = device.lower()
        if requested.startswith('cuda'):
            if not torch.cuda.is_available():
                raise ValueError("Requested device 'cuda' is not available.")
            return torch.device(device)
        if requested.startswith('mps'):
            if not torch.backends.mps.is_available():
                raise ValueError("Requested device 'mps' is not available.")
            return torch.device(device)
        if requested.startswith('cpu'):
            return torch.device(device)

        raise ValueError(
            f"Unsupported device '{device}'. Use one of: cpu, cuda, mps."
        )

    def __str__(self):
        total = sum(p.numel() for p in self.net.parameters())
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f'3D CNN Model\nLR: {self.lr}, BATCH SIZE: {self.batch_size}\n')
        print(self.net)
        print(f'Total parameters: {total}, trainable: {trainable}')
        return 'lol'

    def _weights_paths(self, weights: str) -> Tuple[str, str]:
        return f'{weights}.h5', f'{weights}.pt'

    def _ensure_converted_pt(self, weights: str) -> str:
        return ensure_pretrained_weights(
            weights_prefix=weights,
            max_attempts=3,
            backoff_seconds=1.0,
            fetch_if_missing=True,
        )

    def load_model(self, weights: str, history: str = ''):
        pt_path = self._ensure_converted_pt(weights)
        checkpoint = torch.load(pt_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state = checkpoint['state_dict']
        else:
            state = checkpoint

        self.net.load_state_dict(state)
        self.net.to(self.device)
        self.net.eval()
        self.weights_loaded = True

        if history:
            with open(history + '.pkl', 'rb') as h:
                self.loss_history = pickle.load(h)

    def save_model(self, weights: str, history: str = ''):
        _, pt_path = self._weights_paths(weights)
        torch.save(
            {
                'state_dict': self.net.state_dict(),
                'meta': {
                    'width': self.width,
                    'nres': self.nres,
                    'grid_size': self.grid_size,
                    'num_channels': self.num_channels,
                },
            },
            pt_path,
        )
        if history:
            with open(history + '.pkl', 'wb') as f:
                pickle.dump(self.loss_history, f)

    def _to_torch_batch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        labels_t = torch.as_tensor(labels, dtype=torch.float32, device=self.device)

        # NHWDC -> NCDHW
        x_t = x_t.permute(0, 4, 1, 2, 3).contiguous()
        y_t = y_t.permute(0, 4, 1, 2, 3).contiguous()
        return x_t, y_t, labels_t

    def train(self, epochs: int):
        self.net.train()
        for e in range(epochs):
            start = time.time()
            for i, (x, y, labels) in enumerate(self.data_gen):
                x_t, y_t, labels_t = self._to_torch_batch(x, y, labels)
                self.optimizer.zero_grad(set_to_none=True)

                out = self.net(x_t, labels_t)
                l = self.loss(x_t[:, :4, ...], y_t, labels_t, out)
                l.backward()
                self.optimizer.step()

                end = time.time()
                print(
                    'Epoch: %d/%d' % (e + 1, epochs),
                    'Iteration:',
                    i,
                    'Losses [MAE, ROI]: [%.3e, %.3e] | Time per step: %.2f s'
                    % (
                        self.loss_history['mae'][-1],
                        self.loss_history['roi'][-1],
                        end - start,
                    ),
                    end='\r',
                )
                start = time.time()

                if i % 10000 == 0:
                    self.save_model('backup')

    def validate(self):
        self.net.eval()
        maes = []
        rois = []
        with torch.no_grad():
            for i, (x, y, labels) in enumerate(self.val_gen):
                if i > 0:
                    print('Batch:', i, np.mean(rois), end='\r')
                x_t, y_t, labels_t = self._to_torch_batch(x, y, labels)
                out = self.net(x_t, labels_t)

                x4 = x_t[:, :4, ...]
                mae = torch.mean(torch.abs(y_t - out))
                mask = (x4 != y_t).to(dtype=torch.float32)
                roi = torch.mean(torch.abs(y_t - out) * mask) * 100.0

                maes.append(float(mae.cpu().item()))
                rois.append(float(roi.cpu().item()))

        print('MAE:', np.mean(maes), 'ROI:', np.mean(rois))

    def loss(self, x, y, labels, out):
        del labels  # kept for API compatibility

        mae = torch.mean(torch.abs(y - out))
        mask = (x != y).to(dtype=torch.float32)
        roi = torch.mean(torch.abs(y - out) * mask) * 100.0

        mae_val = float(mae.detach().cpu().item())
        roi_val = float(roi.detach().cpu().item())

        if self.loss_history['mae']:
            mae_ema = mae_val * (1 - self.ema) + self.ema * self.loss_history['mae'][-1]
            self.loss_history['mae'].append(mae_ema)
            roi_ema = roi_val * (1 - self.ema) + self.ema * self.loss_history['roi'][-1]
            self.loss_history['roi'].append(roi_ema)
        else:
            self.loss_history['mae'].append(mae_val)
            self.loss_history['roi'].append(roi_val)

        return roi + mae


class InputBoxReader:
    # Input generator creating 26 input channels for the DNN.
    # Takes in a special dictionary containing information about microenvironment
    # or a filename containing such a dictionary.
    # This version puts backbone atoms for each amino acid into separate channel
    # as well as into element channels.
    # Side chain atoms only go into element channels.
    # The structure is like this:
    #     Channels 1-4 - CNOS
    #     Channel  5 - all other elements
    #     Channel  6 - charges
    #     Channels 7-26 - backbones of amino acids, each channel for one AA
    # Output tensor has four element chanels (CNOS)
    def __init__(
        self,
        remove_sidechains: str = 'none',
        from_dict: bool = True,
        include_water: bool = False,
        charges_filename: str = 'charges.rtp',
    ):
        self.grid_size = GRID_SIZE
        self.grid_spacing = BOX_SIZE * 2 / GRID_SIZE  # grid step
        self.offset = 10 * GRID_SIZE // 40  # to include atoms on the border
        self.total_size = GRID_SIZE + 2 * self.offset
        self.remove_sidechains = remove_sidechains
        self.from_dict = from_dict  # choose dict or filename as input
        self.include_water = include_water

        assert self.remove_sidechains in ['all', 'random', 'none']

        # preparing the kernel and grid
        size = round(SIGMA * 4)  # kernel size
        self.grid = np.mgrid[
            -size : size + self.grid_spacing : self.grid_spacing,
            -size : size + self.grid_spacing : self.grid_spacing,
            -size : size + self.grid_spacing : self.grid_spacing,
        ]

        # defining a kernel
        kernel = np.exp(-np.sum(self.grid * self.grid, axis=0) / SIGMA**2 / 2)
        kernel /= np.sqrt(2 * np.pi) * SIGMA
        self.kernel = kernel[1:-1, 1:-1, 1:-1]
        self.norm = np.sum(self.kernel)

        # read in the charges from special file
        self.charges = defaultdict(lambda: 0)  # output 0 if the key is absent
        with open(charges_filename, 'r') as f:
            for line in f:
                if line[0] == '[' or line[0] == ' ':
                    if re.match(r'\A\[ .{1,3} \]\Z', line[:-1]):
                        key = re.match(r'\A\[ (.{1,3}) \]\Z', line[:-1])[1]
                        self.charges[key] = defaultdict(lambda: 0)
                    else:
                        l = re.split(r' +', line[:-1])
                        self.charges[key][l[1]] = float(l[3])

    def __call__(self, box: [str, dict]):
        # input is either a dictionary or a filename with a dictionary stored in it
        if not self.from_dict:
            box = np.load(box, allow_pickle=True)
            box = box['arr_0'].item()

        # list of all amino acids except for target
        # in the end we want it to contain all amino acids
        # whose side chains we want to see in the input
        amino_acids = set(box['resids'])
        amino_acids.remove(int(box['target']['id']))

        if self.remove_sidechains == 'random':
            # if we choose to randomly remove amino acid sidechains
            # then we have 50% chance to remove them all
            # and 50% chance to remove random fraction of them
            if np.random.rand() < 0.25:
                p = np.random.rand()
                amino_acids = set([a for a in amino_acids if np.random.rand() < p])
            else:
                amino_acids = set()
        elif self.remove_sidechains == 'all':
            amino_acids = set()

        x = np.zeros([self.total_size, self.total_size, self.total_size, 27], dtype=np.float32)
        y = np.zeros([self.total_size, self.total_size, self.total_size, 4], dtype=np.float32)

        centers = (np.array(box['positions']) + BOX_SIZE) / self.grid_spacing
        centers += self.offset
        cr = np.round(centers).astype(np.int32)
        offsets = cr - centers
        offsets = offsets[:, :, None, None, None]

        i0 = self.kernel.shape[0] // 2
        i1 = self.kernel.shape[0] - i0

        for ind, a in enumerate(box['types']):
            if box['resnames'][ind] != 'HOH' or self.include_water:
                # defines fine position of the kernel
                dist = self.grid + offsets[ind] * self.grid_spacing
                kernel = np.exp(-np.sum(dist * dist, axis=0) / SIGMA**2 / 2)
                kernel = kernel[1:-1, 1:-1, 1:-1] * self.norm / np.sum(kernel)

                # defining indeces to put atom into
                xa, xb = cr[ind][0] - i0, cr[ind][0] + i1
                ya, yb = cr[ind][1] - i0, cr[ind][1] + i1
                za, zb = cr[ind][2] - i0, cr[ind][2] + i1

                # define the channel for the atom
                if a == 'C':
                    ch = 0
                elif a == 'N':
                    ch = 1
                elif a == 'O':
                    ch = 2
                elif a == 'S':
                    ch = 3
                else:
                    ch = 4

                aa = box['resnames'][ind]  # amino acid
                an = box['names'][ind]  # atom name

                # filling in input element channels,
                # charge channel and all of output channels

                # check if the atom is in target side chain
                if ind in box['target']['atomids']:
                    # target atoms only go into output
                    if ch != 4:
                        y[xa:xb, ya:yb, za:zb, ch] += kernel
                elif an in BB_ATOMS or box['resids'][ind] in amino_acids:
                    # otherwise, the atom goes into input
                    # element channels
                    x[xa:xb, ya:yb, za:zb, ch] += kernel
                    # all CNOS atoms also go into output
                    if ch != 4:
                        y[xa:xb, ya:yb, za:zb, ch] += kernel
                    # add charges as same kernels multiplied
                    # by partial charge value
                    if aa in self.charges:
                        # if charge value is known, use it
                        x[xa:xb, ya:yb, za:zb, 5] += kernel * self.charges[aa][an]
                    else:
                        # otherwise use default values
                        charge = kernel * self.charges['RST'][an[:1]]
                        x[xa:xb, ya:yb, za:zb, 5] += charge
                # filling in amino acid backbone channels
                if an in BB_ATOMS:
                    if aa in THE20:
                        x[xa:xb, ya:yb, za:zb, 6 + THE20[aa]] += kernel
                    else:
                        x[xa:xb, ya:yb, za:zb, 6 + 20] += kernel
            b = self.offset
        return x[b:-b, b:-b, b:-b, :], y[b:-b, b:-b, b:-b, :]


class _DLPBoxesIterable(IterableDataset):
    def __init__(
        self,
        num_channels: int,
        grid_size: int,
        randomize: bool,
        folder: str,
        remove_sidechains: str,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.grid_size = grid_size
        self.randomize = randomize
        self.folder = folder
        self.remove_sidechains = remove_sidechains

    def _list_files(self) -> List[str]:
        files: List[str] = []
        if not self.folder or not os.path.isdir(self.folder):
            return files

        for f in os.listdir(self.folder):
            ff_path = os.path.join(self.folder, f)
            if not os.path.isdir(ff_path):
                continue
            for file in os.listdir(ff_path):
                files.append(os.path.join(ff_path, file))

        if self.randomize:
            np.random.shuffle(files)
        return files

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        files = self._list_files()
        input_reader = InputBoxReader(remove_sidechains=self.remove_sidechains)

        if self.folder and not self.folder.endswith(os.sep):
            folder_prefix = self.folder + os.sep
        else:
            folder_prefix = self.folder

        s = len(folder_prefix) + 12
        e = s + 3

        for sample_path in files:
            label = np.zeros((20), dtype=np.float32)
            label[THE20[sample_path[s:e]]] = 1
            x, y = input_reader(sample_path)
            yield x, y, label


class DataGenerator:
    # Pretty standard data generator preserving the old API surface.
    def __new__(
        cls,
        batch_size: int = 32,
        num_channels: int = 27,
        grid_size: int = GRID_SIZE,
        randomize: bool = True,
        remove_sidechains: str = 'all',
        folder: str = '',
    ):
        dataset = _DLPBoxesIterable(
            num_channels=num_channels,
            grid_size=grid_size,
            randomize=randomize,
            folder=folder,
            remove_sidechains=remove_sidechains,
        )
        return DataLoader(dataset, batch_size=batch_size)
