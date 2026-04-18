from __future__ import annotations

from pathlib import Path

import h5py
import pytest
import torch

import dlpacker_pytorch.dlpacker as dlp_mod
from dlpacker_pytorch.utils import WEIGHT_URL, WeightBootstrapError, ensure_pretrained_weights


def _write_min_h5(path: Path) -> None:
    with h5py.File(path, 'w') as f:
        f.create_dataset('dummy', data=[1.0])


def _write_valid_pt(path: Path) -> None:
    torch.save({'state_dict': {'x': torch.ones(1)}}, path)


def test_bootstrap_uses_existing_valid_pt_without_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prefix = tmp_path / 'DLPacker_weights'
    pt_path = prefix.with_suffix('.pt')
    _write_valid_pt(pt_path)

    def _never_fetch(_):
        raise AssertionError('fetch should not be called')

    monkeypatch.setattr('dlpacker_pytorch.utils._fetch_and_extract_once', _never_fetch)

    out = ensure_pretrained_weights(str(prefix), fetch_if_missing=True)
    assert out == str(pt_path)


def test_bootstrap_converts_from_h5_when_pt_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prefix = tmp_path / 'DLPacker_weights'
    h5_path = prefix.with_suffix('.h5')
    _write_min_h5(h5_path)

    def _fake_convert(keras_h5_path: str, out_pt_path: str, **_):
        assert keras_h5_path == str(h5_path)
        _write_valid_pt(Path(out_pt_path))
        return out_pt_path

    monkeypatch.setattr('dlpacker_pytorch.utils.convert_keras_h5_to_pt', _fake_convert)

    out = ensure_pretrained_weights(str(prefix), fetch_if_missing=False)
    assert out == str(prefix.with_suffix('.pt'))
    assert prefix.with_suffix('.pt').exists()


def test_bootstrap_regenerates_when_pt_corrupt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prefix = tmp_path / 'DLPacker_weights'
    h5_path = prefix.with_suffix('.h5')
    pt_path = prefix.with_suffix('.pt')

    _write_min_h5(h5_path)
    pt_path.write_text('corrupt checkpoint')

    called = {'count': 0}

    def _fake_convert(**kwargs):
        called['count'] += 1
        _write_valid_pt(Path(kwargs['out_pt_path']))
        return kwargs['out_pt_path']

    monkeypatch.setattr('dlpacker_pytorch.utils.convert_keras_h5_to_pt', _fake_convert)

    out = ensure_pretrained_weights(str(prefix), fetch_if_missing=False)
    assert out == str(pt_path)
    assert called['count'] == 1


def test_bootstrap_fetches_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prefix = tmp_path / 'DLPacker_weights'
    h5_path = prefix.with_suffix('.h5')

    def _fake_fetch(output_dir: str):
        _write_min_h5(Path(output_dir) / h5_path.name)
        return [h5_path.name]

    def _fake_convert(**kwargs):
        _write_valid_pt(Path(kwargs['out_pt_path']))
        return kwargs['out_pt_path']

    monkeypatch.setattr('dlpacker_pytorch.utils._fetch_and_extract_once', _fake_fetch)
    monkeypatch.setattr('dlpacker_pytorch.utils.convert_keras_h5_to_pt', _fake_convert)

    out = ensure_pretrained_weights(str(prefix), max_attempts=2, backoff_seconds=0)
    assert out == str(prefix.with_suffix('.pt'))


def test_bootstrap_retries_after_transient_fetch_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prefix = tmp_path / 'DLPacker_weights'
    h5_name = 'DLPacker_weights.h5'

    call_count = {'n': 0}

    def _flaky_fetch(output_dir: str):
        call_count['n'] += 1
        if call_count['n'] == 1:
            raise RuntimeError('temporary network issue')
        _write_min_h5(Path(output_dir) / h5_name)
        return [h5_name]

    def _fake_convert(**kwargs):
        _write_valid_pt(Path(kwargs['out_pt_path']))
        return kwargs['out_pt_path']

    monkeypatch.setattr('dlpacker_pytorch.utils._fetch_and_extract_once', _flaky_fetch)
    monkeypatch.setattr('dlpacker_pytorch.utils.convert_keras_h5_to_pt', _fake_convert)

    out = ensure_pretrained_weights(str(prefix), max_attempts=3, backoff_seconds=0)
    assert out.endswith('.pt')
    assert call_count['n'] == 2


def test_bootstrap_persistent_fetch_error_has_actionable_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prefix = tmp_path / 'DLPacker_weights'

    def _always_fail(_):
        raise RuntimeError('offline')

    monkeypatch.setattr('dlpacker_pytorch.utils._fetch_and_extract_once', _always_fail)

    with pytest.raises(WeightBootstrapError) as exc:
        ensure_pretrained_weights(
            str(prefix),
            max_attempts=2,
            backoff_seconds=0,
            fetch_if_missing=True,
        )

    msg = str(exc.value)
    assert WEIGHT_URL in msg
    assert 'DLPACKER_PRETRAINED_WEIGHT' in msg
    assert 'convert_keras_weights.py' in msg
    assert str(prefix.with_suffix('.h5')) in msg
    assert str(prefix.with_suffix('.pt')) in msg


def test_failed_conversion_does_not_leave_partial_pt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prefix = tmp_path / 'DLPacker_weights'
    h5_path = prefix.with_suffix('.h5')
    _write_min_h5(h5_path)

    def _bad_convert(**kwargs):
        tmp_out = Path(kwargs['out_pt_path'])
        tmp_out.write_text('partial checkpoint')
        raise RuntimeError('conversion failed')

    monkeypatch.setattr('dlpacker_pytorch.utils.convert_keras_h5_to_pt', _bad_convert)

    with pytest.raises(WeightBootstrapError):
        ensure_pretrained_weights(
            str(prefix),
            fetch_if_missing=False,
        )

    assert not prefix.with_suffix('.pt').exists()
    assert not Path(str(prefix.with_suffix('.pt')) + '.tmp').exists()


def test_dlpacker_callsite_uses_unified_bootstrap(monkeypatch: pytest.MonkeyPatch):
    called = {}

    def _fake_ensure(*, weights_prefix: str, max_attempts: int, backoff_seconds: float, fetch_if_missing: bool):
        called['weights_prefix'] = weights_prefix
        called['max_attempts'] = max_attempts
        called['backoff_seconds'] = backoff_seconds
        called['fetch_if_missing'] = fetch_if_missing
        return weights_prefix + '.pt'

    monkeypatch.setattr(dlp_mod, 'ensure_pretrained_weights', _fake_ensure)
    dlp_mod._ensure_weights_available('/tmp/demo_weights')

    assert called == {
        'weights_prefix': '/tmp/demo_weights',
        'max_attempts': 3,
        'backoff_seconds': 1.0,
        'fetch_if_missing': True,
    }
