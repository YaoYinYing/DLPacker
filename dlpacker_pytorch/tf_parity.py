from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def tensorflow_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:
        return False


def _import_tf():
    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - exercised in optional tests
        raise RuntimeError(
            'TensorFlow is required for TF parity checks. '
            'Install test extra/dependency and retry.'
        ) from exc
    return tf


def build_tf_generator(
    *,
    width: int,
    nres: int,
    grid_size: int,
    num_channels: int,
):
    tf = _import_tf()
    K = tf.keras

    inp = K.layers.Input(shape=(grid_size, grid_size, grid_size, num_channels))
    labels = K.layers.Input(shape=(20,))

    fc = K.layers.Dense(grid_size * grid_size * grid_size, activation='relu')(labels)
    fc = tf.reshape(fc, shape=(-1, grid_size, grid_size, grid_size, 1))
    l0 = K.layers.Concatenate(axis=-1)([inp, fc])

    def res_identity(x, f1, f2):
        x_in = x
        x = K.layers.Conv3D(f1, 1, strides=1, padding='valid', activation='relu')(x)
        x = K.layers.Conv3D(f1, 3, strides=1, padding='same', activation='relu')(x)
        x = K.layers.Conv3D(f2, 1, strides=1, padding='valid')(x)
        x = K.layers.Add()([x, x_in])
        return K.layers.Activation('relu')(x)

    l1 = K.layers.Conv3D(width, 3, padding='same', strides=2, activation='relu')(l0)
    l2 = K.layers.Conv3D(2 * width, 3, padding='same', strides=2, activation='relu')(l1)
    l3 = K.layers.Conv3D(4 * width, 3, padding='same', strides=1, activation='relu')(l2)
    for _ in range(nres):
        l3 = res_identity(l3, 2 * width, 4 * width)
    l = K.layers.Concatenate(axis=-1)([l3, l2])
    l = K.layers.Conv3D(4 * width, 3, padding='same')(l)
    l = K.layers.LeakyReLU(alpha=0.2)(l)

    l = K.layers.UpSampling3D(size=2)(l)
    l = K.layers.Concatenate(axis=-1)([l, l1])
    l = K.layers.Conv3D(2 * width, 3, padding='same')(l)
    l = K.layers.LeakyReLU(alpha=0.2)(l)

    l = K.layers.UpSampling3D(size=2)(l)
    l = K.layers.Concatenate(axis=-1)([l, inp])
    l = K.layers.Conv3D(4, 3, padding='same')(l)

    l = l + inp[..., :4]
    return K.Model(inputs=[inp, labels], outputs=l, name='Generator')


def load_tf_model_from_h5(
    h5_path: str,
    *,
    width: int = 128,
    nres: int = 6,
    grid_size: int = 40,
    num_channels: int = 27,
):
    model = build_tf_generator(
        width=width,
        nres=nres,
        grid_size=grid_size,
        num_channels=num_channels,
    )
    model.load_weights(h5_path)
    return model


def postprocess_prediction(raw_output: np.ndarray, raw_input: np.ndarray, label: str) -> np.ndarray:
    pred = raw_output - raw_input[..., :4]
    pred[pred < 0] = 0
    pred = pred[5:-5, 5:-5, 5:-5, :]
    dpred = pred.reshape(15, 2, 15, 2, 15, 2, 4).mean(axis=(1, 3, 5))
    if label not in ['ASN', 'GLN', 'HIS']:
        dpred = np.sum(dpred, axis=-1)
    return dpred


@dataclass
class RotamerParity:
    best_idx: int
    topk_idx: Tuple[int, ...]


def rank_rotamers(library_grids: np.ndarray, prediction: np.ndarray, top_k: int = 5) -> RotamerParity:
    scores = np.abs(library_grids - prediction)
    scores = np.mean(scores, axis=tuple(range(1, prediction.ndim + 1)))
    order = np.argsort(scores)
    k = max(1, min(top_k, int(order.shape[0])))
    return RotamerParity(best_idx=int(order[0]), topk_idx=tuple(int(x) for x in order[:k]))
