from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import ACTION_SPACE, OBS_CHANNELS, TILE_TYPES

OBS_ELEMENTS = OBS_CHANNELS * TILE_TYPES
OBS_BYTES = (OBS_ELEMENTS + 7) // 8
ACTION_BYTES = (ACTION_SPACE + 7) // 8
STORAGE_FORMAT = "dual-bitpack-sparse-float16-v1"


@dataclass
class PackedObservations:
    nonzero: np.ndarray
    nonone: np.ndarray
    values: np.ndarray
    offsets: np.ndarray


def pack_observations(observations: np.ndarray) -> PackedObservations:
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim != 3 or observations.shape[1:] != (OBS_CHANNELS, TILE_TYPES):
        raise ValueError(f"wrong observation shape: {observations.shape}")
    if not np.isfinite(observations).all():
        raise ValueError("observations contain NaN or infinity")
    flat = observations.reshape(len(observations), OBS_ELEMENTS)
    nonzero_bool = flat != 0
    nonone_bool = nonzero_bool & (flat != 1)
    nonzero = np.packbits(nonzero_bool, axis=1, bitorder="little")
    nonone = np.packbits(nonone_bool, axis=1, bitorder="little")
    counts = nonone_bool.sum(axis=1, dtype=np.uint64)
    offsets = np.empty(len(observations) + 1, dtype=np.uint64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    values = flat[nonone_bool].astype(np.float16, copy=False)
    return PackedObservations(nonzero, nonone, values, offsets)


def unpack_observations(packed: PackedObservations) -> np.ndarray:
    length = len(packed.nonzero)
    if packed.nonzero.shape != (length, OBS_BYTES):
        raise ValueError(f"wrong nonzero shape: {packed.nonzero.shape}")
    if packed.nonone.shape != (length, OBS_BYTES):
        raise ValueError(f"wrong nonone shape: {packed.nonone.shape}")
    if packed.offsets.shape != (length + 1,):
        raise ValueError(f"wrong offsets shape: {packed.offsets.shape}")

    nonzero = np.unpackbits(
        packed.nonzero, axis=1, count=OBS_ELEMENTS, bitorder="little"
    ).astype(np.float32, copy=False)
    nonone = np.unpackbits(
        packed.nonone, axis=1, count=OBS_ELEMENTS, bitorder="little"
    ).astype(bool, copy=False)
    if int(packed.offsets[-1]) != int(nonone.sum()):
        raise ValueError("sparse value count does not match non-one mask")
    nonzero[nonone] = packed.values.astype(np.float32, copy=False)
    return nonzero.reshape(length, OBS_CHANNELS, TILE_TYPES)


def pack_action_masks(masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 2 or masks.shape[1] != ACTION_SPACE:
        raise ValueError(f"wrong action-mask shape: {masks.shape}")
    return np.packbits(masks, axis=1, bitorder="little")


def unpack_action_masks(masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks, dtype=np.uint8)
    if masks.ndim != 2 or masks.shape[1] != ACTION_BYTES:
        raise ValueError(f"wrong packed action-mask shape: {masks.shape}")
    return np.unpackbits(masks, axis=1, count=ACTION_SPACE, bitorder="little").astype(bool)


def save_shard(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    """Write one compressed, self-describing shard."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    required = {"obs", "action_mask"}
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"missing arrays: {sorted(missing)}")
    payload = dict(arrays)
    obs = pack_observations(payload.pop("obs"))
    action_mask = pack_action_masks(payload.pop("action_mask"))
    np.savez_compressed(
        destination,
        storage_format=np.asarray(STORAGE_FORMAT),
        obs_nonzero=obs.nonzero,
        obs_nonone=obs.nonone,
        obs_values=obs.values,
        obs_offsets=obs.offsets,
        action_mask=action_mask,
        **payload,
    )


def load_shard(path: str | Path, *, unpack_obs: bool = True) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    if arrays.pop("storage_format").item() != STORAGE_FORMAT:
        raise ValueError(f"unsupported storage format in {path}")
    if unpack_obs:
        arrays["obs"] = unpack_observations(
            PackedObservations(
                arrays.pop("obs_nonzero"),
                arrays.pop("obs_nonone"),
                arrays.pop("obs_values"),
                arrays.pop("obs_offsets"),
            )
        )
        arrays["action_mask"] = unpack_action_masks(arrays["action_mask"])
    return arrays
