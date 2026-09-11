from __future__ import annotations

import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import ACTION_SPACE, OBS_CHANNELS, TILE_TYPES

OBS_ELEMENTS = OBS_CHANNELS * TILE_TYPES
OBS_BYTES = (OBS_ELEMENTS + 7) // 8
ACTION_BYTES = (ACTION_SPACE + 7) // 8
# v4 adds the kyoku index that the packer needs to keep a resampled corpus
# reproducible.  The marker prevents a run from mixing earlier shards silently.
STORAGE_FORMAT = "dual-bitpack-sparse-float16-v4"
STAGED_GAME_FORMAT = "riichi-analysis-staged-game-v1"
PACK_FORMAT = "riichi-analysis-global-pack-v1"


@dataclass
class PackedObservations:
    nonzero: np.ndarray
    nonone: np.ndarray
    values: np.ndarray
    offsets: np.ndarray


def pack_observations(observations: np.ndarray) -> PackedObservations:
    observations = np.asarray(observations)
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


def _sample_count(packed: dict[str, np.ndarray]) -> int:
    return len(packed["obs_offsets"]) - 1


def pack_shard_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Pack one shard's arrays into their stored representation."""

    required = {"obs", "action_mask"}
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"missing arrays: {sorted(missing)}")
    payload = dict(arrays)
    obs = pack_observations(payload.pop("obs"))
    return {
        "storage_format": np.asarray(STORAGE_FORMAT),
        "obs_nonzero": obs.nonzero,
        "obs_nonone": obs.nonone,
        "obs_values": obs.values,
        "obs_offsets": obs.offsets,
        "action_mask": pack_action_masks(payload.pop("action_mask")),
        **payload,
    }


def unpack_shard_arrays(packed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Inverse of :func:`pack_shard_arrays`."""

    arrays = dict(packed)
    arrays.pop("storage_format", None)
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


def write_packed_shard(path: str | Path, packed: dict[str, np.ndarray]) -> None:
    """Write a packed shard exactly as given, replacing the destination atomically."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = dict(packed)
    payload.setdefault("storage_format", np.asarray(STORAGE_FORMAT))
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, destination)


def read_packed_shard(path: str | Path) -> dict[str, np.ndarray]:
    """Read a shard without expanding the packed observations."""

    with np.load(Path(path), allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    marker = arrays.get("storage_format")
    if marker is None or marker.item() != STORAGE_FORMAT:
        raise ValueError(f"unsupported storage format in {path}")
    return arrays


def save_shard(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    """Write one compressed, self-describing shard."""

    write_packed_shard(path, pack_shard_arrays(arrays))


def load_shard(path: str | Path, *, unpack_obs: bool = True) -> dict[str, np.ndarray]:
    packed = read_packed_shard(path)
    return unpack_shard_arrays(packed) if unpack_obs else packed


def concatenate_packed(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate packed shards while rebasing the sparse observation offsets."""

    if not parts:
        raise ValueError("nothing to concatenate")
    names = set(parts[0])
    if any(set(part) != names for part in parts):
        raise ValueError("packed shard schemas differ")
    result: dict[str, np.ndarray] = {"storage_format": parts[0]["storage_format"]}
    for name in sorted(names - {"storage_format"}):
        if name == "obs_offsets":
            pieces = [parts[0][name][:-1]]
            total = int(parts[0][name][-1])
            for part in parts[1:]:
                pieces.append(part[name][:-1] + total)
                total += int(part[name][-1])
            result[name] = np.concatenate(
                [*pieces, np.asarray([total], dtype=parts[0][name].dtype)]
            )
        else:
            result[name] = np.concatenate([part[name] for part in parts], axis=0)
    return result


def permute_packed(packed: dict[str, np.ndarray], order: np.ndarray) -> dict[str, np.ndarray]:
    """Reorder samples inside a packed shard without expanding the observations."""

    order = np.asarray(order, dtype=np.int64)
    offsets = packed["obs_offsets"]
    count = _sample_count(packed)
    if order.ndim != 1 or len(order) != count:
        raise ValueError("order must cover every sample exactly once")
    if len(order) and (int(order.min()) < 0 or int(order.max()) >= count):
        raise ValueError("order index is outside the sample range")

    # Offsets are unsigned, so the address arithmetic below runs in int64 and
    # the rebased offsets are cast back before they are stored.
    counts = np.diff(offsets).astype(np.int64, copy=False)
    selected = counts[order]
    new_offsets = np.zeros(len(order) + 1, dtype=np.int64)
    np.cumsum(selected, out=new_offsets[1:])

    starts = offsets[:-1][order].astype(np.int64, copy=False)
    element_start = np.repeat(starts, selected)
    within_run = np.arange(int(selected.sum()), dtype=np.int64) - np.repeat(
        new_offsets[:-1], selected
    )
    values = packed["obs_values"][element_start + within_run]

    result: dict[str, np.ndarray] = {
        "storage_format": packed["storage_format"],
        "obs_nonzero": packed["obs_nonzero"][order],
        "obs_nonone": packed["obs_nonone"][order],
        "obs_values": values,
        "obs_offsets": new_offsets.astype(offsets.dtype, copy=False),
    }
    for name, value in packed.items():
        if name in result or name == "storage_format":
            continue
        result[name] = value[order]
    return result


def save_chunk_archive(
    path: str | Path, arrays: dict[str, np.ndarray], chunk_samples: int
) -> dict[str, object]:
    """Stage one game as independently compressed sample chunks."""

    if chunk_samples <= 0:
        raise ValueError("chunk size must be positive")
    packed = pack_shard_arrays(arrays)
    total = _sample_count(packed)
    destinations = Path(path)
    destinations.parent.mkdir(parents=True, exist_ok=True)
    temporary = destinations.with_suffix(destinations.suffix + ".tmp")
    lengths: list[int] = []
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        for member_index, start in enumerate(range(0, total, chunk_samples)):
            stop = min(total, start + chunk_samples)
            chunk = {"storage_format": packed["storage_format"]}
            for name, value in packed.items():
                if name not in {"storage_format", "obs_offsets"}:
                    chunk[name] = value[start:stop]
            # A chunk keeps one more offset than it has samples so its own
            # sparse-value runs stay addressable after the rebase.
            offsets = packed["obs_offsets"][start : stop + 1]
            chunk["obs_offsets"] = offsets - offsets[0]
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **chunk)
            archive.writestr(f"chunk_{member_index:05d}.npz", buffer.getvalue())
            lengths.append(stop - start)
        meta: dict[str, object] = {
            "format": STAGED_GAME_FORMAT,
            "samples": total,
            "chunkSamples": chunk_samples,
            "chunkLengths": lengths,
        }
        archive.writestr("meta.json", json.dumps(meta, separators=(",", ":")))
    os.replace(temporary, destinations)
    return meta


def read_chunk_archive_meta(path: str | Path) -> dict[str, object]:
    with zipfile.ZipFile(Path(path), "r") as archive:
        meta = json.loads(archive.read("meta.json"))
    if meta.get("format") != STAGED_GAME_FORMAT:
        raise ValueError(f"unsupported staged game format in {path}")
    return meta


def read_chunk_payload(path: str | Path, member_index: int) -> bytes:
    with zipfile.ZipFile(Path(path), "r") as archive:
        return archive.read(f"chunk_{member_index:05d}.npz")
