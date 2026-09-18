from __future__ import annotations

import io
import json
import os
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import ACTION_SPACE, OBS_CHANNELS, TILE_TYPES
from .model_input import (
    LEGACY_MODEL_INPUT_SCHEMA_ID,
    MODEL_INPUT_CHANNELS,
    MODEL_INPUT_SCHEMA_ID,
)
from .semantic_input import EVENT_FIELDS, EVENT_MEMORY_SCHEMA_ID

ACTION_BYTES = (ACTION_SPACE + 7) // 8
# v4 adds the kyoku index, so a corpus can later be resampled a whole kyoku at
# a time instead of a sample at a time.  The marker prevents a run from mixing
# earlier shards silently.
STORAGE_FORMAT = "dual-bitpack-sparse-float16-v6"
LEGACY_STORAGE_FORMATS = frozenset(
    {"dual-bitpack-sparse-float16-v5", "dual-bitpack-sparse-float16-v4"}
)
SUPPORTED_STORAGE_FORMATS = frozenset({STORAGE_FORMAT, *LEGACY_STORAGE_FORMATS})
STAGED_GAME_FORMAT = "riichi-analysis-staged-game-v3"
LEGACY_STAGED_GAME_FORMATS = frozenset(
    {"riichi-analysis-staged-game-v2", "riichi-analysis-staged-game-v1"}
)
PACK_FORMAT = "riichi-analysis-global-pack-v3"
PACKED_METADATA_FIELDS = frozenset(
    {
        "storage_format",
        "obs_channels",
        "model_input_schema",
        "event_memory_schema",
    }
)
PACK_CATALOG_FIELDS = frozenset(
    {"event_catalog", "event_catalog_games", "event_catalog_offsets"}
)
EVENT_CATALOG_MEMBER = "event_catalog.npy"
EVENT_REFERENCE_FIELDS = frozenset({"history_start", "history_length"})


@dataclass
class PackedObservations:
    nonzero: np.ndarray
    nonone: np.ndarray
    values: np.ndarray
    offsets: np.ndarray
    channels: int = OBS_CHANNELS


def pack_observations(observations: np.ndarray) -> PackedObservations:
    observations = np.asarray(observations)
    if observations.ndim != 3 or observations.shape[2] != TILE_TYPES:
        raise ValueError(f"wrong observation shape: {observations.shape}")
    if not np.isfinite(observations).all():
        raise ValueError("observations contain NaN or infinity")
    channels = int(observations.shape[1])
    if channels <= 0:
        raise ValueError("observations must contain at least one channel")
    flat = observations.reshape(len(observations), channels * TILE_TYPES)
    nonzero_bool = flat != 0
    nonone_bool = nonzero_bool & (flat != 1)
    nonzero = np.packbits(nonzero_bool, axis=1, bitorder="little")
    nonone = np.packbits(nonone_bool, axis=1, bitorder="little")
    counts = nonone_bool.sum(axis=1, dtype=np.uint64)
    offsets = np.empty(len(observations) + 1, dtype=np.uint64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    values = flat[nonone_bool].astype(np.float16, copy=False)
    return PackedObservations(nonzero, nonone, values, offsets, channels)


def unpack_observations(packed: PackedObservations) -> np.ndarray:
    length = len(packed.nonzero)
    elements = packed.channels * TILE_TYPES
    packed_bytes = (elements + 7) // 8
    if packed.nonzero.shape != (length, packed_bytes):
        raise ValueError(f"wrong nonzero shape: {packed.nonzero.shape}")
    if packed.nonone.shape != (length, packed_bytes):
        raise ValueError(f"wrong nonone shape: {packed.nonone.shape}")
    if packed.offsets.shape != (length + 1,):
        raise ValueError(f"wrong offsets shape: {packed.offsets.shape}")

    nonzero = np.unpackbits(
        packed.nonzero, axis=1, count=elements, bitorder="little"
    ).astype(np.float32, copy=False)
    nonone = np.unpackbits(
        packed.nonone, axis=1, count=elements, bitorder="little"
    ).astype(bool, copy=False)
    if int(packed.offsets[-1]) != int(nonone.sum()):
        raise ValueError("sparse value count does not match non-one mask")
    nonzero[nonone] = packed.values.astype(np.float32, copy=False)
    return nonzero.reshape(length, packed.channels, TILE_TYPES)


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
    if obs.channels == MODEL_INPUT_CHANNELS:
        input_schema = MODEL_INPUT_SCHEMA_ID
    elif obs.channels == OBS_CHANNELS:
        input_schema = LEGACY_MODEL_INPUT_SCHEMA_ID
    else:
        raise ValueError(f"unsupported observation channel count: {obs.channels}")
    return {
        "storage_format": np.asarray(STORAGE_FORMAT),
        "obs_channels": np.asarray(obs.channels, dtype=np.uint16),
        "model_input_schema": np.asarray(input_schema),
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
    channels = int(arrays.pop("obs_channels", np.asarray(OBS_CHANNELS)).item())
    arrays.pop("model_input_schema", None)
    arrays.pop("event_memory_schema", None)
    arrays["obs"] = unpack_observations(
        PackedObservations(
            arrays.pop("obs_nonzero"),
            arrays.pop("obs_nonone"),
            arrays.pop("obs_values"),
            arrays.pop("obs_offsets"),
            channels,
        )
    )
    arrays["action_mask"] = unpack_action_masks(arrays["action_mask"])
    return arrays


def write_packed_shard(path: str | Path, packed: dict[str, np.ndarray]) -> None:
    """Write a packed shard exactly as given, replacing the destination atomically."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(destination)
    payload = dict(packed)
    payload.setdefault("storage_format", np.asarray(STORAGE_FORMAT))
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        _replace_atomically(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _temporary_sibling(destination: Path) -> Path:
    """Return a collision-free temporary name beside an atomic destination."""

    return destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )


def _replace_atomically(source: Path, destination: Path, attempts: int = 8) -> None:
    """Replace a file despite brief Windows sharing violations at the destination."""

    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.01 * (attempt + 1))


def read_packed_shard(path: str | Path) -> dict[str, np.ndarray]:
    """Read a shard without expanding the packed observations."""

    with np.load(Path(path), allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    marker = arrays.get("storage_format")
    if marker is None or marker.item() not in SUPPORTED_STORAGE_FORMATS:
        raise ValueError(f"unsupported storage format in {path}")
    return normalize_packed_metadata(arrays)


def normalize_packed_metadata(
    packed: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Attach the explicit v8 input contract omitted by legacy v4 shards."""

    arrays = dict(packed)
    marker = arrays.get("storage_format")
    if marker is None or marker.item() not in SUPPORTED_STORAGE_FORMATS:
        raise ValueError("unsupported packed storage format")
    arrays.setdefault("obs_channels", np.asarray(OBS_CHANNELS, dtype=np.uint16))
    arrays.setdefault(
        "model_input_schema",
        np.asarray(LEGACY_MODEL_INPUT_SCHEMA_ID),
    )
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
    if any(PACK_CATALOG_FIELDS.intersection(part) for part in parts):
        raise ValueError("event catalogs must be deduplicated explicitly")
    names = set(parts[0])
    if any(set(part) != names for part in parts):
        raise ValueError("packed shard schemas differ")
    for name in PACKED_METADATA_FIELDS.intersection(names):
        if any(part[name].item() != parts[0][name].item() for part in parts[1:]):
            raise ValueError(f"packed shard metadata differs: {name}")
    result: dict[str, np.ndarray] = {
        name: parts[0][name] for name in PACKED_METADATA_FIELDS.intersection(names)
    }
    for name in sorted(names - PACKED_METADATA_FIELDS):
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
        **{name: packed[name] for name in PACKED_METADATA_FIELDS if name in packed},
        "obs_nonzero": packed["obs_nonzero"][order],
        "obs_nonone": packed["obs_nonone"][order],
        "obs_values": values,
        "obs_offsets": new_offsets.astype(offsets.dtype, copy=False),
    }
    for name, value in packed.items():
        if name in result or name in PACKED_METADATA_FIELDS:
            continue
        if name in PACK_CATALOG_FIELDS:
            result[name] = value
            continue
        result[name] = value[order]
    return result


def slice_packed(packed: dict[str, np.ndarray], start: int, stop: int) -> dict[str, np.ndarray]:
    """Take a range of samples out of a packed shard without expanding it.

    The packed observations store one offset per sample, so a contiguous range
    can be copied out and its offsets rebased. This is what lets training read
    a whole pack while only ever holding one batch of dense observations.
    """

    offsets = packed["obs_offsets"]
    count = _sample_count(packed)
    if start < 0 or stop > count or start > stop:
        raise ValueError(f"sample range {start}:{stop} is outside 0:{count}")

    result: dict[str, np.ndarray] = {
        **{name: packed[name] for name in PACKED_METADATA_FIELDS if name in packed},
        "obs_offsets": (offsets[start : stop + 1] - offsets[start]).astype(
            offsets.dtype, copy=False
        ),
        "obs_values": packed["obs_values"][int(offsets[start]) : int(offsets[stop])],
    }
    for name, value in packed.items():
        if name in result or name in PACKED_METADATA_FIELDS:
            continue
        if name in PACK_CATALOG_FIELDS:
            result[name] = value
            continue
        result[name] = value[start:stop]
    return result


def validate_event_references(
    catalog: np.ndarray, starts: np.ndarray, lengths: np.ndarray
) -> None:
    """Validate sample references into one deduplicated public-event catalog."""

    catalog = np.asarray(catalog)
    starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
    if catalog.ndim != 2 or catalog.shape[1] != EVENT_FIELDS:
        raise ValueError("event catalog has the wrong shape")
    if catalog.dtype != np.uint8:
        raise ValueError("event catalog must use uint8 tokens")
    if len(starts) != len(lengths):
        raise ValueError("event-history references do not share one sample count")
    if (starts < 0).any() or (lengths <= 0).any():
        raise ValueError("event-history references must be positive in-bounds ranges")
    if (starts + lengths > len(catalog)).any():
        raise ValueError("event-history reference lies outside the catalog")


def save_chunk_archive(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    chunk_samples: int,
    *,
    event_catalog: np.ndarray | None = None,
) -> dict[str, object]:
    """Stage one game as independently compressed sample chunks."""

    if chunk_samples <= 0:
        raise ValueError("chunk size must be positive")
    reference_fields = EVENT_REFERENCE_FIELDS.intersection(arrays)
    if reference_fields and reference_fields != EVENT_REFERENCE_FIELDS:
        raise ValueError("event histories require both start and length")
    if reference_fields:
        if event_catalog is None:
            raise ValueError("semantic samples require one event catalog")
        validate_event_references(
            event_catalog, arrays["history_start"], arrays["history_length"]
        )
    elif event_catalog is not None:
        raise ValueError("event catalog was supplied without sample references")
    packed = pack_shard_arrays(arrays)
    total = _sample_count(packed)
    destinations = Path(path)
    destinations.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(destinations)
    lengths: list[int] = []
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            if event_catalog is not None:
                event_buffer = io.BytesIO()
                np.save(event_buffer, np.asarray(event_catalog, dtype=np.uint8))
                archive.writestr(EVENT_CATALOG_MEMBER, event_buffer.getvalue())
            for member_index, start in enumerate(range(0, total, chunk_samples)):
                stop = min(total, start + chunk_samples)
                chunk = {
                    name: packed[name]
                    for name in PACKED_METADATA_FIELDS
                    if name in packed
                }
                for name, value in packed.items():
                    if name not in {*PACKED_METADATA_FIELDS, "obs_offsets", "obs_values"}:
                        chunk[name] = value[start:stop]
                # The sparse values are not one per sample, so they are cut by the
                # offset window rather than by the sample range. A chunk keeps one
                # more offset than it has samples so its own value runs stay
                # addressable after the rebase.
                offsets = packed["obs_offsets"][start : stop + 1]
                chunk["obs_offsets"] = offsets - offsets[0]
                chunk["obs_values"] = packed["obs_values"][int(offsets[0]) : int(offsets[-1])]
                buffer = io.BytesIO()
                np.savez_compressed(buffer, **chunk)
                archive.writestr(f"chunk_{member_index:05d}.npz", buffer.getvalue())
                lengths.append(stop - start)
            meta: dict[str, object] = {
                "format": STAGED_GAME_FORMAT,
                "modelInputSchema": packed["model_input_schema"].item(),
                "observationChannels": int(packed["obs_channels"].item()),
                "samples": total,
                "chunkSamples": chunk_samples,
                "chunkLengths": lengths,
            }
            if event_catalog is not None:
                meta.update(
                    {
                        "eventMemorySchema": EVENT_MEMORY_SCHEMA_ID,
                        "eventCount": len(event_catalog),
                    }
                )
            archive.writestr("meta.json", json.dumps(meta, separators=(",", ":")))
        _replace_atomically(temporary, destinations)
    finally:
        temporary.unlink(missing_ok=True)
    return meta


def read_chunk_archive_meta(path: str | Path) -> dict[str, object]:
    with zipfile.ZipFile(Path(path), "r") as archive:
        meta = json.loads(archive.read("meta.json"))
    if meta.get("format") not in {STAGED_GAME_FORMAT, *LEGACY_STAGED_GAME_FORMATS}:
        raise ValueError(f"unsupported staged game format in {path}")
    return meta


def read_chunk_payload(path: str | Path, member_index: int) -> bytes:
    with zipfile.ZipFile(Path(path), "r") as archive:
        return archive.read(f"chunk_{member_index:05d}.npz")


def read_chunk_event_catalog(path: str | Path) -> np.ndarray | None:
    """Read a staged game's shared event catalog, or ``None`` for legacy games."""

    meta = read_chunk_archive_meta(path)
    schema = meta.get("eventMemorySchema")
    if schema is None:
        return None
    if schema != EVENT_MEMORY_SCHEMA_ID:
        raise ValueError(f"unsupported event-memory schema in {path}")
    with (
        zipfile.ZipFile(Path(path), "r") as archive,
        archive.open(EVENT_CATALOG_MEMBER) as handle,
    ):
        catalog = np.load(handle, allow_pickle=False)
    if len(catalog) != int(meta.get("eventCount", -1)):
        raise ValueError(f"event catalog count does not match metadata in {path}")
    validate_event_references(
        catalog,
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
    )
    return catalog
