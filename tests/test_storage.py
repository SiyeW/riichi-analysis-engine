import io

import numpy as np
import pytest

from riichi_analysis_engine.constants import ACTION_SPACE, OBS_CHANNELS, TILE_TYPES
from riichi_analysis_engine.storage import (
    PackedObservations,
    concatenate_packed,
    load_shard,
    pack_action_masks,
    pack_observations,
    pack_shard_arrays,
    permute_packed,
    read_chunk_archive_meta,
    read_chunk_payload,
    save_chunk_archive,
    unpack_action_masks,
    unpack_observations,
    unpack_shard_arrays,
)


def sample_arrays(count: int, *, kyoku: int = 0) -> dict[str, np.ndarray]:
    obs = np.zeros((count, OBS_CHANNELS, TILE_TYPES), dtype=np.float32)
    for index in range(count):
        obs[index, index % OBS_CHANNELS, index % TILE_TYPES] = 1
        obs[index, (index * 7) % OBS_CHANNELS, (index * 3) % TILE_TYPES] = 0.25
    masks = np.zeros((count, ACTION_SPACE), dtype=bool)
    masks[:, 0] = True
    masks[np.arange(count) % 2 == 0, 45] = True
    return {
        "obs": obs,
        "action_mask": masks,
        "policy": np.arange(count, dtype=np.int8),
        "perspective": np.arange(count, dtype=np.uint8) % 4,
        "event_index": np.arange(count, dtype=np.int32),
        "kyoku_index": np.full(count, kyoku, dtype=np.int32),
        "wall_count": np.full((count, TILE_TYPES, 5), 0.2, dtype=np.float32),
    }


def test_sparse_observation_round_trip() -> None:
    source = np.zeros((3, OBS_CHANNELS, TILE_TYPES), dtype=np.float32)
    source[0, 0, 0] = 1
    source[1, 7, :] = 0.25
    source[2, 500, 12] = 1.0144928
    packed = pack_observations(source)
    restored = unpack_observations(
        PackedObservations(packed.nonzero, packed.nonone, packed.values, packed.offsets)
    )
    np.testing.assert_allclose(restored, source, rtol=0, atol=5e-4)


def test_action_mask_round_trip() -> None:
    source = np.zeros((2, ACTION_SPACE), dtype=bool)
    source[0, [0, 34, 45]] = True
    source[1, [37, 43]] = True
    np.testing.assert_array_equal(unpack_action_masks(pack_action_masks(source)), source)


def test_rank_augmented_storage_rejects_older_schema_marker(tmp_path) -> None:
    for marker in ("dual-bitpack-sparse-float16-v2", "dual-bitpack-sparse-float16-v3"):
        path = tmp_path / f"{marker}.npz"
        np.savez_compressed(path, storage_format=np.asarray(marker))

        with pytest.raises(ValueError, match="unsupported storage format"):
            load_shard(path)


def test_packed_round_trip_keeps_every_array() -> None:
    arrays = sample_arrays(5, kyoku=2)
    restored = unpack_shard_arrays(pack_shard_arrays(arrays))

    np.testing.assert_allclose(restored["obs"], arrays["obs"], rtol=0, atol=5e-4)
    for name in ("action_mask", "policy", "perspective", "event_index", "kyoku_index"):
        np.testing.assert_array_equal(restored[name], arrays[name])
    np.testing.assert_array_equal(restored["wall_count"], arrays["wall_count"])


def test_concatenate_and_permute_preserve_every_sample() -> None:
    combined = concatenate_packed(
        [
            pack_shard_arrays(sample_arrays(4, kyoku=0)),
            pack_shard_arrays(sample_arrays(3, kyoku=1)),
        ]
    )
    assert len(combined["obs_offsets"]) - 1 == 7

    order = np.asarray([6, 0, 3, 1, 5, 2, 4])
    permuted = permute_packed(combined, order)
    expected = unpack_shard_arrays(combined)
    actual = unpack_shard_arrays(permuted)

    np.testing.assert_allclose(actual["obs"], expected["obs"][order], rtol=0, atol=5e-4)
    for name in ("action_mask", "policy", "perspective", "event_index", "kyoku_index"):
        np.testing.assert_array_equal(actual[name], expected[name][order])
    # The permutation must preserve each chunk's kyoku membership.
    assert sorted(actual["kyoku_index"].tolist()) == sorted(
        expected["kyoku_index"].tolist()
    )


def test_permute_rejects_an_incomplete_order() -> None:
    packed = pack_shard_arrays(sample_arrays(3))
    for order in (np.asarray([0, 1]), np.asarray([0, 1, 3])):
        with pytest.raises(ValueError):
            permute_packed(packed, order)


def test_chunk_archive_round_trip(tmp_path) -> None:
    arrays = sample_arrays(20, kyoku=3)
    path = tmp_path / "game-000000.zip"
    meta = save_chunk_archive(path, arrays, 16)

    assert meta["samples"] == 20
    assert meta["chunkLengths"] == [16, 4]
    assert read_chunk_archive_meta(path)["chunkLengths"] == [16, 4]

    parts = []
    for member in range(2):
        with np.load(io.BytesIO(read_chunk_payload(path, member)), allow_pickle=False) as source:
            parts.append({name: source[name] for name in source.files})
    restored = unpack_shard_arrays(concatenate_packed(parts))

    assert len(restored["policy"]) == 20
    np.testing.assert_allclose(restored["obs"], arrays["obs"], rtol=0, atol=5e-4)
    for name in ("action_mask", "policy", "perspective", "event_index", "kyoku_index"):
        np.testing.assert_array_equal(restored[name], arrays[name])
