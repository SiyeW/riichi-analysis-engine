import io
import shutil
import uuid
from pathlib import Path

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
    slice_packed,
    unpack_action_masks,
    unpack_observations,
    unpack_shard_arrays,
)


@pytest.fixture()
def scratch() -> Path:
    """A scratch directory inside the workspace, because the sandbox denies tmp_path."""

    root = Path(__file__).resolve().parents[1] / "runs" / "test-storage" / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def sample_arrays(count: int, *, kyoku: int = 0) -> dict[str, np.ndarray]:
    # The number of values that are neither zero nor one varies per sample, the
    # way real observations do, so slicing them by sample index cannot pass.
    obs = np.zeros((count, OBS_CHANNELS, TILE_TYPES), dtype=np.float32)
    for index in range(count):
        obs[index, index % OBS_CHANNELS, index % TILE_TYPES] = 1
        for extra in range(index % 4):
            obs[index, (index * 7 + extra) % OBS_CHANNELS, (index * 3 + extra) % TILE_TYPES] = 0.25
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


def test_rank_augmented_storage_rejects_older_schema_marker(scratch: Path) -> None:
    for marker in ("dual-bitpack-sparse-float16-v2", "dual-bitpack-sparse-float16-v3"):
        path = scratch / f"{marker}.npz"
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


def test_slice_packed_matches_the_unpacked_range() -> None:
    arrays = sample_arrays(9, kyoku=1)
    packed = pack_shard_arrays(arrays)
    expected = unpack_shard_arrays(packed)

    for start, stop in ((0, 9), (2, 5), (0, 1), (8, 9), (4, 4)):
        sliced = unpack_shard_arrays(slice_packed(packed, start, stop))
        assert len(sliced["policy"]) == stop - start
        np.testing.assert_allclose(sliced["obs"], expected["obs"][start:stop], rtol=0, atol=5e-4)
        for name in ("action_mask", "policy", "perspective", "event_index", "kyoku_index"):
            np.testing.assert_array_equal(sliced[name], expected[name][start:stop])


def test_slice_packed_rejects_a_range_outside_the_shard() -> None:
    packed = pack_shard_arrays(sample_arrays(3))
    for start, stop in ((-1, 2), (0, 4), (2, 1)):
        with pytest.raises(ValueError, match="outside"):
            slice_packed(packed, start, stop)


def test_chunk_archive_round_trip(scratch: Path) -> None:
    arrays = sample_arrays(20, kyoku=3)
    path = scratch / "game-000000.zip"
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


def test_each_chunk_holds_the_values_its_own_offsets_address(scratch: Path) -> None:
    # The sparse values are not one per sample. Cutting them by sample index
    # instead of by the offset window leaves a chunk addressing values it does
    # not have, which only shows up once a sample carries more than one value.
    arrays = sample_arrays(20, kyoku=3)
    path = scratch / "game-000000.zip"
    meta = save_chunk_archive(path, arrays, 16)
    packed = pack_shard_arrays(arrays)

    for member in range(len(meta["chunkLengths"])):
        with np.load(io.BytesIO(read_chunk_payload(path, member)), allow_pickle=False) as source:
            chunk = {name: source[name] for name in source.files}
        assert len(chunk["obs_values"]) == int(chunk["obs_offsets"][-1])
        assert len(chunk["obs_offsets"]) == len(chunk["policy"]) + 1
        np.testing.assert_array_equal(chunk["policy"], packed["policy"][
            member * 16 : member * 16 + len(chunk["policy"])
        ])
