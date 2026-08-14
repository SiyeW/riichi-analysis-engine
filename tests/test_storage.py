import numpy as np

from riichi_analysis_engine.constants import ACTION_SPACE, OBS_CHANNELS, TILE_TYPES
from riichi_analysis_engine.storage import (
    PackedObservations,
    pack_action_masks,
    pack_observations,
    unpack_action_masks,
    unpack_observations,
)


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

