import numpy as np

from riichi_analysis_engine.constants import MORTAL_OBS_CHANNELS, OBS_CHANNELS, TILE_TYPES
from riichi_analysis_engine.observation_layout import POLICY_CONTEXT_START
from riichi_analysis_engine.observations import add_all_player_ranks, all_player_rank_features


def test_all_player_rank_features_encode_each_relative_player() -> None:
    features = all_player_rank_features([25_000, 30_000, 20_000, 25_000])
    assert features.shape == (16, TILE_TYPES)
    # Relative seat order breaks equal-score ties, matching the model's rank
    # convention: player 1, then 0, then 3, then 2.
    assert np.all(features[0 * 4 + 1] == 1)
    assert np.all(features[1 * 4 + 0] == 1)
    assert np.all(features[2 * 4 + 3] == 1)
    assert np.all(features[3 * 4 + 2] == 1)


def test_rank_features_are_inserted_before_decision_context() -> None:
    source = np.zeros((MORTAL_OBS_CHANNELS, TILE_TYPES), dtype=np.float32)
    source[869].fill(3.0)
    source[870].fill(5.0)
    result = add_all_player_ranks(source, [25_000] * 4)

    assert result.shape == (OBS_CHANNELS, TILE_TYPES)
    assert np.all(result[869] == 3.0)
    assert np.all(result[POLICY_CONTEXT_START] == 5.0)
