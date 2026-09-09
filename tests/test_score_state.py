import numpy as np
import pytest

from riichi_analysis_engine.score_state import PublicScoreState, relative_scores


def test_relative_scores_match_libriichi_controlled_player_order() -> None:
    assert np.array_equal(
        relative_scores([25_000, 30_000, 20_000, 25_000], 1),
        np.asarray([30_000, 20_000, 25_000, 25_000]),
    )


def test_public_score_state_uses_start_scores_and_riichi_payment() -> None:
    state = PublicScoreState()
    state.process({"type": "start_kyoku", "scores": [25_000, 30_000, 20_000, 25_000]})
    state.process({"type": "reach_accepted", "actor": 1})
    assert np.array_equal(state.relative(1), np.asarray([29_000, 20_000, 25_000, 25_000]))


def test_public_score_state_applies_settlement_deltas() -> None:
    state = PublicScoreState()
    state.process({"type": "start_kyoku", "scores": [25_000, 25_000, 25_000, 25_000]})
    state.process({"type": "hora", "deltas": [7_700, -7_700, 0, 0]})
    assert np.array_equal(state.relative(0), np.asarray([32_700, 17_300, 25_000, 25_000]))


def test_public_score_state_requires_start_scores() -> None:
    state = PublicScoreState()
    with pytest.raises(ValueError, match="before start_kyoku"):
        state.relative(0)
