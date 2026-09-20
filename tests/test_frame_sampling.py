from types import SimpleNamespace

import pytest

from riichi_analysis_engine.frame_sampling import (
    FrameSamplingPlan,
    has_rare_legal_action,
    is_terminal_preceding,
)


def test_sampling_is_deterministic_and_identity_sensitive() -> None:
    plan = FrameSamplingPlan(seed=17, ordinary_rate=0.5)
    first = [plan.keep("source-a/game", index, "ordinary") for index in range(64)]

    assert first == [plan.keep("source-a/game", index, "ordinary") for index in range(64)]
    assert first != [plan.keep("source-b/game", index, "ordinary") for index in range(64)]
    assert any(first)
    assert not all(first)


def test_decisive_frames_are_always_retained() -> None:
    plan = FrameSamplingPlan(
        seed=17,
        rare_action_rate=0.0,
        state_change_rate=0.0,
        ordinary_rate=0.0,
    )
    assert plan.keep("game", 3, "decisive")


def test_terminal_and_rare_action_classification() -> None:
    events = [
        {"type": "dahai", "actor": 0, "pai": "4s"},
        {"type": "hora", "actor": 3, "target": 0},
        {"type": "end_kyoku"},
    ]
    assert is_terminal_preceding(events, 0)
    assert has_rare_legal_action([SimpleNamespace(can_ron_agari=True)])
    assert not has_rare_legal_action([SimpleNamespace(can_ron_agari=False)])


def test_sampling_rates_are_validated() -> None:
    with pytest.raises(ValueError, match="ordinary"):
        FrameSamplingPlan(seed=1, ordinary_rate=1.1)

