from types import SimpleNamespace

import pytest

from riichi_analysis_engine.frame_sampling import (
    FrameSamplingPlan,
    has_rare_legal_action,
)


def test_sampling_is_deterministic_and_identity_sensitive() -> None:
    plan = FrameSamplingPlan(seed=17, ordinary_rate=0.5)
    first = [plan.keep("source-a/game", index, 2, "analysis_ordinary") for index in range(64)]

    assert first == [
        plan.keep("source-a/game", index, 2, "analysis_ordinary") for index in range(64)
    ]
    assert first != [
        plan.keep("source-b/game", index, 2, "analysis_ordinary") for index in range(64)
    ]
    assert first != [
        plan.keep("source-a/game", index, 3, "analysis_ordinary") for index in range(64)
    ]
    assert any(first)
    assert not all(first)


def test_exact_baseline_anchors_are_always_retained() -> None:
    plan = FrameSamplingPlan(
        seed=17,
        rare_action_rate=0.0,
        state_change_rate=0.0,
        ordinary_rate=0.0,
    )
    assert plan.keep("game", 3, 1, "analysis_baseline_anchor")


def test_policy_and_analysis_classification_are_independent() -> None:
    event = {"type": "dahai", "actor": 0, "pai": "4s"}
    candidate = SimpleNamespace(can_ron_agari=True)

    assert has_rare_legal_action(candidate)
    assert not has_rare_legal_action(SimpleNamespace(can_ron_agari=False))
    assert FrameSamplingPlan.policy_stratum(event, candidate) == "policy_rare_action"
    assert (
        FrameSamplingPlan.analysis_stratum(event, baseline_anchor=False)
        == "analysis_ordinary"
    )


def test_sampling_metadata_separates_policy_and_analysis_rates() -> None:
    metadata = FrameSamplingPlan(seed=7).metadata()

    assert metadata["schema"] == "riichi-analysis-frame-sampling-v2"
    assert metadata["policyRates"]["rare_action"] == 1.0
    assert metadata["analysisRates"]["baseline_anchor"] == 1.0


def test_sampling_rates_are_validated() -> None:
    with pytest.raises(ValueError, match="ordinary"):
        FrameSamplingPlan(seed=1, ordinary_rate=1.1)
