import numpy as np

from riichi_analysis_engine.runtime import (
    AnalysisRuntime,
    _valued_distribution,
    candidate_action_index,
)


def test_candidate_action_mapping() -> None:
    assert candidate_action_index({"type": "dahai", "pai": "5mr"}) == 34
    assert candidate_action_index({"type": "reach"}) == 37
    assert candidate_action_index(
        {"type": "chi", "pai": "3m", "consumed": ["1m", "2m"]}
    ) == 40
    assert candidate_action_index({"type": "pon"}) == 41
    assert candidate_action_index({"type": "ankan"}) == 42
    assert candidate_action_index({"type": "hora"}) == 43
    assert candidate_action_index({"type": "ryukyoku"}) == 44
    assert candidate_action_index({"type": "none"}) == 45


def test_open_ended_dora_bucket_remains_a_string() -> None:
    distribution = _valued_distribution(
        np.asarray([0.25, 0.75]),
        (0, "7+"),
    )
    assert distribution == [
        {"value": 0, "probability": 0.25},
        {"value": "7+", "probability": 0.75},
    ]


def test_v2_representations_follow_the_negotiated_protocol() -> None:
    runtime = AnalysisRuntime.__new__(AnalysisRuntime)
    runtime.format_version = 2
    assert runtime.representations("opponent-dora-count", 1) == ["expected-value"]
    assert runtime.representations("opponent-dora-count", 2) == [
        "distribution",
        "point-estimate",
    ]
    assert runtime.representations("opponent-score", 1) == [
        "distribution",
        "expected-value",
    ]
