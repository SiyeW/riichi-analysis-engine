import numpy as np
import pytest
import torch

from riichi_analysis_engine.architecture import ModelArchitecture
from riichi_analysis_engine.constants import ACTION_SPACE, MORTAL_OBS_CHANNELS, TILE_TYPES
from riichi_analysis_engine.observation_layout import ANALYSIS_CHANNELS
from riichi_analysis_engine.model import RiichiAnalysisModel
from riichi_analysis_engine.prediction_values import DORA_VALUES, SCORE_VALUES
from riichi_analysis_engine.runtime import (
    AnalysisRuntime,
    _distribution,
    _prediction_from_distribution,
    _valued_distribution,
    best_candidate,
    candidate_action_index,
    complete_candidate_policy,
)
from riichi_analysis_engine.score_state import PublicScoreState


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


def test_probability_serializers_preserve_exact_endpoints() -> None:
    probabilities = np.asarray([1.0, 0.0, 0.0])
    assert _distribution(probabilities) == [
        {"value": 0, "probability": 1.0},
        {"value": 1, "probability": 0.0},
        {"value": 2, "probability": 0.0},
    ]
    assert _prediction_from_distribution(probabilities) == {
        "distribution": [
            {"value": 0, "probability": 1.0},
            {"value": 1, "probability": 0.0},
            {"value": 2, "probability": 0.0},
        ],
        "expectedValue": 0.0,
    }


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


def _ankan(candidate_id: str, tile: str) -> dict:
    return {
        "candidateId": candidate_id,
        "action": {"type": "ankan", "consumed": [tile] * 4},
    }


def test_multiple_kan_candidates_use_conditional_policy_not_equal_shares() -> None:
    primary_logits = np.full(46, -100.0)
    primary_logits[0] = 0.0
    primary_logits[42] = 2.0
    kan_logits = np.full(46, -100.0)
    kan_logits[0] = 0.0  # 1m
    kan_logits[1] = np.log(3.0)  # 2m
    candidates = [
        {"candidateId": "discard", "action": {"type": "dahai", "pai": "1m"}},
        _ankan("kan-1m", "1m"),
        _ankan("kan-2m", "2m"),
    ]
    mask = np.zeros(46, dtype=bool)
    mask[[0, 1]] = True

    values = complete_candidate_policy(
        primary_logits,
        candidates,
        kan_selection_logits=kan_logits,
        kan_selection_mask=mask,
    )

    assert values["kan-2m"] == pytest.approx(values["kan-1m"] * 3.0)
    assert sum(values.values()) == pytest.approx(1.0)
    assert best_candidate(candidates, values)["candidateId"] == "kan-2m"


def test_kan_candidate_order_does_not_change_values_or_tie_breaking() -> None:
    primary_logits = np.zeros(46)
    primary_logits[42] = 1.0
    kan_logits = np.zeros(46)
    mask = np.zeros(46, dtype=bool)
    mask[[0, 1]] = True
    candidates = [_ankan("kan-2m", "2m"), _ankan("kan-1m", "1m")]
    first = complete_candidate_policy(
        primary_logits,
        candidates,
        kan_selection_logits=kan_logits,
        kan_selection_mask=mask,
    )
    reversed_candidates = list(reversed(candidates))
    second = complete_candidate_policy(
        primary_logits,
        reversed_candidates,
        kan_selection_logits=kan_logits,
        kan_selection_mask=mask,
    )

    assert first == second
    assert best_candidate(candidates, first)["candidateId"] == "kan-1m"
    assert best_candidate(reversed_candidates, second)["candidateId"] == "kan-1m"


def test_kan_candidate_must_be_available_in_conditional_state() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        complete_candidate_policy(
            np.zeros(46),
            [_ankan("kan-1m", "1m")],
            kan_selection_logits=np.zeros(46),
            kan_selection_mask=np.zeros(46, dtype=bool),
        )


def test_v6_runtime_adds_ranks_from_public_score_events() -> None:
    class FakePlayerState:
        player_id = 1

        @staticmethod
        def encode_obs(version: int, at_kan_select: bool) -> tuple[np.ndarray, np.ndarray]:
            assert version == 4
            assert not at_kan_select
            return (
                np.zeros((MORTAL_OBS_CHANNELS, TILE_TYPES), dtype=np.float32),
                np.ones(ACTION_SPACE, dtype=bool),
            )

    runtime = AnalysisRuntime.__new__(AnalysisRuntime)
    runtime.format_version = 6
    scores = PublicScoreState()
    scores.process({"type": "start_kyoku", "scores": [25_000, 30_000, 20_000, 25_000]})

    observation, _mask = runtime._encode_observation(
        FakePlayerState(), scores, at_kan_select=False
    )

    # PlayerState is relative to player 1, whose rank is first. The rank block
    # begins immediately after the shared state features.
    assert np.all(observation[ANALYSIS_CHANNELS - 16] == 1.0)


def test_v6_runtime_reconstructs_weight_architecture(tmp_path, monkeypatch) -> None:
    architecture = ModelArchitecture(
        analysis_channels=8,
        analysis_blocks=1,
        analysis_latent_width=16,
        state_width=16,
        future_width=16,
        policy_context_channels=4,
        policy_context_blocks=1,
        policy_context_width=8,
        policy_width=16,
    )
    checkpoint = tmp_path / "weights.pt"
    torch.save(
        {
            "format": "riichi-analysis-model-v6",
            "model": RiichiAnalysisModel(architecture=architecture).state_dict(),
            "architecture": {
                "model": architecture.to_dict(),
                "predictionValues": {
                    "dora": list(DORA_VALUES),
                    "score": list(SCORE_VALUES),
                },
            },
        },
        checkpoint,
    )
    monkeypatch.setattr("riichi_analysis_engine.runtime._load_player_state", lambda: object)

    runtime = AnalysisRuntime(checkpoint, "cpu")

    assert runtime.format_version == 6
    assert runtime.model.architecture == architecture
