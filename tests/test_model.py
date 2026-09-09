import torch

from riichi_analysis_engine.architecture import ModelArchitecture
from riichi_analysis_engine.constants import MORTAL_OBS_CHANNELS, OBS_CHANNELS
from riichi_analysis_engine.model import RiichiAnalysisModel, count_parameters
from riichi_analysis_engine.observation_layout import POLICY_CONTEXT_START


def test_output_shapes() -> None:
    model = RiichiAnalysisModel(
        architecture=ModelArchitecture(
            analysis_channels=32,
            analysis_blocks=2,
            analysis_latent_width=64,
            state_width=64,
            future_width=48,
            policy_context_channels=16,
            policy_context_blocks=1,
            policy_context_width=32,
            policy_width=48,
        )
    )
    outputs = model(torch.zeros(2, OBS_CHANNELS, 34))
    assert outputs["shanten"].shape == (2, 3, 7)
    assert outputs["deal_in_tile"].shape == (2, 3, 34)
    assert outputs["concealed_count"].shape == (2, 3, 34, 5)
    assert outputs["wall_count"].shape == (2, 34, 5)
    assert outputs["concealed_red_count"].shape == (2, 3, 3, 2)
    assert outputs["wall_red_count"].shape == (2, 3, 2)
    assert outputs["dora_distribution"].shape == (2, 3, 8)
    assert outputs["dora_point"].shape == (2, 3)
    assert outputs["score_distribution"].shape == (2, 3, 59)
    assert outputs["score_point"].shape == (2, 3)
    assert "outcome_any_win" not in outputs
    assert "outcome_winner" not in outputs
    assert "deal_in_player" not in outputs
    assert outputs["outcome"].shape == (2, 33)
    assert outputs["placement"].shape == (2, 24)
    assert outputs["policy"].shape == (2, 46)


def test_default_parameter_budget() -> None:
    model = RiichiAnalysisModel()
    assert count_parameters(model) == {
        "encoder": 9_529_040,
        "state": 1_228_862,
        "future": 666_512,
        "policy_context": 556_856,
        "policy": 685_486,
        "total": 12_666_756,
    }


def test_v4_summary_heads_remain_compatible() -> None:
    model = RiichiAnalysisModel(
        channels=32,
        blocks=2,
        state_width=64,
        future_width=48,
        format_version=4,
    )
    outputs = model(torch.zeros(1, MORTAL_OBS_CHANNELS, 34))
    assert outputs["outcome_any_win"].shape == (1,)
    assert outputs["outcome_winner"].shape == (1, 4)
    assert outputs["deal_in_player"].shape == (1, 4)
    assert outputs["outcome"].shape == (1, 33)


def test_legacy_parameter_budget_is_stable() -> None:
    model = RiichiAnalysisModel(format_version=1)
    assert count_parameters(model)["total"] == 26_430_494


def test_v2_parameter_budget_is_stable() -> None:
    model = RiichiAnalysisModel(format_version=2)
    assert count_parameters(model)["total"] == 26_576_604


def test_v6_architecture_round_trips_and_keeps_prediction_heads_state_only() -> None:
    architecture = ModelArchitecture(
        analysis_channels=16,
        analysis_blocks=1,
        analysis_latent_width=32,
        state_width=24,
        future_width=20,
        policy_context_channels=12,
        policy_context_blocks=1,
        policy_context_width=16,
        policy_width=24,
    )
    assert ModelArchitecture.from_dict(architecture.to_dict()) == architecture
    torch.manual_seed(7)
    model = RiichiAnalysisModel(architecture=architecture).eval()
    first = torch.randn(1, OBS_CHANNELS, 34)
    second = first.clone()
    second[:, POLICY_CONTEXT_START:] = torch.randn_like(second[:, POLICY_CONTEXT_START:])
    first_outputs = model(first)
    second_outputs = model(second)
    assert torch.equal(first_outputs["shanten"], second_outputs["shanten"])
    assert torch.equal(first_outputs["outcome"], second_outputs["outcome"])
    assert not torch.equal(first_outputs["policy"], second_outputs["policy"])
