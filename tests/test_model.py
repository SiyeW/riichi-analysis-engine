import torch

from riichi_analysis_engine.architecture import (
    ModelArchitecture,
    StructuredModelArchitecture,
)
from riichi_analysis_engine.constants import MORTAL_OBS_CHANNELS, OBS_CHANNELS
from riichi_analysis_engine.model import RiichiAnalysisModel, count_parameters
from riichi_analysis_engine.model_input import ANALYSIS_CHANNELS, MODEL_INPUT_CHANNELS
from riichi_analysis_engine.observation_layout import POLICY_CONTEXT_START


def test_output_shapes() -> None:
    model = RiichiAnalysisModel(
        format_version=7,
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
        ),
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
    assert "score_point" not in outputs
    assert "outcome_any_win" not in outputs
    assert "outcome_winner" not in outputs
    assert "deal_in_player" not in outputs
    assert outputs["outcome"].shape == (2, 33)
    assert outputs["placement"].shape == (2, 24)
    assert outputs["policy"].shape == (2, 46)


def test_default_parameter_budget() -> None:
    model = RiichiAnalysisModel()
    assert count_parameters(model) == {
        "shared": 12_762_080,
        "opponent": 11_971_614,
        "hidden": 4_516_400,
        "value": 4_563_532,
        "kyoku": 4_084_646,
        "match": 3_274_108,
        "policy_context": 1_260_086,
        "policy": 12_295_118,
        "total": 54_727_584,
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
    model = RiichiAnalysisModel(format_version=6, architecture=architecture).eval()
    first = torch.randn(1, OBS_CHANNELS, 34)
    second = first.clone()
    second[:, POLICY_CONTEXT_START:] = torch.randn_like(
        second[:, POLICY_CONTEXT_START:]
    )
    first_outputs = model(first)
    second_outputs = model(second)
    assert torch.equal(first_outputs["shanten"], second_outputs["shanten"])
    assert torch.equal(first_outputs["outcome"], second_outputs["outcome"])
    assert not torch.equal(first_outputs["policy"], second_outputs["policy"])


def test_v8_family_towers_keep_global_waits_and_structured_outputs() -> None:
    architecture = StructuredModelArchitecture(
        shared_channels=16,
        shared_blocks=1,
        family_latent_width=32,
        opponent_latent_width=40,
        policy_latent_width=48,
        opponent_blocks=1,
        hidden_blocks=1,
        value_blocks=1,
        kyoku_blocks=1,
        match_blocks=1,
        policy_blocks=1,
        task_width=24,
        tile_width=8,
        policy_context_channels=8,
        policy_context_blocks=1,
        policy_context_width=16,
        policy_width=32,
    )
    assert StructuredModelArchitecture.from_dict(architecture.to_dict()) == architecture
    model = RiichiAnalysisModel(format_version=8, architecture=architecture)

    outputs = model(torch.zeros(2, OBS_CHANNELS, 34))

    assert outputs["shanten"].shape == (2, 3, 7)
    assert outputs["deal_in_tile"].shape == (2, 3, 34)
    assert outputs["hidden_source_affinity"].shape == (2, 4, 34)
    assert outputs["hidden_red_source"].shape == (2, 3, 4)
    assert outputs["dora_tail"].shape == (2, 3)
    assert outputs["kyoku_accounts"].shape == (2, 5)
    assert model.wait_head.in_features == architecture.opponent_latent_width
    assert count_parameters(model)["total"] == sum(
        parameter.numel() for parameter in model.parameters()
    )


def test_v9_policy_context_cannot_change_analysis_outputs() -> None:
    architecture = StructuredModelArchitecture(
        shared_channels=8,
        shared_blocks=1,
        family_latent_width=16,
        opponent_latent_width=20,
        policy_latent_width=24,
        opponent_blocks=1,
        hidden_blocks=1,
        value_blocks=1,
        kyoku_blocks=1,
        match_blocks=1,
        policy_blocks=1,
        task_width=12,
        tile_width=6,
        policy_context_channels=4,
        policy_context_blocks=1,
        policy_context_width=8,
        policy_width=16,
    )
    model = RiichiAnalysisModel(format_version=9, architecture=architecture).eval()
    first = torch.randn(2, MODEL_INPUT_CHANNELS, 34)
    second = first.clone()
    second[:, ANALYSIS_CHANNELS:] = torch.randn_like(second[:, ANALYSIS_CHANNELS:])

    first_outputs = model(first)
    second_outputs = model(second)
    for name in first_outputs.keys() - {"policy"}:
        assert torch.equal(first_outputs[name], second_outputs[name])
    assert not torch.equal(first_outputs["policy"], second_outputs["policy"])
