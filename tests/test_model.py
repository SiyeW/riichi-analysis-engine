import torch

from riichi_analysis_engine.model import RiichiAnalysisModel, count_parameters


def test_output_shapes() -> None:
    model = RiichiAnalysisModel(channels=32, blocks=2, state_width=64, future_width=48)
    outputs = model(torch.zeros(2, 1012, 34))
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
        "encoder": 23_663_488,
        "state": 1_900_350,
        "future": 996_368,
        "policy": 47_150,
        "total": 26_607_356,
    }


def test_v4_summary_heads_remain_compatible() -> None:
    model = RiichiAnalysisModel(
        channels=32,
        blocks=2,
        state_width=64,
        future_width=48,
        format_version=4,
    )
    outputs = model(torch.zeros(1, 1012, 34))
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
