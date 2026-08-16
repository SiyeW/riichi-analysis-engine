import torch

from riichi_analysis_engine.model import RiichiAnalysisModel, count_parameters


def test_output_shapes() -> None:
    model = RiichiAnalysisModel(channels=32, blocks=2, state_width=64, future_width=48)
    outputs = model(torch.zeros(2, 1012, 34))
    assert outputs["shanten"].shape == (2, 3, 7)
    assert outputs["deal_in_tile"].shape == (2, 3, 34)
    assert outputs["concealed_count"].shape == (2, 3, 34, 5)
    assert outputs["wall_count"].shape == (2, 34, 5)
    assert outputs["dora_distribution"].shape == (2, 3, 8)
    assert outputs["dora_point"].shape == (2, 3)
    assert outputs["score_distribution"].shape == (2, 3, 59)
    assert outputs["score_point"].shape == (2, 3)
    assert outputs["outcome_any_win"].shape == (2,)
    assert outputs["outcome_winner"].shape == (2, 4)
    assert outputs["target"].shape == (2, 4, 4)
    assert outputs["placement"].shape == (2, 24)
    assert outputs["policy"].shape == (2, 46)


def test_default_parameter_budget() -> None:
    model = RiichiAnalysisModel()
    assert count_parameters(model) == {
        "encoder": 23_663_488,
        "state": 1_875_750,
        "future": 990_216,
        "policy": 47_150,
        "total": 26_576_604,
    }


def test_legacy_parameter_budget_is_stable() -> None:
    model = RiichiAnalysisModel(format_version=1)
    assert count_parameters(model)["total"] == 26_430_494
