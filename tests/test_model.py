import torch

from riichi_analysis_engine.model import RiichiAnalysisModel, count_parameters


def test_output_shapes() -> None:
    model = RiichiAnalysisModel(channels=32, blocks=2, state_width=64, future_width=48)
    outputs = model(torch.zeros(2, 1012, 34))
    assert outputs["shanten"].shape == (2, 3, 7)
    assert outputs["deal_in_tile"].shape == (2, 3, 34)
    assert outputs["concealed_count"].shape == (2, 3, 34, 5)
    assert outputs["wall_count"].shape == (2, 34, 5)
    assert outputs["target"].shape == (2, 4, 4)
    assert outputs["placement"].shape == (2, 24)
    assert outputs["policy"].shape == (2, 46)


def test_default_parameter_budget() -> None:
    model = RiichiAnalysisModel()
    assert count_parameters(model) == {
        "encoder": 23_663_488,
        "state": 1_875_750,
        "future": 835_647,
        "policy": 47_150,
        "total": 26_422_035,
    }

