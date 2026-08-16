import pytest
import torch

from riichi_analysis_engine.losses import _masked_mean, score_class_indices


def test_masked_mean_does_not_multiply_non_finite_unselected_values() -> None:
    values = torch.tensor([2.0, float("inf")])
    mask = torch.tensor([True, False])

    assert _masked_mean(values, mask).item() == 2.0


def test_masked_mean_of_empty_selection_is_zero() -> None:
    values = torch.tensor([1.0, 2.0])
    mask = torch.tensor([False, False])

    assert _masked_mean(values, mask).item() == 0.0


def test_score_class_indices_use_actual_settlement_values() -> None:
    indices = score_class_indices(torch.tensor([1000, 7700, 288000]))
    assert indices.tolist() == sorted(indices.tolist())


def test_score_class_indices_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="700"):
        score_class_indices(torch.tensor([700]))
