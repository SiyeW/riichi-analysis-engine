import pytest
import torch

from riichi_analysis_engine.architecture import ModelArchitecture
from riichi_analysis_engine.constants import ACTION_SPACE, OBS_CHANNELS, TILE_TYPES
from riichi_analysis_engine.losses import (
    LOSS_TERMS,
    LearnedUncertaintyBalancer,
    _masked_mean,
    score_class_indices,
    multitask_loss,
)
from riichi_analysis_engine.model import RiichiAnalysisModel


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


def test_uncertainty_balancer_learns_only_from_active_terms() -> None:
    balancer = LearnedUncertaintyBalancer()
    losses = {name: torch.tensor(2.0, requires_grad=True) for name in LOSS_TERMS}
    active = {name: name != "dora_point" for name in LOSS_TERMS}

    total, weights = balancer(losses, active)
    total.backward()

    assert set(weights) == set(LOSS_TERMS)
    inactive_index = LOSS_TERMS.index("dora_point")
    assert balancer.log_variance.grad is not None
    assert balancer.log_variance.grad[inactive_index].item() == 0.0
    assert balancer.log_variance.grad[LOSS_TERMS.index("policy")].item() != 0.0


def test_rank_augmented_model_and_balanced_losses_backpropagate() -> None:
    """Keep the v6 observation split, heads, labels, and balancer connected."""

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
    model = RiichiAnalysisModel(architecture=architecture)
    batch_size = 2
    action_mask = torch.ones(batch_size, ACTION_SPACE, dtype=torch.bool)
    batch = {
        "obs": torch.zeros(batch_size, OBS_CHANNELS, TILE_TYPES),
        "action_mask": action_mask,
        "policy": torch.tensor([0, 1]),
        "shanten": torch.zeros(batch_size, 3, dtype=torch.long),
        "furiten_no_yaku": torch.zeros(batch_size, 3),
        "deal_in_tile": torch.zeros(batch_size, 3, 34),
        "concealed_count": torch.zeros(batch_size, 3, 34, dtype=torch.long),
        "concealed_red_count": torch.zeros(batch_size, 3, 3, dtype=torch.long),
        "wall_count": torch.zeros(batch_size, 34, dtype=torch.long),
        "wall_red_count": torch.zeros(batch_size, 3, dtype=torch.long),
        "dora": torch.zeros(batch_size, 3, dtype=torch.long),
        "score": torch.zeros(batch_size, 3, dtype=torch.long),
        "winner_mask": torch.zeros(batch_size, 3, dtype=torch.bool),
        "outcome": torch.zeros(batch_size, dtype=torch.long),
        "kyoku_delta": torch.zeros(batch_size, 4),
        "placement": torch.zeros(batch_size, dtype=torch.long),
        "match_score": torch.zeros(batch_size, 4),
    }

    total, losses, active, weights = multitask_loss(
        model(batch["obs"]), batch, LearnedUncertaintyBalancer()
    )
    assert torch.isfinite(total)
    assert set(losses) == set(LOSS_TERMS)
    assert set(weights) == set(LOSS_TERMS)
    assert not active["dora_distribution"]
    assert not active["score_distribution"]
    total.backward()
    assert model.encoder.net[0].weight.grad is not None
