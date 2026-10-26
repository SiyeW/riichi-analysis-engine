import pytest
import torch

from riichi_analysis_engine.constants import ACTION_SPACE, OBS_CHANNELS, TILE_TYPES
from riichi_analysis_engine.losses import (
    LOSS_TERMS_V8,
    BatchSupervision,
    LearnedUncertaintyBalancer,
    multitask_loss,
)
from riichi_analysis_engine.observation_layout import JIKAZE_CHANNEL, WIND_TILE_START
from riichi_analysis_engine.ordered_losses import (
    cdf_distance,
    hidden_count_distance,
    placement_distance,
)
from riichi_analysis_engine.prediction_values import (
    PLACEMENT_PERMUTATIONS,
    SCORE_VALUES,
)


def test_cdf_point_errors_use_real_nonuniform_distances_and_units():
    p = torch.eye(3)
    distance = cdf_distance(
        p, torch.full((3,), 1000), values=(1000, 2000, 8000), unit=1000
    )
    assert distance.tolist() == [0, 1, 7]
    assert (
        cdf_distance(
            torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([100]), values=(0, 1, 2)
        ).item()
        == 0
    )


def test_equal_means_do_not_hide_distribution_error():
    p = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0, 0.5]])
    assert (p @ torch.arange(5.0)).tolist() == [2, 2]
    assert cdf_distance(p, torch.tensor([2, 2])).tolist() == [0, 1]


def test_open_dora_bucket_scores_known_boundaries_only():
    p = torch.zeros(2, 8)
    p[:, 7] = 1
    assert cdf_distance(p, torch.tensor([7, 12])).tolist() == [0, 0]


def test_placement_uses_rank_marginals_not_permutation_ids():
    p = torch.eye(24)[[0, 1, 23]]
    assert PLACEMENT_PERMUTATIONS[1] == (0, 1, 3, 2)
    assert placement_distance(p, torch.zeros(3, dtype=torch.long)).tolist() == [
        0,
        0.5,
        2,
    ]


def test_hidden_anchors_use_soft_total_targets_and_ignore_empty_inventory():
    joint = torch.zeros(2, 4, 34, 10)
    joint[..., 0] = 1
    joint[:, :, 0, 0] = 0.5
    joint[:, :, 0, 4] = 0.5  # total2/red0, not ordinal state-index4
    baseline = joint.clone()
    labels = torch.zeros(2, 4, 34, dtype=torch.long)
    inventory = torch.zeros(2, 34)
    inventory[:, 0] = 2
    assert (
        hidden_count_distance(
            joint, baseline, labels, inventory, torch.ones(2, dtype=torch.bool)
        ).item()
        == 0
    )
    assert (
        hidden_count_distance(
            joint, baseline, labels, inventory, torch.zeros(2, dtype=torch.bool)
        ).item()
        == 0.5
    )
    assert (
        hidden_count_distance(
            joint, baseline, labels, inventory * 0, torch.zeros(2, dtype=torch.bool)
        ).item()
        == 0
    )


def structured_batch():
    torch.manual_seed(73)
    torch.set_num_threads(1)
    shapes = {
        "policy": (2, ACTION_SPACE),
        "shanten": (2, 3, 7),
        "furiten_no_yaku": (2, 3),
        "deal_in_tile": (2, 3, 34),
        "hidden_joint_residual": (2, 4, 34, 10),
        "dora_distribution": (2, 3, 8),
        "dora_tail": (2, 3),
        "score_distribution": (2, 3, len(SCORE_VALUES)),
        "outcome": (2, 33),
        "kyoku_accounts": (2, 5),
        "placement": (2, 24),
        "match_score": (2, 4),
    }
    outputs = {k: torch.randn(shape, requires_grad=True) for k, shape in shapes.items()}
    batch = {
        "obs": torch.zeros(2, OBS_CHANNELS, TILE_TYPES),
        "action_mask": torch.ones(2, ACTION_SPACE, dtype=torch.bool),
        "policy": torch.tensor([0, 1]),
        "analysis_active": torch.ones(2, dtype=torch.bool),
        "shanten": torch.tensor([[0, 1, 2], [0, 2, 1]]),
        "furiten_no_yaku": torch.zeros(2, 3),
        "deal_in_tile": torch.zeros(2, 3, 34),
        "concealed_count": torch.zeros(2, 3, 34, dtype=torch.long),
        "concealed_red_count": torch.zeros(2, 3, 3, dtype=torch.long),
        "wall_count": torch.zeros(2, 34, dtype=torch.long),
        "wall_red_count": torch.zeros(2, 3, dtype=torch.long),
        "hidden_baseline_anchor": torch.tensor([True, False]),
        "current_concealed_dora": torch.tensor([[1, 8, 3], [0, 2, 1]]),
        "dora": torch.zeros(2, 3, dtype=torch.long),
        "score": torch.tensor([[1000, 0, 0], [1000, 0, 0]]),
        "winner_mask": torch.tensor([[True, False, False], [True, False, False]]),
        "outcome": torch.zeros(2, dtype=torch.long),
        "kyoku_delta": torch.zeros(2, 4),
        "placement": torch.tensor([0, 23]),
        "match_score": torch.full((2, 4), 25000.0),
    }
    batch["obs"][:, JIKAZE_CHANNEL, WIND_TILE_START] = 1
    batch["concealed_count"][:, :, 0] = torch.tensor([2, 0, 1])
    batch["concealed_count"][:, :, 1] = torch.tensor([0, 2, 1])
    batch["wall_count"][:, :2] = 1
    return outputs, batch


def test_five_additions_preserve_other_losses_and_twelve_balancer_parameters():
    outputs, batch = structured_batch()
    balancer = LearnedUncertaintyBalancer(LOSS_TERMS_V8)
    _, old, _, _ = multitask_loss(outputs, batch, balancer)
    components = {}
    total, new, active, weights = multitask_loss(
        outputs,
        batch,
        balancer,
        ordered_distance=True,
        loss_components=components,
        supervision=BatchSupervision.from_cpu(batch),
    )
    assert len(new) == len(weights) == len(balancer.log_variance) == 12
    assert tuple(new) == LOSS_TERMS_V8
    selected = {
        "shanten",
        "hidden_allocation",
        "dora_distribution",
        "score_distribution",
        "placement",
    }
    for name in new:
        if name in selected:
            assert torch.equal(old[name], components["base/" + name])
            assert components["distance/" + name] >= 0
            assert torch.equal(new[name], old[name] + components["distance/" + name])
        else:
            assert torch.equal(old[name], new[name])
    total.backward()
    assert all(
        v.grad is not None and torch.isfinite(v.grad).all() for v in outputs.values()
    )
    assert torch.isfinite(balancer.log_variance.grad).all()
    assert all(active.values())


def test_analysis_mask_cpu_metadata_and_empty_winners_keep_semantics():
    outputs, batch = structured_batch()
    batch["analysis_active"][1] = False
    batch["winner_mask"][:] = False
    components = {}
    _, losses, active, _ = multitask_loss(
        outputs,
        batch,
        LearnedUncertaintyBalancer(LOSS_TERMS_V8),
        ordered_distance=True,
        loss_components=components,
        supervision=BatchSupervision.from_cpu(batch),
    )
    one = {k: v[:1] for k, v in batch.items()}
    one_outputs = {k: v[:1] for k, v in outputs.items()}
    _, other, _, _ = multitask_loss(
        one_outputs,
        one,
        LearnedUncertaintyBalancer(LOSS_TERMS_V8),
        ordered_distance=True,
    )
    for name in LOSS_TERMS_V8[1:]:
        assert torch.equal(losses[name], other[name])
    assert not active["score_distribution"] and losses["score_distribution"] == 0
    assert components["distance/score_distribution"] == 0
    assert active["dora_distribution"] and active["dora_tail"]


def test_policy_only_rows_add_no_distance_or_analysis_gradient():
    outputs, batch = structured_batch()
    batch["analysis_active"][:] = False
    components = {}
    total, losses, active, _ = multitask_loss(
        outputs,
        batch,
        LearnedUncertaintyBalancer(LOSS_TERMS_V8),
        ordered_distance=True,
        loss_components=components,
    )
    assert all(not active[k] and losses[k] == 0 for k in LOSS_TERMS_V8[1:])
    assert len(components) == 10 and all(v == 0 for v in components.values())
    total.backward()
    assert all(v.grad is None for k, v in outputs.items() if k != "policy")


@pytest.mark.parametrize("unit", [0, -1, float("inf"), float("nan")])
def test_invalid_units_rejected(unit):
    with pytest.raises(ValueError):
        cdf_distance(torch.ones(2) / 2, torch.tensor(0), unit=unit)
