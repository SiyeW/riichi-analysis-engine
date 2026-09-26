"""Regression oracle for removing unused moments, not changing the projection."""

import pytest
import torch

from riichi_analysis_engine.joint_counts import (
    _values,
    family_inventory,
    joint_count_baseline,
    project_joint_counts,
)


def reference_projection(residual, inventory, capacities):
    """Original full-moment computation, intentionally retained only in tests."""
    baseline = joint_count_baseline(inventory, capacities)
    logits = (
        baseline.clamp_min(torch.finfo(torch.float32).tiny).log() + residual.float()
    ).masked_fill(baseline == 0, -torch.inf)
    total, red = _values(logits)
    column_target, red_target = family_inventory(inventory.float())
    row_bias = logits.new_zeros((len(logits), 4, 1, 1))
    column_bias = logits.new_zeros((len(logits), 1, 34, 1))
    red_bias = torch.zeros_like(column_bias)

    def moments():
        p = (logits + total * (row_bias + column_bias) + red * red_bias).softmax(-1)
        mt, mr = (p * total).sum(-1), (p * red).sum(-1)
        dt, dr = total - mt.unsqueeze(-1), red - mr.unsqueeze(-1)
        return (
            p,
            mt,
            mr,
            (p * dt.square()).sum(-1),
            (p * dr.square()).sum(-1),
            (p * dt * dr).sum(-1),
        )

    for _ in range(32):
        _, mt, _, vt, _, _ = moments()
        delta = (capacities.float() - mt.sum(-1)) / vt.sum(-1).clamp_min(1e-4)
        row_bias = row_bias + delta.clamp(-1, 1)[:, :, None, None]
        _, mt, mr, vt, vr, cov = moments()
        a, d, b = vt.sum(1) + 1e-4, vr.sum(1) + 1e-4, cov.sum(1)
        et, er = column_target - mt.sum(1), red_target - mr.sum(1)
        determinant = (a * d - b.square()).clamp_min(1e-8)
        dt, dr = (d * et - b * er) / determinant, (a * er - b * et) / determinant
        column_bias = column_bias + dt.clamp(-1, 1)[:, None, :, None]
        red_bias = red_bias + dr.clamp(-1, 1)[:, None, :, None]
    return moments()[0]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
@pytest.mark.parametrize("scale", [0.0, 0.2, 1.0, 3.0])
def test_projection_preserves_probability_loss_and_residual_gradient(device, scale):
    torch.manual_seed(53)
    inventory = torch.full((3, 37), 4, device=device)
    inventory[:, (4, 13, 22)] = 3
    inventory[:, 34:] = 1
    inventory[1, :4] = 0
    inventory[2] = 0
    capacities = torch.tensor(
        [[13, 13, 13, 97], [10, 10, 10, 90], [0, 0, 0, 0]], device=device
    )
    residual = (torch.randn(3, 4, 34, 10, device=device) * scale).requires_grad_()
    old_residual = residual.detach().clone().requires_grad_()
    weights = torch.rand_like(residual)
    actual = project_joint_counts(residual, inventory, capacities).probability
    expected = reference_projection(old_residual, inventory, capacities)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_loss = (actual * weights).square().sum()
    expected_loss = (expected * weights).square().sum()
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(residual.grad, old_residual.grad, rtol=0, atol=0)
