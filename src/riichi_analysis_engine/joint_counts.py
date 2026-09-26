"""v14 count distributions: total-five and red-five are one random variable.

Each of 34 tile families has ten states (total count 0..4, red count 0..1).
Impossible states, including every red state of a non-five, have exactly zero
mass. Projection enforces public margins in expectation, not a joint allocation
of all hands. No hidden information is used other than public inventories/sizes.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from .hidden_transport import FIVE_TILE_INDICES, hidden_count_distribution_loss


def _values(reference: Tensor) -> tuple[Tensor, Tensor]:
    total = torch.arange(5, device=reference.device, dtype=reference.dtype)
    total = total.repeat_interleave(2).view(1, 1, 1, 10)
    red = torch.arange(2, device=reference.device, dtype=reference.dtype)
    return total, red.repeat(5).view(1, 1, 1, 10)


def family_inventory(physical: Tensor) -> tuple[Tensor, Tensor]:
    total = physical[:, :34].clone()
    red = torch.zeros_like(total)
    red[:, FIVE_TILE_INDICES] = physical[:, 34:]
    return total + red, red


def joint_count_baseline(inventory: Tensor, capacities: Tensor) -> Tensor:
    if inventory.ndim != 2 or inventory.shape[1] != 37:
        raise ValueError("joint counts require 37 physical inventories")
    if capacities.shape != (len(inventory), 4):
        raise ValueError("joint counts require four source capacities")
    if (inventory < 0).any() or (capacities < 0).any():
        raise ValueError("negative hidden-count constraints")
    if (inventory[:, :34] > 4).any() or (inventory[:, 34:] > 1).any():
        raise ValueError("invalid hidden-count inventory")
    if not torch.equal(inventory.sum(-1), capacities.sum(-1)):
        raise ValueError("hidden inventories and source capacities do not balance")
    # Small combinatorial calculation in float64 avoids cancellation in lgamma.
    total_inventory, red_inventory = family_inventory(inventory.double())
    if (total_inventory > 4).any():
        raise ValueError("normal plus red inventory exceeds four")
    normal = (total_inventory - red_inventory)[:, None, :, None]
    red_inventory = red_inventory[:, None, :, None]
    size = capacities.double()[:, :, None, None]
    population = inventory.sum(-1).double()[:, None, None, None]
    total, red = _values(normal)
    normal_count = total - red
    other = population - normal - red_inventory
    valid = (
        (normal_count >= 0)
        & (normal_count <= normal)
        & (red <= red_inventory)
        & (total <= size)
        & (size - total <= other)
    )

    def choose(n: Tensor, k: Tensor) -> Tensor:
        # Never evaluate lgamma on negative integers even on masked branches.
        return (
            torch.lgamma(n + 1)
            - torch.lgamma(k.clamp_min(0) + 1)
            - torch.lgamma((n - k).clamp_min(0) + 1)
        )

    logits = (
        choose(normal, normal_count)
        + choose(red_inventory, red)
        + choose(other, size - total)
        - choose(population, size)
    )
    return logits.masked_fill(~valid, -torch.inf).softmax(-1).float()


def joint_count_marginals(probability: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if probability.ndim != 4 or probability.shape[1:] != (4, 34, 10):
        raise ValueError("wrong joint count-distribution shape")
    joint = probability.reshape(*probability.shape[:-1], 5, 2)
    total = joint.sum(-1)
    red = joint[:, :, FIVE_TILE_INDICES].sum(-2)
    # Normal count = total - red, derived from the SAME joint distribution.
    normal = joint[..., 0].clone()
    normal[..., :-1] = normal[..., :-1] + joint[..., 1:, 1]
    physical_red = torch.nn.functional.pad(red, (0, 3))
    return total, red, torch.cat((normal, physical_red), dim=2)


@dataclass(frozen=True)
class JointCountPrediction:
    probability: Tensor
    baseline: Tensor

    def marginals(self) -> tuple[Tensor, Tensor, Tensor]:
        return joint_count_marginals(self.probability)

    def loss(
        self, physical_counts: Tensor, inventory: Tensor, anchor: Tensor
    ) -> Tensor:
        total, red = family_inventory(physical_counts.flatten(0, 1))
        indices = (2 * total + red).reshape(len(physical_counts), 4, 34)
        total_inventory, _ = family_inventory(inventory)
        return hidden_count_distribution_loss(
            self.probability, self.baseline, indices, total_inventory, anchor
        )


def project_joint_counts(
    residual: Tensor, inventory: Tensor, capacities: Tensor, *, iterations: int = 32
) -> JointCountPrediction:
    if residual.shape != (len(inventory), 4, 34, 10):
        raise ValueError("wrong joint hidden-count residual shape")
    if iterations <= 0:
        raise ValueError("joint projection iterations must be positive")
    baseline, logits, column_target, red_target, source_target = joint_projection_inputs(
        residual, inventory, capacities
    )
    return JointCountPrediction(
        project_joint_probability(logits, column_target, red_target, source_target, iterations),
        baseline,
    )


def joint_projection_inputs(
    residual: Tensor, inventory: Tensor, capacities: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Validate constraints outside the device-only iterative calculation."""
    if residual.shape != (len(inventory), 4, 34, 10):
        raise ValueError("wrong joint hidden-count residual shape")
    baseline = joint_count_baseline(inventory, capacities)
    logits = (
        baseline.clamp_min(torch.finfo(torch.float32).tiny).log() + residual.float()
    ).masked_fill(baseline == 0, -torch.inf)
    column_target, red_target = family_inventory(inventory.float())
    return baseline, logits, column_target, red_target, capacities.float()


def project_joint_probability(
    logits: Tensor, column_target: Tensor, red_target: Tensor, source_target: Tensor,
    iterations: int = 32,
) -> Tensor:
    """One mathematical implementation shared by eager and captured execution."""
    total, red = _values(logits)
    row_bias = logits.new_zeros((len(logits), 4, 1, 1))
    column_bias = logits.new_zeros((len(logits), 1, 34, 1))
    red_bias = torch.zeros_like(column_bias)

    def probability() -> Tensor:
        return (logits + total * (row_bias + column_bias) + red * red_bias).softmax(-1)

    for _ in range(iterations):
        # The source update only needs total-count moments. Computing red
        # variance/covariance here creates unused GPU work on every iteration.
        p = probability()
        mt = (p * total).sum(-1)
        vt = (p * (total - mt.unsqueeze(-1)).square()).sum(-1)
        delta = (source_target - mt.sum(-1)) / vt.sum(-1).clamp_min(1e-4)
        row_bias = row_bias + delta.clamp(-1, 1)[:, :, None, None]
        p = probability()
        mt, mr = (p * total).sum(-1), (p * red).sum(-1)
        dt, dr = total - mt.unsqueeze(-1), red - mr.unsqueeze(-1)
        vt, vr = (p * dt.square()).sum(-1), (p * dr.square()).sum(-1)
        cov = (p * dt * dr).sum(-1)
        # Joint 2x2 Newton update retains total/red covariance. A small ridge
        # handles deterministic or absent-red columns without a singular solve.
        a, d, b = vt.sum(1) + 1e-4, vr.sum(1) + 1e-4, cov.sum(1)
        et, er = column_target - mt.sum(1), red_target - mr.sum(1)
        determinant = (a * d - b.square()).clamp_min(1e-8)
        dt, dr = (d * et - b * er) / determinant, (a * er - b * et) / determinant
        column_bias = column_bias + dt.clamp(-1, 1)[:, None, :, None]
        red_bias = red_bias + dr.clamp(-1, 1)[:, None, :, None]
    return probability()
