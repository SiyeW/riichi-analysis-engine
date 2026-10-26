"""Value-aware scores of existing distributions; no prediction parameters."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from functools import lru_cache
from math import isfinite

import torch
from torch import Tensor

from .prediction_values import PLACEMENT_PERMUTATIONS


@lru_cache(maxsize=32)
def _support(
    classes: int, values: tuple[float, ...] | None, unit: float, device: torch.device
):
    if values is None:
        support = torch.arange(classes, dtype=torch.float32, device=device)
    else:
        if (
            len(values) != classes
            or not all(isfinite(v) for v in values)
            or any(a >= b for a, b in itertools.pairwise(values))
        ):
            raise ValueError(
                "CDF support must be strictly increasing and match classes"
            )
        support = torch.tensor(values, dtype=torch.float32, device=device)
    return support[:-1], (support[1:] - support[:-1]) / unit


@lru_cache(maxsize=16)
def _placement_support(device: torch.device):
    permutations = torch.tensor(PLACEMENT_PERMUTATIONS, device=device, dtype=torch.long)
    return permutations, torch.nn.functional.one_hot(permutations, 4).float()


def cdf_distance(
    probability: Tensor,
    labels: Tensor | None = None,
    *,
    target_probability: Tensor | None = None,
    values: Sequence[float] | None = None,
    unit: float = 1.0,
) -> Tensor:
    """Discrete CRPS, unreduced over observations, at real support gaps.

    Open terminal buckets are scored only at their known finite boundaries.
    Soft targets retain the original analytic-anchor supervision contract.
    Inputs are normalized probabilities, never arbitrary class-ID coordinates.
    """
    if (labels is None) == (target_probability is None):
        raise ValueError("CDF distance requires exactly one hard or soft target")
    classes = probability.shape[-1]
    if classes < 2 or not isfinite(unit) or unit <= 0:
        raise ValueError("CDF score requires at least two classes and a positive unit")
    probability = probability.float()
    # Immutable device constants are built once, not copied CPU→GPU each update.
    thresholds, gaps = _support(
        classes, None if values is None else tuple(values), unit, probability.device
    )
    if target_probability is not None:
        if target_probability.shape != probability.shape:
            raise ValueError("soft CDF target shape differs from prediction")
        target = target_probability.float().cumsum(-1)[..., :-1]
    else:
        if labels.shape != probability.shape[:-1]:
            raise ValueError("CDF label shape differs from prediction")
        target = (labels.unsqueeze(-1) <= thresholds).float()
    forecast = probability.cumsum(-1)[..., :-1]
    return ((forecast - target).square() * gaps).sum(-1)


def placement_distance(probability: Tensor, labels: Tensor) -> Tensor:
    """Mean rank RPS over players, from the same coherent24-permutation law."""
    if probability.shape[-1] != len(PLACEMENT_PERMUTATIONS):
        raise ValueError("placement distance requires24 joint permutations")
    permutations, rank_indicator = _placement_support(probability.device)
    # Elementwise FP32 reductions remain FP32 even in an outer AMP context.
    marginal = (probability.float()[..., :, None, None] * rank_indicator).sum(-3)
    return cdf_distance(marginal, permutations[labels.long()]).mean(-1)


def hidden_count_distance(
    probability: Tensor,
    baseline: Tensor,
    total_labels: Tensor,
    total_inventory: Tensor,
    anchor: Tensor,
) -> Tensor:
    """0..4 total-count marginal RPS with the SAME per-frame active-cell mean."""
    if probability.shape[-3:] != (4, 34, 10) or baseline.shape != probability.shape:
        raise ValueError("ordered hidden counts require the projected joint law")
    total = probability.float().reshape(*probability.shape[:-1], 5, 2).sum(-1)
    baseline_total = baseline.float().reshape(*baseline.shape[:-1], 5, 2).sum(-1)
    observed = cdf_distance(total, total_labels)
    analytic = cdf_distance(total, target_probability=baseline_total)
    distance = torch.where(anchor.bool()[:, None, None], analytic, observed)
    active = (total_inventory[:, None, :] > 0).expand_as(distance)
    per_frame = (distance * active).sum((1, 2)) / active.sum((1, 2)).clamp_min(1)
    return per_frame.mean()
