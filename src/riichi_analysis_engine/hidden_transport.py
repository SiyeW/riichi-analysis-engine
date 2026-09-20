from __future__ import annotations

import math

import torch
from torch import Tensor

HIDDEN_SOURCES = 4
TILE_TYPES = 34
RED_TYPES = 3
PHYSICAL_TILE_TYPES = TILE_TYPES + RED_TYPES
FIVE_TILE_INDICES = (4, 13, 22)
COUNT_CLASSES = 5


def physical_hidden_counts(
    concealed: Tensor,
    wall: Tensor,
    concealed_red: Tensor,
    wall_red: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Split total fives into normal/red physical categories.

    The returned column masses and row capacities are public constraints even
    though the allocation table itself is the supervised target.
    """

    if concealed.ndim != 3 or concealed.shape[1:] != (3, TILE_TYPES):
        raise ValueError(f"wrong concealed-count shape: {tuple(concealed.shape)}")
    batch = len(concealed)
    if wall.shape != (batch, TILE_TYPES):
        raise ValueError(f"wrong wall-count shape: {tuple(wall.shape)}")
    if concealed_red.shape != (batch, 3, RED_TYPES):
        raise ValueError(f"wrong concealed-red shape: {tuple(concealed_red.shape)}")
    if wall_red.shape != (batch, RED_TYPES):
        raise ValueError(f"wrong wall-red shape: {tuple(wall_red.shape)}")

    total = torch.cat((concealed.long(), wall.long().unsqueeze(1)), dim=1)
    red = torch.cat((concealed_red.long(), wall_red.long().unsqueeze(1)), dim=1)
    normal = total.clone()
    for suit, tile in enumerate(FIVE_TILE_INDICES):
        normal[:, :, tile] -= red[:, :, suit]
    if (normal < 0).any():
        raise ValueError("red-five count exceeds the corresponding five count")
    physical = torch.cat((normal, red), dim=-1)
    inventory = physical.sum(dim=1)
    capacities = total.sum(dim=-1)
    if ((inventory < 0) | (inventory > 4)).any():
        raise ValueError("physical hidden inventory lies outside 0..4")
    if (inventory[:, TILE_TYPES:] > 1).any():
        raise ValueError("red-five inventory lies outside 0..1")
    if not torch.equal(inventory.sum(dim=-1), capacities.sum(dim=-1)):
        raise ValueError("physical inventories and source capacities do not balance")
    return physical, inventory, capacities


def physical_affinities(tile_affinity: Tensor, red_affinity: Tensor) -> Tensor:
    if tile_affinity.ndim != 3 or tile_affinity.shape[1:] != (
        HIDDEN_SOURCES,
        TILE_TYPES,
    ):
        raise ValueError(f"wrong tile-affinity shape: {tuple(tile_affinity.shape)}")
    if red_affinity.shape != (len(tile_affinity), RED_TYPES, HIDDEN_SOURCES):
        raise ValueError(f"wrong red-affinity shape: {tuple(red_affinity.shape)}")
    return torch.cat((tile_affinity, red_affinity.transpose(1, 2)), dim=-1)


def balanced_source_probabilities(
    affinity: Tensor,
    inventory: Tensor,
    capacities: Tensor,
    *,
    iterations: int = 24,
) -> Tensor:
    """Project source affinities to a transport plan with exact mean margins.

    Every physical tile copy is assigned to exactly one source. The resulting
    multinomial model therefore conserves each tile inventory in every draw,
    while Sinkhorn scaling makes every source's total correct in expectation.
    This is the fast fallback for the exact integer-table partition function,
    whose measured cost is too high for the target GPU.
    """

    if iterations <= 0:
        raise ValueError("transport iterations must be positive")
    if affinity.ndim != 3 or affinity.shape[1:] != (
        HIDDEN_SOURCES,
        PHYSICAL_TILE_TYPES,
    ):
        raise ValueError(f"wrong physical-affinity shape: {tuple(affinity.shape)}")
    if inventory.shape != (len(affinity), PHYSICAL_TILE_TYPES):
        raise ValueError(f"wrong physical-inventory shape: {tuple(inventory.shape)}")
    if capacities.shape != (len(affinity), HIDDEN_SOURCES):
        raise ValueError(f"wrong source-capacity shape: {tuple(capacities.shape)}")
    if not torch.equal(inventory.sum(-1), capacities.sum(-1)):
        raise ValueError("physical inventories and source capacities do not balance")

    logits = affinity.float()
    row_mass = capacities.float()
    column_mass = inventory.float()
    active_rows = row_mass > 0
    active_columns = column_mass > 0
    floor = torch.finfo(logits.dtype).min / 4
    log_rows = torch.where(active_rows, row_mass.clamp_min(1).log(), floor)
    log_columns = torch.where(active_columns, column_mass.clamp_min(1).log(), floor)
    log_v = torch.where(active_columns, torch.zeros_like(column_mass), floor)
    log_u = torch.where(active_rows, torch.zeros_like(row_mass), floor)
    for _ in range(iterations):
        row_norm = torch.logsumexp(logits + log_v.unsqueeze(1), dim=-1)
        log_u = torch.where(active_rows, log_rows - row_norm, floor)
        column_norm = torch.logsumexp(logits + log_u.unsqueeze(-1), dim=1)
        log_v = torch.where(active_columns, log_columns - column_norm, floor)
    transport = torch.exp(logits + log_u.unsqueeze(-1) + log_v.unsqueeze(1))
    probabilities = torch.where(
        active_columns.unsqueeze(1),
        transport / column_mass.clamp_min(1).unsqueeze(1),
        torch.zeros_like(transport),
    )
    return probabilities


def hidden_transport_nll(probabilities: Tensor, physical_counts: Tensor) -> Tensor:
    """Mean per-physical-family multinomial NLL for the true allocation."""

    if probabilities.shape != physical_counts.shape:
        raise ValueError("transport probabilities and targets must have the same shape")
    counts = physical_counts.float()
    inventory = counts.sum(dim=1)
    log_probability = probabilities.clamp_min(
        torch.finfo(probabilities.dtype).tiny
    ).log()
    log_likelihood = (counts * log_probability).sum(dim=1)
    log_likelihood = log_likelihood + torch.lgamma(inventory + 1)
    log_likelihood = log_likelihood - torch.lgamma(counts + 1).sum(dim=1)
    active = inventory > 0
    per_sample = -(log_likelihood * active).sum(dim=-1) / active.sum(dim=-1).clamp_min(
        1
    )
    return per_sample.mean()


def _binomial_marginals(copy_count: Tensor, probability: Tensor) -> Tensor:
    result = probability.new_zeros((*probability.shape, 5))
    for copies in range(5):
        selected = copy_count == copies
        if not selected.any():
            continue
        p = probability[selected]
        for count in range(copies + 1):
            # Select the last probability axis first.  Indexing ``result``
            # with the three-dimensional mask and ``count`` together happens
            # to work for a singleton batch but treats the source axis as an
            # indexed dimension once the batch has multiple samples.
            result[..., count][selected] = (
                math.comb(copies, count) * p.pow(count) * (1.0 - p).pow(copies - count)
            )
    return result


def count_marginals(
    source_probabilities: Tensor,
    physical_inventory: Tensor,
) -> tuple[Tensor, Tensor]:
    """Derive protocol total-tile and optional red-five distributions."""

    if source_probabilities.ndim != 3 or source_probabilities.shape[1:] != (
        HIDDEN_SOURCES,
        PHYSICAL_TILE_TYPES,
    ):
        raise ValueError("wrong source-probability shape")
    if physical_inventory.shape != (len(source_probabilities), PHYSICAL_TILE_TYPES):
        raise ValueError("wrong physical-inventory shape")
    batch = len(source_probabilities)
    normal = _binomial_marginals(
        physical_inventory[:, None, :TILE_TYPES].expand(-1, HIDDEN_SOURCES, -1),
        source_probabilities[:, :, :TILE_TYPES],
    )
    red_probability = source_probabilities[:, :, TILE_TYPES:]
    red_present = physical_inventory[:, None, TILE_TYPES:].float()
    red = torch.stack(
        (1.0 - red_probability * red_present, red_probability * red_present), dim=-1
    )
    total = normal.clone()
    for suit, tile in enumerate(FIVE_TILE_INDICES):
        base = normal[:, :, tile]
        shifted = torch.cat((torch.zeros_like(base[..., :1]), base[..., :-1]), dim=-1)
        combined = base * red[:, :, suit, :1] + shifted * red[:, :, suit, 1:]
        total[:, :, tile] = combined
    return total.view(batch, HIDDEN_SOURCES, TILE_TYPES, 5), red


def theoretical_count_baseline(
    physical_inventory: Tensor,
    source_capacities: Tensor,
) -> Tensor:
    """Return exact no-behaviour marginal count distributions.

    Conditional on the publicly known physical inventory and source sizes, an
    individual source is a draw without replacement from the remaining hidden
    tiles.  Its count for one physical tile family is therefore
    hypergeometric.  These marginals are the analytic baseline that the v13
    residual head refines; the network must not relearn them from sampled
    opening hands.
    """

    if physical_inventory.ndim != 2 or physical_inventory.shape[1] != PHYSICAL_TILE_TYPES:
        raise ValueError("wrong physical-inventory shape")
    if source_capacities.shape != (len(physical_inventory), HIDDEN_SOURCES):
        raise ValueError("wrong source-capacity shape")
    if not torch.equal(physical_inventory.sum(-1), source_capacities.sum(-1)):
        raise ValueError("physical inventories and source capacities do not balance")

    inventory = physical_inventory.float()[:, None, :, None]
    capacities = source_capacities.float()[:, :, None, None]
    population = physical_inventory.sum(-1).float()[:, None, None, None]
    counts = torch.arange(
        COUNT_CLASSES, device=physical_inventory.device, dtype=torch.float32
    ).view(1, 1, 1, -1)

    def log_choose(total: Tensor, selected: Tensor) -> Tensor:
        return (
            torch.lgamma(total + 1.0)
            - torch.lgamma(selected + 1.0)
            - torch.lgamma(total - selected + 1.0)
        )

    valid = (
        (counts <= inventory)
        & (counts <= capacities)
        & (capacities - counts <= population - inventory)
    )
    safe_population = population.clamp_min(1.0)
    log_probability = (
        log_choose(inventory, counts)
        + log_choose(population - inventory, capacities - counts)
        - log_choose(safe_population, capacities)
    )
    floor = torch.finfo(log_probability.dtype).min
    log_probability = torch.where(valid, log_probability, floor)
    baseline = torch.softmax(log_probability, dim=-1)

    empty = population == 0
    if empty.any():
        deterministic_zero = torch.zeros_like(baseline)
        deterministic_zero[..., 0] = 1.0
        baseline = torch.where(empty, deterministic_zero, baseline)
    return baseline


def projected_count_distributions(
    residual_logits: Tensor,
    physical_inventory: Tensor,
    source_capacities: Tensor,
    *,
    iterations: int = 32,
) -> tuple[Tensor, Tensor]:
    """Apply learned residuals and restore public expectation constraints.

    Unlike the historical independent-copy/binomial decoder, this operates on
    complete 0..4 count distributions.  Alternating exponential tilts enforce
    every source capacity and physical-tile inventory in expectation while
    preserving arbitrary shapes such as high mass at 0 and 2 but little at 1.
    """

    expected_shape = (
        len(physical_inventory),
        HIDDEN_SOURCES,
        PHYSICAL_TILE_TYPES,
        COUNT_CLASSES,
    )
    if residual_logits.shape != expected_shape:
        raise ValueError(
            f"wrong hidden-count residual shape: {tuple(residual_logits.shape)}"
        )
    if iterations <= 0:
        raise ValueError("count projection iterations must be positive")

    baseline = theoretical_count_baseline(physical_inventory, source_capacities)
    floor = torch.finfo(torch.float32).min
    logits = torch.where(
        baseline > 0,
        baseline.clamp_min(torch.finfo(torch.float32).tiny).log()
        + residual_logits.float(),
        torch.full_like(residual_logits, floor, dtype=torch.float32),
    )
    count_values = torch.arange(
        COUNT_CLASSES, device=logits.device, dtype=logits.dtype
    ).view(1, 1, 1, -1)
    row_bias = logits.new_zeros((len(logits), HIDDEN_SOURCES))
    column_bias = logits.new_zeros((len(logits), PHYSICAL_TILE_TYPES))
    row_target = source_capacities.float()
    column_target = physical_inventory.float()

    def moments() -> tuple[Tensor, Tensor, Tensor]:
        adjusted = logits + count_values * (
            row_bias[:, :, None, None] + column_bias[:, None, :, None]
        )
        probability = torch.softmax(adjusted, dim=-1)
        mean = (probability * count_values).sum(-1)
        variance = (
            probability * (count_values - mean.unsqueeze(-1)).square()
        ).sum(-1)
        return probability, mean, variance

    # Newton updates on one family of margins at a time are the moment-space
    # counterpart of iterative proportional fitting.  Clamp only the dual
    # update, never the learned distribution, to keep extreme early logits
    # numerically recoverable.
    for _ in range(iterations):
        _probability, mean, variance = moments()
        row_delta = (row_target - mean.sum(-1)) / variance.sum(-1).clamp_min(1e-4)
        row_bias = row_bias + row_delta.clamp(-1.0, 1.0)

        _probability, mean, variance = moments()
        column_delta = (column_target - mean.sum(1)) / variance.sum(1).clamp_min(1e-4)
        column_bias = column_bias + column_delta.clamp(-1.0, 1.0)

    probability, _mean, _variance = moments()
    return probability, baseline


def hidden_count_distribution_loss(
    probability: Tensor,
    baseline: Tensor,
    physical_counts: Tensor,
    physical_inventory: Tensor,
    baseline_anchor: Tensor,
) -> Tensor:
    """Train anchors from analytic soft labels and all later frames from truth."""

    if probability.shape != baseline.shape:
        raise ValueError("hidden-count prediction and baseline shapes differ")
    if physical_counts.shape != probability.shape[:-1]:
        raise ValueError("hidden-count targets have the wrong shape")
    if baseline_anchor.shape != (len(probability),):
        raise ValueError("hidden-count anchor mask has the wrong shape")

    log_probability = probability.float().clamp_min(
        torch.finfo(torch.float32).tiny
    ).log()
    sampled = -log_probability.gather(
        -1, physical_counts.long().unsqueeze(-1)
    ).squeeze(-1)
    baseline_float = baseline.float()
    analytic = (
        baseline_float
        * (
            baseline_float.clamp_min(torch.finfo(torch.float32).tiny).log()
            - log_probability
        )
    ).sum(-1)
    per_family = torch.where(
        baseline_anchor.bool()[:, None, None], analytic, sampled
    )
    active = (physical_inventory[:, None, :] > 0).expand(
        -1, HIDDEN_SOURCES, -1
    )
    per_sample = (per_family * active).sum(dim=(1, 2)) / active.sum(
        dim=(1, 2)
    ).clamp_min(1)
    return per_sample.mean()


def physical_count_marginals(
    physical_distributions: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compose protocol base-tile and red-five marginals from 37 entities."""

    if physical_distributions.ndim != 4 or physical_distributions.shape[1:] != (
        HIDDEN_SOURCES,
        PHYSICAL_TILE_TYPES,
        COUNT_CLASSES,
    ):
        raise ValueError("wrong physical count-distribution shape")
    normal = physical_distributions[:, :, :TILE_TYPES]
    red = physical_distributions[:, :, TILE_TYPES:, :2]
    total = normal.clone()
    for suit, tile in enumerate(FIVE_TILE_INDICES):
        base = normal[:, :, tile]
        shifted = torch.cat((torch.zeros_like(base[..., :1]), base[..., :-1]), dim=-1)
        total[:, :, tile] = base * red[:, :, suit, :1] + shifted * red[:, :, suit, 1:]
    return total, red
