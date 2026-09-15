from __future__ import annotations

import math

import torch
from torch import Tensor

HIDDEN_SOURCES = 4
TILE_TYPES = 34
RED_TYPES = 3
PHYSICAL_TILE_TYPES = TILE_TYPES + RED_TYPES
FIVE_TILE_INDICES = (4, 13, 22)


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
            result[selected, count] = (
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
