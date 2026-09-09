from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .prediction_values import DORA_TAIL_START, SCORE_VALUES


# A term represents one independently supervised prediction, not a manually
# chosen product priority. Their relative influence is learned during training
# by LearnedUncertaintyBalancer and recorded with every checkpoint.
LOSS_TERMS = (
    "policy",
    "shanten",
    "furiten_no_yaku",
    "deal_in_tile",
    "concealed_count",
    "concealed_red_count",
    "wall_count",
    "wall_red_count",
    "dora_distribution",
    "dora_point",
    "score_distribution",
    "score_point",
    "outcome",
    "kyoku_delta",
    "placement",
    "match_score",
)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        return values.new_zeros(())
    return selected.mean()


def score_class_indices(values: Tensor) -> Tensor:
    vocabulary = torch.as_tensor(SCORE_VALUES, device=values.device, dtype=values.dtype)
    indices = torch.searchsorted(vocabulary, values)
    bounded = indices.clamp_max(len(SCORE_VALUES) - 1)
    if ((indices >= len(SCORE_VALUES)) | (vocabulary[bounded] != values)).any():
        invalid = values[(indices >= len(SCORE_VALUES)) | (vocabulary[bounded] != values)]
        raise ValueError(f"score labels contain unsupported values: {invalid.unique().tolist()}")
    return bounded


def multitask_losses(
    outputs: Mapping[str, Tensor], batch: Mapping[str, Tensor]
) -> tuple[dict[str, Tensor], dict[str, bool]]:
    """Return raw proper losses and whether each has real supervision.

    Empty conditional subsets are retained as differentiable zeroes for a
    uniform return schema, but are excluded from uncertainty updates. This is
    essential for dora/score heads, which have no label in a no-win batch.
    """

    losses: dict[str, Tensor] = {}
    active: dict[str, bool] = {}
    policy_valid = batch["policy"] >= 0
    policy_logits = outputs["policy"].masked_fill(~batch["action_mask"], -torch.inf)
    if policy_valid.any():
        losses["policy"] = F.cross_entropy(
            policy_logits[policy_valid], batch["policy"][policy_valid].long()
        )
        active["policy"] = True
    else:
        losses["policy"] = outputs["policy"].sum() * 0
        active["policy"] = False

    losses["shanten"] = F.cross_entropy(
        outputs["shanten"].reshape(-1, 7), batch["shanten"].reshape(-1).long()
    )
    active["shanten"] = True
    tenpai = batch["shanten"] == 0
    furiten_raw = F.binary_cross_entropy_with_logits(
        outputs["furiten_no_yaku"],
        batch["furiten_no_yaku"].float(),
        reduction="none",
    )
    losses["furiten_no_yaku"] = _masked_mean(furiten_raw, tenpai)
    active["furiten_no_yaku"] = bool(tenpai.any())
    losses["deal_in_tile"] = F.binary_cross_entropy_with_logits(
        outputs["deal_in_tile"], batch["deal_in_tile"].float()
    )
    active["deal_in_tile"] = True
    losses["concealed_count"] = F.cross_entropy(
        outputs["concealed_count"].reshape(-1, 5),
        batch["concealed_count"].reshape(-1).long(),
    )
    active["concealed_count"] = True
    losses["concealed_red_count"] = F.cross_entropy(
        outputs["concealed_red_count"].reshape(-1, 2),
        batch["concealed_red_count"].reshape(-1).long(),
    )
    active["concealed_red_count"] = True
    losses["wall_count"] = F.cross_entropy(
        outputs["wall_count"].reshape(-1, 5), batch["wall_count"].reshape(-1).long()
    )
    active["wall_count"] = True
    losses["wall_red_count"] = F.cross_entropy(
        outputs["wall_red_count"].reshape(-1, 2),
        batch["wall_red_count"].reshape(-1).long(),
    )
    active["wall_red_count"] = True

    winner_mask = batch["winner_mask"].bool()
    if winner_mask.any():
        dora_labels = batch["dora"].long()[winner_mask]
        losses["dora_distribution"] = F.cross_entropy(
            outputs["dora_distribution"][winner_mask], dora_labels.clamp_max(DORA_TAIL_START)
        )
        losses["dora_point"] = F.mse_loss(
            F.softplus(outputs["dora_point"])[winner_mask], dora_labels.float()
        )
        score_labels = batch["score"].long()[winner_mask]
        losses["score_distribution"] = F.cross_entropy(
            outputs["score_distribution"][winner_mask], score_class_indices(score_labels)
        )
        losses["score_point"] = F.mse_loss(
            F.softplus(outputs["score_point"])[winner_mask], score_labels.float() / 1000.0
        )
        for name in (
            "dora_distribution",
            "dora_point",
            "score_distribution",
            "score_point",
        ):
            active[name] = True
    else:
        zero = outputs["dora_distribution"].sum() * 0
        for name in (
            "dora_distribution",
            "dora_point",
            "score_distribution",
            "score_point",
        ):
            losses[name] = zero
            active[name] = False

    losses["outcome"] = F.cross_entropy(outputs["outcome"], batch["outcome"].long())
    active["outcome"] = True
    losses["kyoku_delta"] = F.smooth_l1_loss(
        outputs["kyoku_delta"], batch["kyoku_delta"].float() / 10_000.0
    )
    active["kyoku_delta"] = True
    losses["placement"] = F.cross_entropy(outputs["placement"], batch["placement"].long())
    active["placement"] = True
    losses["match_score"] = F.smooth_l1_loss(
        outputs["match_score"], batch["match_score"].float() / 10_000.0
    )
    active["match_score"] = True
    if tuple(losses) != LOSS_TERMS or tuple(active) != LOSS_TERMS:
        raise RuntimeError("multi-task loss terms are incomplete")
    return losses, active


class LearnedUncertaintyBalancer(nn.Module):
    """Learn each supervised objective's relative scale during training.

    The objective for an active term is ``exp(-s) * loss + s``. Unlike the
    previous fixed hand-tuned dictionary, the resulting scale depends on the
    data, loss evolution, and model capacity. The log variances are training
    state, never inference-model parameters.
    """

    def __init__(self, names: tuple[str, ...] = LOSS_TERMS) -> None:
        super().__init__()
        if not names:
            raise ValueError("at least one loss term is required")
        if len(set(names)) != len(names):
            raise ValueError("loss term names must be unique")
        self.names = names
        self.log_variance = nn.Parameter(torch.zeros(len(names)))

    def weights(self) -> dict[str, Tensor]:
        return {
            name: torch.exp(-self.log_variance[index])
            for index, name in enumerate(self.names)
        }

    def forward(
        self, losses: Mapping[str, Tensor], active: Mapping[str, bool]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if tuple(losses) != self.names or tuple(active) != self.names:
            raise ValueError("losses must match the balancer's declared terms")
        terms: list[Tensor] = []
        weights: dict[str, Tensor] = {}
        for index, name in enumerate(self.names):
            weight = torch.exp(-self.log_variance[index])
            weights[name] = weight
            if active[name]:
                terms.append(weight * losses[name] + self.log_variance[index])
        if not terms:
            return next(iter(losses.values())).sum() * 0, weights
        return torch.stack(terms).sum(), weights


def multitask_loss(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    balancer: LearnedUncertaintyBalancer | None = None,
) -> tuple[Tensor, dict[str, Tensor], dict[str, bool], dict[str, Tensor]]:
    """Compute raw losses, then combine them with the supplied balancer."""

    losses, active = multitask_losses(outputs, batch)
    if balancer is None:
        weights = {name: losses[name].new_ones(()) for name in LOSS_TERMS}
        total = torch.stack([losses[name] for name in LOSS_TERMS if active[name]]).sum()
    else:
        total, weights = balancer(losses, active)
    return total, losses, active, weights
