from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


DEFAULT_WEIGHTS = {
    "policy": 1.0,
    "shanten": 1.0,
    "furiten_no_yaku": 0.2,
    "deal_in_tile": 0.5,
    "concealed_count": 1.0,
    "wall_count": 1.0,
    "dora": 0.2,
    "score": 0.2,
    "outcome": 0.7,
    "deal_in_player": 0.3,
    "target": 0.2,
    "kyoku_delta": 0.2,
    "placement": 0.5,
    "match_score": 0.2,
}


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def multitask_loss(
    outputs: dict[str, Tensor],
    batch: dict[str, Tensor],
    weights: dict[str, float] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    weights = DEFAULT_WEIGHTS if weights is None else weights
    losses: dict[str, Tensor] = {}

    policy_valid = batch["policy"] >= 0
    policy_logits = outputs["policy"].masked_fill(~batch["action_mask"], -torch.inf)
    if policy_valid.any():
        losses["policy"] = F.cross_entropy(
            policy_logits[policy_valid], batch["policy"][policy_valid].long()
        )
    else:
        losses["policy"] = outputs["policy"].sum() * 0

    losses["shanten"] = F.cross_entropy(
        outputs["shanten"].reshape(-1, 7), batch["shanten"].reshape(-1).long()
    )
    furiten_raw = F.binary_cross_entropy_with_logits(
        outputs["furiten_no_yaku"],
        batch["furiten_no_yaku"].float(),
        reduction="none",
    )
    losses["furiten_no_yaku"] = _masked_mean(
        furiten_raw, batch["shanten"] == 0
    )
    losses["deal_in_tile"] = F.binary_cross_entropy_with_logits(
        outputs["deal_in_tile"], batch["deal_in_tile"].float()
    )
    losses["concealed_count"] = F.cross_entropy(
        outputs["concealed_count"].reshape(-1, 5),
        batch["concealed_count"].reshape(-1).long(),
    )
    losses["wall_count"] = F.cross_entropy(
        outputs["wall_count"].reshape(-1, 5), batch["wall_count"].reshape(-1).long()
    )

    winner_mask = batch["winner_mask"].bool()
    dora_error = F.smooth_l1_loss(
        F.softplus(outputs["dora"]), batch["dora"].float(), reduction="none"
    )
    losses["dora"] = _masked_mean(dora_error, winner_mask)
    score_error = F.smooth_l1_loss(
        F.softplus(outputs["score"]), batch["score"].float() / 1000.0, reduction="none"
    )
    losses["score"] = _masked_mean(score_error, winner_mask)
    outcome_label = sum(
        batch["win"][:, player].long() << player for player in range(4)
    )
    losses["outcome"] = F.cross_entropy(outputs["outcome"], outcome_label)
    losses["deal_in_player"] = F.binary_cross_entropy_with_logits(
        outputs["deal_in_player"], batch["deal_in_player"].float()
    )
    target_error = F.cross_entropy(
        outputs["target"].reshape(-1, 4),
        batch["target"].clamp_min(0).reshape(-1).long(),
        reduction="none",
    ).reshape_as(batch["target"])
    losses["target"] = _masked_mean(target_error, batch["target"] >= 0)
    losses["kyoku_delta"] = F.smooth_l1_loss(
        outputs["kyoku_delta"], batch["kyoku_delta"].float() / 10_000.0
    )
    losses["placement"] = F.cross_entropy(outputs["placement"], batch["placement"].long())
    losses["match_score"] = F.smooth_l1_loss(
        outputs["match_score"], batch["match_score"].float() / 10_000.0
    )

    total = sum(losses[name] * weights[name] for name in DEFAULT_WEIGHTS)
    return total, losses
