from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .prediction_values import DORA_TAIL_START, SCORE_VALUES

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
    if winner_mask.any():
        dora_labels = batch["dora"].long()[winner_mask]
        dora_classes = dora_labels.clamp_max(DORA_TAIL_START)
        dora_distribution = F.cross_entropy(
            outputs["dora_distribution"][winner_mask], dora_classes
        )
        dora_point = F.mse_loss(
            F.softplus(outputs["dora_point"])[winner_mask], dora_labels.float()
        )
        score_labels = batch["score"].long()[winner_mask]
        score_distribution = F.cross_entropy(
            outputs["score_distribution"][winner_mask], score_class_indices(score_labels)
        )
        score_point = F.mse_loss(
            F.softplus(outputs["score_point"])[winner_mask], score_labels.float() / 1000.0
        )
        losses["dora"] = dora_distribution + dora_point
        losses["score"] = score_distribution + score_point
    else:
        losses["dora"] = outputs["dora_distribution"].sum() * 0
        losses["score"] = outputs["score_distribution"].sum() * 0

    any_win = batch["win"].bool().any(dim=-1)
    any_win_loss = F.binary_cross_entropy_with_logits(
        outputs["outcome_any_win"], any_win.float()
    )
    winner_loss = F.binary_cross_entropy_with_logits(
        outputs["outcome_winner"], batch["win"].float(), reduction="none"
    )
    losses["outcome"] = any_win_loss + _masked_mean(
        winner_loss, any_win.unsqueeze(-1).expand_as(winner_loss)
    )
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
