from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor


def conditional_deal_in_probabilities(outputs: Mapping[str, Tensor]) -> Tensor:
    """Compose absolute tile risk from tenpai, ron eligibility, and legal waits."""

    tenpai = outputs["shanten"].softmax(-1)[..., 0]
    eligible = 1.0 - outputs["furiten_no_yaku"].sigmoid()
    legal_wait = outputs["deal_in_tile"].sigmoid()
    return tenpai.unsqueeze(-1) * eligible.unsqueeze(-1) * legal_wait


def zero_sum_accounts(values: Tensor) -> Tensor:
    """Project account changes onto exact zero-sum conservation."""

    return values - values.mean(dim=-1, keepdim=True)


def fixed_total_values(values: Tensor, total: Tensor) -> Tensor:
    """Project four predicted values onto a supplied system total."""

    if values.shape[-1] != 4:
        raise ValueError("fixed-total projection requires four player values")
    return values + (total.unsqueeze(-1) - values.sum(dim=-1, keepdim=True)) / 4.0
