"""Semantic checks for a structured supervised-training batch.

The packed-storage checks prove that an archive can be decoded.  They cannot
prove that it was produced by the observation and label contract the
structured model needs.  Keep that distinction explicit: this module checks a
small decoded batch before a long run consumes any samples.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from .analysis_observation import channel_index as analysis_channel_index
from .constants import ACTION_SPACE, OBS_CHANNELS, RANK_FEATURE_CHANNELS, TILE_TYPES
from .hidden_transport import physical_hidden_counts
from .kyoku_outcome import OUTCOME_COUNT
from .model_input import MODEL_INPUT_CHANNELS, SHARED_MODEL_INPUT_CHANNELS
from .observation_layout import (
    JIKAZE_CHANNEL,
    MORTAL_ANALYSIS_CHANNELS,
    WIND_TILE_START,
)
from .prediction_values import (
    DEALER_SCORE_VALUE_SET,
    NON_DEALER_SCORE_VALUE_SET,
    SCORE_VALUE_SET,
)
from .semantic_input import (
    EVENT_ACTOR,
    EVENT_FIELDS,
    EVENT_TARGET,
    EVENT_TILE,
    EVENT_TYPE,
    PUBLIC_EVENT_TYPE_TO_ID,
    PUBLIC_EVENT_TYPES,
)

V8_TRAINING_FIELDS = frozenset(
    {
        "obs",
        "action_mask",
        "policy",
        "shanten",
        "furiten_no_yaku",
        "deal_in_tile",
        "concealed_count",
        "concealed_red_count",
        "wall_count",
        "wall_red_count",
        "dora",
        "score",
        "winner_mask",
        "outcome",
        "kyoku_delta",
        "placement",
        "match_score",
    }
)
SEMANTIC_TRAINING_FIELDS = frozenset({"event_tokens", "event_mask"})


def _fail(message: str) -> None:
    raise ValueError(f"structured training-batch contract violation: {message}")


def _shape(batch: Mapping[str, Tensor], name: str, expected: tuple[int, ...]) -> Tensor:
    value = batch[name]
    if tuple(value.shape[1:]) != expected:
        tail = ", ".join(map(str, expected)) or ""
        _fail(f"{name} has shape {tuple(value.shape)}, expected (batch, {tail})")
    return value


def _finite(batch: Mapping[str, Tensor]) -> None:
    for name, value in batch.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            _fail(f"{name} contains a non-finite value")


def _observation_contract(observation: Tensor) -> Tensor:
    if observation.ndim != 3 or observation.shape[2] != TILE_TYPES:
        _fail(
            f"obs has shape {tuple(observation.shape)}, expected a supported structured input"
        )
    channels = int(observation.shape[1])
    if channels in {MODEL_INPUT_CHANNELS, SHARED_MODEL_INPUT_CHANNELS}:
        jikaze = analysis_channel_index("jikaze")
        rank_start = analysis_channel_index("rank_p0_r0")
    elif channels == OBS_CHANNELS:
        jikaze = JIKAZE_CHANNEL
        rank_start = MORTAL_ANALYSIS_CHANNELS
    else:
        _fail(f"obs has unsupported channel count {channels}")
    winds = observation[:, jikaze, WIND_TILE_START : WIND_TILE_START + 4]
    if not torch.equal(
        (winds > 0).sum(dim=-1),
        torch.ones(len(observation), dtype=torch.long, device=observation.device),
    ):
        _fail("obs does not encode exactly one controlled-player seat wind")

    ranks = observation[:, rank_start : rank_start + RANK_FEATURE_CHANNELS].reshape(
        len(observation), 4, 4, TILE_TYPES
    )
    if not torch.equal(ranks, ranks[..., :1].expand_as(ranks)):
        _fail("all-player rank planes are not constant across tile positions")
    active_ranks = ranks[..., 0] > 0
    if not torch.equal(
        active_ranks.sum(dim=-1),
        torch.ones((len(observation), 4), dtype=torch.long, device=observation.device),
    ):
        _fail("obs does not encode exactly one current rank for every player")
    return winds.argmax(dim=-1)


def _policy_contract(batch: Mapping[str, Tensor], batch_size: int) -> None:
    raw_mask = _shape(batch, "action_mask", (ACTION_SPACE,))
    if not ((raw_mask == 0) | (raw_mask == 1)).all():
        _fail("action_mask must be binary")
    mask = raw_mask.bool()
    policy = _shape(batch, "policy", ()).long()
    if len(policy) != batch_size:
        _fail("policy batch length differs from obs")
    if (policy < -1).any():
        _fail("policy contains a negative value other than -1")
    valid = policy >= 0
    if (policy[valid] >= ACTION_SPACE).any():
        _fail("policy contains an action outside the action space")
    rows = torch.arange(batch_size, device=policy.device)[valid]
    if valid.any() and not mask[rows, policy[valid]].all():
        _fail("policy contains an action that its legal-action mask rejects")


def _target_contract(
    batch: Mapping[str, Tensor], batch_size: int, self_wind: Tensor
) -> None:
    shanten = _shape(batch, "shanten", (3,)).long()
    if len(shanten) != batch_size or ((shanten < 0) | (shanten > 6)).any():
        _fail("shanten must contain only classes 0 through 6")
    for name, expected in (
        ("furiten_no_yaku", (3,)),
        ("winner_mask", (3,)),
    ):
        value = _shape(batch, name, expected)
        if not ((value == 0) | (value == 1)).all():
            _fail(f"{name} must be binary")
    deal_in = _shape(batch, "deal_in_tile", (3, TILE_TYPES))
    if not ((deal_in == 0) | (deal_in == 1)).all():
        _fail("deal_in_tile must be binary")
    count_fields = (
        ("concealed_count", (3, TILE_TYPES)),
        ("concealed_red_count", (3, 3)),
        ("wall_count", (TILE_TYPES,)),
        ("wall_red_count", (3,)),
    )
    for name, expected in count_fields:
        value = _shape(batch, name, expected)
        if (value < 0).any() or not torch.equal(value, value.round()):
            _fail(f"{name} must contain non-negative whole-tile counts")
    physical_hidden_counts(
        batch["concealed_count"],
        batch["wall_count"],
        batch["concealed_red_count"],
        batch["wall_red_count"],
    )

    winner_mask = batch["winner_mask"].bool()
    score = _shape(batch, "score", (3,)).long()
    supported_scores = torch.as_tensor(tuple(SCORE_VALUE_SET), device=score.device)
    invalid_score = winner_mask & ~torch.isin(score, supported_scores)
    if invalid_score.any():
        _fail("winner score is outside the supported riichi score table")
    dealer_relative = 3 - self_wind
    for opponent in range(3):
        winner = winner_mask[:, opponent]
        if not winner.any():
            continue
        # The allowed set depends on the row, so test the two groups separately.
        is_dealer = dealer_relative == opponent
        for rows, values in (
            (winner & is_dealer, DEALER_SCORE_VALUE_SET),
            (winner & ~is_dealer, NON_DEALER_SCORE_VALUE_SET),
        ):
            allowed_scores = torch.as_tensor(tuple(values), device=score.device)
            if (
                rows.any()
                and not torch.isin(score[rows, opponent], allowed_scores).all()
            ):
                _fail("winner score contradicts the controlled player's seat wind")

    outcome = _shape(batch, "outcome", ()).long()
    if ((outcome < 0) | (outcome >= OUTCOME_COUNT)).any():
        _fail(f"outcome must contain only classes 0 through {OUTCOME_COUNT - 1}")
    placement = _shape(batch, "placement", ()).long()
    if ((placement < 0) | (placement >= 24)).any():
        _fail("placement must contain only classes 0 through 23")
    _shape(batch, "kyoku_delta", (4,))
    _shape(batch, "match_score", (4,))


def validate_v8_training_batch(
    batch: Mapping[str, Tensor], *, require_analysis_active: bool = False
) -> dict[str, int]:
    """Reject a decoded batch that cannot safely supervise a v8/v9 model.

    The function is deliberately bounded to the supplied batch.  It is a
    startup preflight, not another full-corpus scan, and it shares the exact
    physical-inventory implementation used by the loss.
    """

    missing = sorted(V8_TRAINING_FIELDS.difference(batch))
    if missing:
        _fail(f"missing fields: {', '.join(missing)}")
    _finite(batch)
    observation = batch["obs"]
    self_wind = _observation_contract(observation)
    batch_size = len(observation)
    inconsistent = [
        name for name in V8_TRAINING_FIELDS if len(batch[name]) != batch_size
    ]
    if inconsistent:
        _fail(
            f"fields do not share obs batch length: {', '.join(sorted(inconsistent))}"
        )
    _policy_contract(batch, batch_size)
    analysis_active = batch.get("analysis_active")
    if require_analysis_active and analysis_active is None:
        _fail("missing field: analysis_active")
    if analysis_active is not None:
        analysis_active = _shape(batch, "analysis_active", ())
        if len(analysis_active) != batch_size:
            _fail("analysis_active batch length differs from obs")
        if not ((analysis_active == 0) | (analysis_active == 1)).all():
            _fail("analysis_active must be binary")
    _target_contract(batch, batch_size, self_wind)
    return {
        "samples": batch_size,
        "seatWinds": int(self_wind.unique().numel()),
        **(
            {"analysisSamples": int(analysis_active.bool().sum())}
            if analysis_active is not None
            else {}
        ),
    }


def validate_semantic_training_batch(
    batch: Mapping[str, Tensor],
    *,
    require_analysis_active: bool = True,
    require_hidden_baseline_anchor: bool = False,
) -> dict[str, int]:
    """Validate the v12 event-memory extension without weakening v8 targets."""

    result = validate_v8_training_batch(
        batch, require_analysis_active=require_analysis_active
    )
    missing = sorted(SEMANTIC_TRAINING_FIELDS.difference(batch))
    if missing:
        _fail(f"missing fields: {', '.join(missing)}")
    tokens = _shape(batch, "event_tokens", (batch["event_tokens"].shape[1], EVENT_FIELDS))
    mask = _shape(batch, "event_mask", (tokens.shape[1],))
    if len(tokens) != result["samples"]:
        _fail("event memory batch length differs from obs")
    if not ((mask == 0) | (mask == 1)).all():
        _fail("event_mask must be binary")
    mask = mask.bool()
    if not mask.any(dim=1).all():
        _fail("every semantic sample must retain at least one public event")
    if (tokens[~mask] != 0).any():
        _fail("padded event tokens must be zero")
    valid = tokens[mask].long()
    if ((valid[:, EVENT_TYPE] < 1) | (valid[:, EVENT_TYPE] > len(PUBLIC_EVENT_TYPES))).any():
        _fail("event memory contains an unsupported event type")
    for field in (EVENT_ACTOR, EVENT_TARGET):
        if ((valid[:, field] < 0) | (valid[:, field] > 4)).any():
            _fail("event memory contains an invalid player reference")
    if ((valid[:, EVENT_TILE] < 0) | (valid[:, EVENT_TILE] > 37)).any():
        _fail("event memory contains an invalid physical tile reference")
    if not (
        tokens[:, 0, EVENT_TYPE] == PUBLIC_EVENT_TYPE_TO_ID["start_kyoku"]
    ).all():
        _fail("event memory must begin at start_kyoku")
    opponent_draw = (
        tokens[..., EVENT_TYPE] == PUBLIC_EVENT_TYPE_TO_ID["tsumo"]
    ) & (tokens[..., EVENT_ACTOR] != 1) & mask
    if (tokens[..., EVENT_TILE][opponent_draw] != 0).any():
        _fail("opponent draw identity leaked into semantic event memory")
    anchor = batch.get("hidden_baseline_anchor")
    if require_hidden_baseline_anchor and anchor is None:
        _fail("missing field: hidden_baseline_anchor")
    if anchor is not None:
        anchor = _shape(batch, "hidden_baseline_anchor", ())
        if len(anchor) != result["samples"]:
            _fail("hidden_baseline_anchor batch length differs from obs")
        if not ((anchor == 0) | (anchor == 1)).all():
            _fail("hidden_baseline_anchor must be binary")
    return {
        **result,
        "eventTokens": int(mask.sum()),
        "maxEventHistory": int(mask.sum(dim=1).max()),
        **(
            {"hiddenBaselineAnchors": int(anchor.bool().sum())}
            if anchor is not None
            else {}
        ),
    }
