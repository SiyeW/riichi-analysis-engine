"""Fixed, inspectable public facts at the shared task reader.

These tokens are not a second learned encoder.  Each owner has a type code and
fixed slots for its exact observable values; the shared reader can attend to
them alongside the relational states without overwriting the original facts.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .analysis_observation import PLANE_CHANNELS
from .constants import TILE_TYPES
from .model_input import RULE_CONTEXT_START
from .physical_tile_features import (
    RED_BASE_TILE_INDICES,
    physical_dora_features,
    physical_stock_features,
)
from .rule_context import RULE_TILE_CHANNELS
from .semantic_input import (
    EVENT_ACTOR,
    EVENT_CONSUMED_START,
    EVENT_FLAGS,
    EVENT_TARGET,
    EVENT_TILE,
    EVENT_TYPE,
    MAX_EVENT_CONSUMED,
    PUBLIC_EVENT_TYPES,
)
from .v15_facts import GLOBAL_WIDTH, PLAYER_WIDTH, V15_FACTS_WIDTH

OWNER_COUNT = 5  # tile, player, global, wall, event
TILE_ID_START = OWNER_COUNT
TILE_RULE_START = TILE_ID_START + 37
TILE_STOCK_START = TILE_RULE_START + RULE_TILE_CHANNELS
TILE_DORA_START = TILE_STOCK_START + 3
TILE_MELD_START = TILE_DORA_START + 3

PLAYER_ID_START = OWNER_COUNT
PLAYER_FACT_START = PLAYER_ID_START + 4
GLOBAL_FACT_START = OWNER_COUNT
WALL_FACT_START = OWNER_COUNT

EVENT_TYPE_START = OWNER_COUNT
EVENT_ACTOR_START = EVENT_TYPE_START + len(PUBLIC_EVENT_TYPES) + 1
EVENT_TARGET_START = EVENT_ACTOR_START + 5
EVENT_TILE_START = EVENT_TARGET_START + 5
EVENT_CONSUMED_START_RAW = EVENT_TILE_START + 38
EVENT_TSUMOGIRI_SLOT = EVENT_CONSUMED_START_RAW + 37
EVENT_POSITION_SLOT = EVENT_TSUMOGIRI_SLOT + 1

MIN_RAW_FACT_WIDTH = max(TILE_MELD_START + 4, EVENT_POSITION_SLOT + 1)


def encode_v16_raw_facts(
    observation: Tensor,
    facts: Tensor,
    event_tokens: Tensor,
    event_mask: Tensor,
    meld_counts: Tensor,
    *,
    width: int = 256,
) -> Tensor:
    """Return [batch, 43 + events, width] with fixed, non-learned slots.

    Counts are only scaled by known constants.  Event consumed tiles are a
    physical-tile multiset, so changing their input order has no effect.
    """

    batch = len(observation)
    if width < MIN_RAW_FACT_WIDTH:
        raise ValueError("shared width cannot hold the fixed raw fact fields")
    if observation.ndim != 3 or observation.shape[2] != TILE_TYPES:
        raise ValueError("v16 raw facts require tile-aligned observations")
    if observation.shape[1] < RULE_CONTEXT_START + RULE_TILE_CHANNELS:
        raise ValueError("v16 raw facts require the complete rule context")
    if facts.shape != (batch, V15_FACTS_WIDTH):
        raise ValueError("v16 raw public facts have the wrong shape")
    if meld_counts.shape != (batch, 4, 37):
        raise ValueError("v16 raw public melds have the wrong shape")
    if (
        event_tokens.ndim != 3
        or event_tokens.shape[0] != batch
        or event_tokens.shape[2] != 9
    ):
        raise ValueError("v16 raw events have the wrong shape")
    if event_mask.shape != event_tokens.shape[:2]:
        raise ValueError("v16 raw event mask has the wrong shape")

    dtype = observation.dtype
    device = observation.device
    tiles = observation.new_zeros(batch, 37, width)
    tiles[..., 0] = 1
    tiles[..., TILE_ID_START:TILE_RULE_START] = torch.eye(
        37, device=device, dtype=dtype
    )
    rules = observation[
        :, RULE_CONTEXT_START : RULE_CONTEXT_START + RULE_TILE_CHANNELS
    ].transpose(1, 2)
    tiles[..., TILE_RULE_START:TILE_STOCK_START] = torch.cat(
        (rules, rules[:, RED_BASE_TILE_INDICES]), dim=1
    )
    public = observation[:, :PLANE_CHANNELS]
    tiles[..., TILE_STOCK_START:TILE_DORA_START] = physical_stock_features(public)
    dora = physical_dora_features(public)
    tiles[..., TILE_DORA_START:TILE_MELD_START] = torch.stack(
        (dora.indicator_count, dora.dora_multiplier, dora.aka_bonus), dim=-1
    )
    tiles[..., TILE_MELD_START : TILE_MELD_START + 4] = (
        meld_counts.transpose(1, 2).to(dtype) / 4
    )

    players = observation.new_zeros(batch, 4, width)
    players[..., 1] = 1
    players[..., PLAYER_ID_START:PLAYER_FACT_START] = torch.eye(
        4, device=device, dtype=dtype
    )
    player_values = (
        facts[:, : 4 * PLAYER_WIDTH].reshape(batch, 4, PLAYER_WIDTH).to(dtype)
    )
    players[..., PLAYER_FACT_START : PLAYER_FACT_START + PLAYER_WIDTH] = (
        player_values * player_values.new_tensor((1 / 100_000, 1, 1, 1, 1, 1 / 14))
    )

    global_state = observation.new_zeros(batch, 1, width)
    global_state[..., 2] = 1
    global_state[..., GLOBAL_FACT_START : GLOBAL_FACT_START + GLOBAL_WIDTH] = facts[
        :, 4 * PLAYER_WIDTH : 4 * PLAYER_WIDTH + GLOBAL_WIDTH
    ].to(dtype)[:, None]
    wall = observation.new_zeros(batch, 1, width)
    wall[..., 3] = 1
    wall[..., WALL_FACT_START] = facts[:, -1:].to(dtype) / 70

    fields = event_tokens.long()
    events = observation.new_zeros(batch, event_tokens.shape[1], width)
    events[..., 4] = 1
    events[..., EVENT_TYPE_START:EVENT_ACTOR_START] = F.one_hot(
        fields[..., EVENT_TYPE], len(PUBLIC_EVENT_TYPES) + 1
    ).to(dtype)
    events[..., EVENT_ACTOR_START:EVENT_TARGET_START] = F.one_hot(
        fields[..., EVENT_ACTOR], 5
    ).to(dtype)
    events[..., EVENT_TARGET_START:EVENT_TILE_START] = F.one_hot(
        fields[..., EVENT_TARGET], 5
    ).to(dtype)
    events[..., EVENT_TILE_START:EVENT_CONSUMED_START_RAW] = F.one_hot(
        fields[..., EVENT_TILE], 38
    ).to(dtype)
    consumed = fields[
        ..., EVENT_CONSUMED_START : EVENT_CONSUMED_START + MAX_EVENT_CONSUMED
    ]
    events[..., EVENT_CONSUMED_START_RAW:EVENT_TSUMOGIRI_SLOT] = (
        F.one_hot((consumed - 1).clamp_min(0), 37)
        .to(dtype)
        .mul((consumed != 0).unsqueeze(-1))
        .sum(dim=-2)
    )
    events[..., EVENT_TSUMOGIRI_SLOT] = (fields[..., EVENT_FLAGS] & 1).to(dtype)
    events[..., EVENT_POSITION_SLOT] = (
        torch.arange(event_tokens.shape[1], device=device, dtype=dtype) / 1024
    )
    events = events * event_mask.to(dtype).unsqueeze(-1)

    return torch.cat((tiles, players, global_state, wall, events), dim=1)
