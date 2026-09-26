"""v15: entity-owned facts enter one common mixed-token reasoning backbone."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .analysis_observation import PLANE_CHANNELS
from .architecture import V15Architecture
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
    EVENT_FIELDS,
    EVENT_FLAGS,
    EVENT_TARGET,
    EVENT_TILE,
    EVENT_TYPE,
    MAX_EVENT_CONSUMED,
    PUBLIC_EVENT_TYPES,
)
from .semantic_model import _sinusoidal_positions
from .semantic_v14 import V14Decoder, V14State
from .v15_facts import GLOBAL_WIDTH, PLAYER_WIDTH, V15_FACTS_WIDTH


class V15Input(nn.Module):
    """Minimal encoding per fact owner; no second history path or broadcast trunk."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.tile_identity = nn.Embedding(38, width, padding_idx=0)
        self.player_identity = nn.Embedding(5, width, padding_idx=0)
        self.rank_identity = nn.Embedding(5, width)
        self.wind_identity = nn.Embedding(5, width)
        self.round_identity = nn.Embedding(5, width)
        self.event_type = nn.Embedding(len(PUBLIC_EVENT_TYPES) + 1, width)
        self.tsumogiri = nn.Embedding(2, width)

        self.tile_rules = nn.Linear(RULE_TILE_CHANNELS, width)
        self.tile_stock = nn.Linear(3, width)
        self.tile_dora = nn.Linear(3, width)
        self.player_numbers = nn.Linear(4, width)
        self.global_numbers = nn.Linear(GLOBAL_WIDTH - 2, width)
        self.wall_number = nn.Linear(1, width)
        self.actor_role = nn.Linear(width, width, bias=False)
        self.target_role = nn.Linear(width, width, bias=False)
        self.tile_roles = nn.ModuleList(
            nn.Linear(width, width, bias=False) for _ in range(1 + MAX_EVENT_CONSUMED)
        )
        self.type_norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(5))

    def forward(
        self,
        observation: Tensor,
        facts: Tensor,
        event_tokens: Tensor,
        event_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if observation.ndim != 3 or observation.shape[2] != TILE_TYPES:
            raise ValueError("v15 observation has the wrong shape")
        if observation.shape[1] < RULE_CONTEXT_START + RULE_TILE_CHANNELS:
            raise ValueError("v15 observation lacks exact self rule facts")
        if facts.shape != (len(observation), V15_FACTS_WIDTH):
            raise ValueError("v15 public facts have the wrong shape")
        if event_tokens.ndim != 3 or event_tokens.shape[2] != EVENT_FIELDS:
            raise ValueError("v15 event fields have the wrong shape")
        if event_mask.shape != event_tokens.shape[:2] or len(event_tokens) != len(
            facts
        ):
            raise ValueError("v15 event mask or batch size does not match")
        if not event_mask.bool().any(dim=-1).all():
            raise ValueError("v15 requires at least one public event")

        rules = observation[
            :, RULE_CONTEXT_START : RULE_CONTEXT_START + RULE_TILE_CHANNELS
        ].transpose(1, 2)
        rules = torch.cat((rules, rules[:, RED_BASE_TILE_INDICES]), dim=1)
        public = observation[:, :PLANE_CHANNELS]
        stock = physical_stock_features(public)
        dora = physical_dora_features(public)
        dora_values = torch.stack(
            (dora.indicator_count, dora.dora_multiplier, dora.aka_bonus), dim=-1
        )
        tile_ids = torch.arange(1, 38, device=observation.device)
        tiles = self.type_norms[0](
            self.tile_identity(tile_ids)[None]
            + self.tile_rules(rules)
            + self.tile_stock(stock)
            + self.tile_dora(dora_values)
        )

        players = facts[:, : 4 * PLAYER_WIDTH].reshape(-1, 4, PLAYER_WIDTH)
        player_ids = torch.arange(1, 5, device=facts.device)
        player_numeric = torch.stack(
            (
                players[..., 0] / 100_000,
                players[..., 3],
                players[..., 4],
                players[..., 5] / 14,
            ),
            dim=-1,
        )
        player_states = self.type_norms[1](
            self.player_identity(player_ids)[None]
            + self.rank_identity(players[..., 1].long())
            + self.wind_identity(players[..., 2].long() + 1)
            + self.player_numbers(player_numeric)
        )

        global_values = facts[:, 4 * PLAYER_WIDTH : 4 * PLAYER_WIDTH + GLOBAL_WIDTH]
        global_numeric = torch.cat(
            (global_values[:, 2:4] / 10, global_values[:, 4:]), dim=-1
        )
        global_state = self.type_norms[2](
            self.round_identity(global_values[:, 0].long())
            + self.wind_identity(global_values[:, 1].long() + 1)
            + self.global_numbers(global_numeric)
        ).unsqueeze(1)
        wall = self.type_norms[3](
            self.wall_number(facts[:, -1:].float() / 70)
        ).unsqueeze(1)

        fields = event_tokens.long()
        actor = fields[..., EVENT_ACTOR]
        target = fields[..., EVENT_TARGET]
        main_tile = fields[..., EVENT_TILE]
        events = (
            self.event_type(fields[..., EVENT_TYPE])
            + self.tsumogiri(fields[..., EVENT_FLAGS] & 1)
            + self.actor_role(self.player_identity(actor)) * (actor != 0).unsqueeze(-1)
            + self.target_role(self.player_identity(target))
            * (target != 0).unsqueeze(-1)
        )
        for offset, projection in enumerate(self.tile_roles):
            tile = (
                main_tile
                if offset == 0
                else fields[..., EVENT_CONSUMED_START + offset - 1]
            )
            events = events + projection(self.tile_identity(tile)) * (
                tile != 0
            ).unsqueeze(-1)
        events = (
            events
            + _sinusoidal_positions(events.shape[1], events.shape[-1], events)[None]
        )
        events = self.type_norms[4](events) * event_mask.unsqueeze(-1)

        values = torch.cat((tiles, player_states, global_state, wall, events), dim=1)
        mask = torch.cat(
            (
                torch.ones(len(values), 43, dtype=torch.bool, device=values.device),
                event_mask.bool(),
            ),
            dim=1,
        )
        return values, mask


class V15SharedBlock(nn.Module):
    def __init__(self, architecture: V15Architecture) -> None:
        super().__init__()
        width = architecture.width
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, architecture.attention_heads, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, architecture.feed_forward_width),
            nn.GELU(),
            nn.Linear(architecture.feed_forward_width, width),
        )

    def forward(self, value: Tensor, mask: Tensor) -> Tensor:
        normalized = self.attention_norm(value)
        update = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~mask,
            need_weights=False,
        )[0]
        value = value + update
        return (value + self.feed_forward(value)) * mask.unsqueeze(-1)


class SemanticV15Model(nn.Module):
    def __init__(self, architecture: V15Architecture) -> None:
        super().__init__()
        self.architecture = architecture
        self.input = V15Input(architecture.width)
        self.blocks = nn.ModuleList(
            V15SharedBlock(architecture) for _ in range(architecture.blocks)
        )
        self.decoder = V14Decoder(architecture)

    def encode(
        self,
        observation: Tensor,
        event_tokens: Tensor,
        event_mask: Tensor,
        facts: Tensor,
    ) -> V14State:
        memory, mask = self.input(observation, facts, event_tokens, event_mask)
        for block in self.blocks:
            memory = block(memory, mask)
        return V14State(
            tiles=memory[:, :37],
            players=memory[:, 37:41],
            global_state=memory[:, 41],
            wall=memory[:, 42],
            events=memory[:, 43:],
            event_mask=event_mask,
            decision_context=None,
        )

    def decode(self, state: V14State) -> dict[str, Tensor]:
        return self.decoder(state)

    def forward(
        self,
        observation: Tensor,
        event_tokens: Tensor,
        event_mask: Tensor,
        facts: Tensor,
    ) -> dict[str, Tensor]:
        return self.decode(self.encode(observation, event_tokens, event_mask, facts))
