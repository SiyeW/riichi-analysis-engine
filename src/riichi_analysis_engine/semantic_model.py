"""Semantic v12 backbones and vectorized structured decoders."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .analysis_observation import PLANE_CHANNELS
from .architecture import SemanticModelArchitecture
from .constants import ACTION_SPACE, TILE_TYPES
from .kyoku_outcome import OUTCOME_COUNT
from .model_input import (
    POLICY_CONTEXT_CHANNELS,
    POLICY_CONTEXT_START,
    RULE_CONTEXT_START,
)
from .physical_tile_features import (
    PHYSICAL_TILE_TYPES,
    RED_BASE_TILE_INDICES,
    RED_ENTITY_FEATURE_NAMES,
    physical_dora_features,
    physical_red_features,
)
from .prediction_values import DORA_VALUES, SCORE_VALUES
from .rule_context import RULE_GLOBAL_CHANNELS, RULE_TILE_CHANNELS
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


@dataclass(frozen=True)
class SemanticState:
    tiles: Tensor
    players: Tensor
    global_state: Tensor
    decision_context: Tensor | None


def _sinusoidal_positions(length: int, width: int, reference: Tensor) -> Tensor:
    """Generate deterministic absolute positions without a fixed length limit."""

    position = torch.arange(length, device=reference.device, dtype=torch.float32)
    frequency = torch.exp(
        torch.arange(0, width, 2, device=reference.device, dtype=torch.float32)
        * (-log(10_000.0) / width)
    )
    encoding = torch.zeros(length, width, device=reference.device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position[:, None] * frequency)
    encoding[:, 1::2] = torch.cos(
        position[:, None] * frequency[: encoding[:, 1::2].shape[1]]
    )
    return encoding.to(dtype=reference.dtype)


class SemanticInputStem(nn.Module):
    """Turn audited planes and event fields into entity-aligned tokens."""

    def __init__(
        self, architecture: SemanticModelArchitecture, *, shared_rule_context: bool
    ) -> None:
        super().__init__()
        width = architecture.width
        self.tile_stem = nn.Sequential(
            nn.Linear(PLANE_CHANNELS, architecture.stem_width),
            nn.LayerNorm(architecture.stem_width),
            nn.GELU(),
            nn.Linear(architecture.stem_width, width),
        )
        self.tile_identity = nn.Embedding(PHYSICAL_TILE_TYPES, width)
        self.dora_attributes = nn.Linear(4, width, bias=False)
        self.red_attributes = nn.Linear(
            len(RED_ENTITY_FEATURE_NAMES), width, bias=False
        )
        event_width = architecture.event_width
        self.event_type = nn.Embedding(len(PUBLIC_EVENT_TYPES) + 1, event_width)
        self.event_actor = nn.Embedding(5, event_width)
        self.event_target = nn.Embedding(5, event_width)
        self.event_tile = nn.Embedding(PHYSICAL_TILE_TYPES + 1, event_width)
        self.event_flags = nn.Embedding(256, event_width)
        self.event_output = nn.Sequential(
            nn.LayerNorm(event_width),
            nn.Linear(event_width, width),
        )
        self.shared_rule_context = shared_rule_context
        if shared_rule_context:
            self.rule_tiles = nn.Sequential(
                nn.Linear(RULE_TILE_CHANNELS, architecture.stem_width),
                nn.GELU(),
                nn.Linear(architecture.stem_width, width),
            )
            self.rule_global = nn.Sequential(
                nn.Linear(RULE_GLOBAL_CHANNELS, architecture.stem_width),
                nn.GELU(),
                nn.Linear(architecture.stem_width, width),
            )
        else:
            self.policy_context = nn.Sequential(
                nn.Linear(POLICY_CONTEXT_CHANNELS, architecture.stem_width),
                nn.GELU(),
                nn.Linear(architecture.stem_width, width),
            )

    def forward(
        self, observation: Tensor, event_tokens: Tensor, event_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if observation.ndim != 3 or observation.shape[2] != TILE_TYPES:
            raise ValueError("semantic model received a malformed observation")
        required = (
            RULE_CONTEXT_START + RULE_TILE_CHANNELS + RULE_GLOBAL_CHANNELS
            if self.shared_rule_context
            else POLICY_CONTEXT_START + POLICY_CONTEXT_CHANNELS
        )
        if observation.shape[1] < required:
            raise ValueError("semantic model received an incompatible input contract")
        if event_tokens.ndim != 3 or event_tokens.shape[2] != EVENT_FIELDS:
            raise ValueError("semantic model received malformed event tokens")
        if event_mask.shape != event_tokens.shape[:2]:
            raise ValueError("semantic event mask does not match event tokens")
        if event_tokens.shape[0] != observation.shape[0]:
            raise ValueError("semantic inputs do not share one batch size")

        analysis = observation[:, :PLANE_CHANNELS]
        base_tiles = analysis.transpose(1, 2)
        red_tiles = base_tiles[:, RED_BASE_TILE_INDICES]
        physical_tiles = torch.cat((base_tiles, red_tiles), dim=1)
        tile_state = self.tile_stem(physical_tiles)
        identities = torch.arange(PHYSICAL_TILE_TYPES, device=observation.device)
        tile_state = tile_state + self.tile_identity(identities).unsqueeze(0)
        tile_state = tile_state + self.dora_attributes(
            physical_dora_features(analysis).stacked()
        )
        tile_state = tile_state + self.red_attributes(physical_red_features(analysis))
        if self.shared_rule_context:
            rule_tiles = observation[
                :, RULE_CONTEXT_START : RULE_CONTEXT_START + RULE_TILE_CHANNELS
            ].transpose(1, 2)
            physical_rule_tiles = torch.cat(
                (rule_tiles, rule_tiles[:, RED_BASE_TILE_INDICES]), dim=1
            )
            tile_state = tile_state + self.rule_tiles(physical_rule_tiles)

        fields = event_tokens.long()
        event_state = (
            self.event_type(fields[..., EVENT_TYPE])
            + self.event_actor(fields[..., EVENT_ACTOR])
            + self.event_target(fields[..., EVENT_TARGET])
            + self.event_tile(fields[..., EVENT_TILE])
            + self.event_flags(fields[..., EVENT_FLAGS])
        )
        for offset in range(MAX_EVENT_CONSUMED):
            event_state = event_state + self.event_tile(
                fields[..., EVENT_CONSUMED_START + offset]
            )
        event_state = self.event_output(event_state)
        event_state = event_state * event_mask.unsqueeze(-1)
        context = (
            self.encode_rule_context(observation)
            if self.shared_rule_context
            else self.encode_policy_context(observation)
        )
        return tile_state, event_state, context

    def encode_rule_context(self, observation: Tensor) -> Tensor:
        start = RULE_CONTEXT_START + RULE_TILE_CHANNELS
        values = observation[:, start : start + RULE_GLOBAL_CHANNELS].mean(dim=-1)
        return self.rule_global(values)

    def encode_policy_context(self, observation: Tensor) -> Tensor:
        if observation.ndim != 3 or observation.shape[1] < POLICY_CONTEXT_START:
            raise ValueError("semantic policy context received a malformed observation")
        policy = observation[:, POLICY_CONTEXT_START:].mean(dim=-1)
        return self.policy_context(policy)


class TileGraphBlock(nn.Module):
    """Relation-aware CNN block without crossing suit and honor boundaries."""

    def __init__(self, width: int, *, prior_version: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.self_projection = nn.Linear(width, width, bias=False)
        self.neighbor_projection = nn.Linear(width, width, bias=False)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )
        adjacency = torch.eye(PHYSICAL_TILE_TYPES)
        groups = (
            ((0, 9), (9, 18), (18, 27), (27, 31), (31, 34))
            if prior_version == 1
            else ((0, 9), (9, 18), (18, 27))
        )
        for start, stop in groups:
            for tile in range(start, stop):
                if tile > start:
                    adjacency[tile, tile - 1] = 1
                if tile + 1 < stop:
                    adjacency[tile, tile + 1] = 1
        for red, base in enumerate(RED_BASE_TILE_INDICES, start=TILE_TYPES):
            adjacency[red, base] = 1
            adjacency[base, red] = 1
        adjacency /= adjacency.sum(dim=-1, keepdim=True)
        self.register_buffer("adjacency", adjacency)

    def forward(self, value: Tensor) -> Tensor:
        normalized = self.norm(value)
        neighbors = torch.einsum("ij,bjd->bid", self.adjacency, normalized)
        value = (
            value
            + self.self_projection(normalized)
            + self.neighbor_projection(neighbors)
        )
        return value + self.feed_forward(value)


class MaskedCausalEventBlock(nn.Module):
    """Append-safe event convolution that never reads padded or future tokens."""

    def __init__(
        self, width: int, *, first_dilation: int = 1, second_dilation: int = 1
    ) -> None:
        super().__init__()
        self.first = nn.Conv1d(width, width, 3, dilation=first_dilation)
        self.second = nn.Conv1d(width, width, 3, dilation=second_dilation)

    @staticmethod
    def _causal(convolution: nn.Conv1d, value: Tensor) -> Tensor:
        left_padding = convolution.dilation[0] * (convolution.kernel_size[0] - 1)
        return convolution(F.pad(value, (left_padding, 0)))

    def forward(self, value: Tensor, mask: Tensor) -> Tensor:
        channel_mask = mask.unsqueeze(1)
        update = self._causal(self.first, value * channel_mask)
        update = F.gelu(update) * channel_mask
        update = self._causal(self.second, update)
        return (value + update) * channel_mask


class MaskedCausalEventPriorBlock(nn.Module):
    """Cheap local event prior that preserves the complete causal memory."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 3, groups=width)
        self.pointwise = nn.Linear(width, width)

    def forward(self, value: Tensor, mask: Tensor) -> Tensor:
        token_mask = mask.unsqueeze(-1)
        normalized = self.norm(value) * token_mask
        local = self.depthwise(F.pad(normalized.transpose(1, 2), (2, 0))).transpose(
            1, 2
        )
        update = self.pointwise(F.gelu(local)) * token_mask
        return (value + update) * token_mask


class CNNBackbone(nn.Module):
    def __init__(self, architecture: SemanticModelArchitecture) -> None:
        super().__init__()
        width = architecture.width
        self.tile_blocks = nn.Sequential(
            *(
                TileGraphBlock(width, prior_version=architecture.semantic_prior_version)
                for _ in range(architecture.backbone_blocks)
            )
        )
        dilation_pairs = ((1, 1),) * architecture.event_blocks
        if architecture.semantic_prior_version >= 2:
            long_range = ((1, 2), (4, 8), (16, 32))
            dilation_pairs = tuple(
                long_range[index] if index < len(long_range) else (1, 2)
                for index in range(architecture.event_blocks)
            )
        self.event_blocks = nn.ModuleList(
            MaskedCausalEventBlock(
                width,
                first_dilation=first_dilation,
                second_dilation=second_dilation,
            )
            for first_dilation, second_dilation in dilation_pairs
        )
        self.position_events = architecture.semantic_prior_version >= 2
        self.player_queries = nn.Parameter(torch.empty(4, width))
        self.global_query = nn.Parameter(torch.empty(1, width))
        self.readout = nn.MultiheadAttention(
            width, architecture.attention_heads, batch_first=True
        )
        self.event_fusion = nn.MultiheadAttention(
            width, architecture.attention_heads, batch_first=True
        )
        nn.init.normal_(self.player_queries, std=width**-0.5)
        nn.init.normal_(self.global_query, std=width**-0.5)

    def forward(
        self,
        tiles: Tensor,
        events: Tensor,
        event_mask: Tensor,
        shared_context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        tiles = self.tile_blocks(tiles)
        if self.position_events:
            events = events + _sinusoidal_positions(
                events.shape[1], events.shape[-1], events
            ).unsqueeze(0) * event_mask.unsqueeze(-1)
        event_sequence = events.transpose(1, 2)
        for block in self.event_blocks:
            event_sequence = block(event_sequence, event_mask)
        events = event_sequence.transpose(1, 2) * event_mask.unsqueeze(-1)
        batch = len(tiles)
        queries = torch.cat((self.player_queries, self.global_query), dim=0)
        queries = queries.unsqueeze(0).expand(batch, -1, -1)
        if shared_context is not None:
            queries = queries + shared_context.unsqueeze(1)
        entities, _ = self.readout(queries, tiles, tiles, need_weights=False)
        entities_from_events, _ = self.event_fusion(
            entities,
            events,
            events,
            key_padding_mask=~event_mask.bool(),
            need_weights=False,
        )
        entities = entities + entities_from_events
        return tiles, entities[:, :4], entities[:, 4]


class DualStreamTransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff_multiplier: int) -> None:
        super().__init__()
        self.state_norm = nn.LayerNorm(width)
        self.state_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.state_ff = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * ff_multiplier),
            nn.GELU(),
            nn.Linear(width * ff_multiplier, width),
        )
        self.event_norm = nn.LayerNorm(width)

    def forward(self, state: Tensor, events: Tensor, event_mask: Tensor) -> Tensor:
        state_norm = self.state_norm(state)
        state = (
            state
            + self.state_attention(
                state_norm, state_norm, state_norm, need_weights=False
            )[0]
        )
        state = (
            state
            + self.cross_attention(
                self.state_norm(state),
                self.event_norm(events),
                self.event_norm(events),
                key_padding_mask=~event_mask.bool(),
                need_weights=False,
            )[0]
        )
        return state + self.state_ff(state)


class TransformerBackbone(nn.Module):
    def __init__(self, architecture: SemanticModelArchitecture) -> None:
        super().__init__()
        width = architecture.width
        self.player_tokens = nn.Parameter(torch.empty(4, width))
        self.global_token = nn.Parameter(torch.empty(1, width))
        self.tile_prior_blocks = nn.Sequential(
            *(
                TileGraphBlock(width, prior_version=architecture.semantic_prior_version)
                for _ in range(architecture.transformer_tile_prior_blocks)
            )
        )
        self.event_prior_blocks = nn.ModuleList(
            MaskedCausalEventPriorBlock(width)
            for _ in range(architecture.transformer_event_prior_blocks)
        )
        event_layer = nn.TransformerEncoderLayer(
            width,
            architecture.attention_heads,
            dim_feedforward=width * architecture.transformer_ff_multiplier,
            batch_first=True,
            norm_first=True,
        )
        self.event_encoder = nn.TransformerEncoder(
            event_layer, architecture.event_blocks, enable_nested_tensor=False
        )
        self.blocks = nn.ModuleList(
            DualStreamTransformerBlock(
                width,
                architecture.attention_heads,
                architecture.transformer_ff_multiplier,
            )
            for _ in range(architecture.backbone_blocks)
        )
        nn.init.normal_(self.player_tokens, std=width**-0.5)
        nn.init.normal_(self.global_token, std=width**-0.5)

    def forward(
        self,
        tiles: Tensor,
        events: Tensor,
        event_mask: Tensor,
        shared_context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        tiles = self.tile_prior_blocks(tiles)
        batch = len(tiles)
        entity_tokens = (
            torch.cat((self.player_tokens, self.global_token), dim=0)
            .unsqueeze(0)
            .expand(batch, -1, -1)
        )
        if shared_context is not None:
            entity_tokens = entity_tokens + shared_context.unsqueeze(1)
        state = torch.cat(
            (
                tiles,
                entity_tokens,
            ),
            dim=1,
        )
        event_mask = event_mask.bool()
        event_length = events.shape[1]
        events = events + _sinusoidal_positions(
            event_length, events.shape[-1], events
        ).unsqueeze(0) * event_mask.unsqueeze(-1)
        for block in self.event_prior_blocks:
            events = block(events, event_mask)
        causal_mask = torch.ones(
            event_length,
            event_length,
            dtype=torch.bool,
            device=events.device,
        ).triu(1)
        events = self.event_encoder(
            events,
            mask=causal_mask,
            src_key_padding_mask=~event_mask,
        ) * event_mask.unsqueeze(-1)
        for block in self.blocks:
            state = block(state, events, event_mask)
        return (
            state[:, :PHYSICAL_TILE_TYPES],
            state[:, PHYSICAL_TILE_TYPES : PHYSICAL_TILE_TYPES + 4],
            state[:, -1],
        )


class PairwiseLogits(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.left = nn.Linear(width, hidden)
        self.right = nn.Linear(width, hidden)
        self.output = nn.Linear(hidden, 1)

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        pair = self.left(left).unsqueeze(2) + self.right(right).unsqueeze(1)
        return self.output(torch.nn.functional.gelu(pair)).squeeze(-1)


class StructuredSemanticDecoder(nn.Module):
    def __init__(
        self, architecture: SemanticModelArchitecture, *, shared_rule_context: bool
    ) -> None:
        super().__init__()
        width = architecture.width
        hidden = architecture.decoder_width
        self.shanten = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 7)
        )
        self.furiten = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.wait = PairwiseLogits(width, hidden)
        self.hidden = PairwiseLogits(width, hidden)
        self.hidden_red = PairwiseLogits(width, hidden)
        self.dora = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, len(DORA_VALUES))
        )
        self.dora_tail = nn.Linear(width, 1)
        self.score = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, len(SCORE_VALUES))
        )
        self.outcome = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, OUTCOME_COUNT)
        )
        self.kyoku_accounts = nn.Linear(width, 5)
        self.placement = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 24)
        )
        self.match_score = nn.Linear(width, 4)
        self.shared_rule_context = shared_rule_context
        self.policy_query = nn.Sequential(
            nn.Linear(width if shared_rule_context else width * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, width),
        )
        self.action_keys = nn.Embedding(ACTION_SPACE, width)

    def forward(self, state: SemanticState) -> dict[str, Tensor]:
        opponents = state.players[:, 1:]
        base_tiles = state.tiles[:, :TILE_TYPES]
        red_tiles = state.tiles[:, TILE_TYPES:]
        return {
            "shanten": self.shanten(opponents),
            "furiten_no_yaku": self.furiten(opponents).squeeze(-1),
            "deal_in_tile": self.wait(opponents, base_tiles),
            "hidden_source_affinity": self.hidden(state.players, base_tiles),
            "hidden_red_source": self.hidden_red(red_tiles, state.players),
            "dora_distribution": self.dora(opponents),
            "dora_tail": self.dora_tail(opponents).squeeze(-1),
            "score_distribution": self.score(opponents),
            "outcome": self.outcome(state.global_state),
            "kyoku_accounts": self.kyoku_accounts(state.global_state),
            "placement": self.placement(state.global_state),
            "match_score": self.match_score(state.global_state),
            "policy": self.policy(state.global_state, state.decision_context),
        }

    def policy(self, global_state: Tensor, decision_context: Tensor | None) -> Tensor:
        policy_input = (
            global_state
            if self.shared_rule_context
            else torch.cat((global_state, decision_context), dim=-1)
        )
        policy_query = self.policy_query(policy_input)
        return torch.einsum("bd,ad->ba", policy_query, self.action_keys.weight) / sqrt(
            global_state.shape[-1]
        )


class SemanticRiichiModel(nn.Module):
    """One v12 model contract with interchangeable CNN/Transformer backbones."""

    def __init__(
        self,
        architecture: SemanticModelArchitecture,
        *,
        shared_rule_context: bool = False,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.shared_rule_context = shared_rule_context
        self.input = SemanticInputStem(
            architecture, shared_rule_context=shared_rule_context
        )
        self.backbone: CNNBackbone | TransformerBackbone = (
            CNNBackbone(architecture)
            if architecture.backbone == "cnn"
            else TransformerBackbone(architecture)
        )
        self.decoder = StructuredSemanticDecoder(
            architecture, shared_rule_context=shared_rule_context
        )

    def forward(
        self, observation: Tensor, event_tokens: Tensor, event_mask: Tensor
    ) -> dict[str, Tensor]:
        return self.decode(self.encode(observation, event_tokens, event_mask))

    def encode(
        self, observation: Tensor, event_tokens: Tensor, event_mask: Tensor
    ) -> SemanticState:
        tiles, events, context = self.input(observation, event_tokens, event_mask)
        tiles, players, global_state = self.backbone(
            tiles,
            events,
            event_mask,
            context if self.shared_rule_context else None,
        )
        return SemanticState(
            tiles=tiles,
            players=players,
            global_state=global_state,
            decision_context=context,
        )

    def decode(self, state: SemanticState) -> dict[str, Tensor]:
        return self.decoder(state)

    def decode_policy(self, state: SemanticState, observation: Tensor) -> Tensor:
        if self.shared_rule_context:
            current = state.decision_context
            if current is None:
                raise RuntimeError("shared rule context is missing from semantic state")
            updated = self.input.encode_rule_context(observation)
            return self.decoder.policy(state.global_state - current + updated, None)
        return self.decoder.policy(
            state.global_state,
            self.input.encode_policy_context(observation),
        )
