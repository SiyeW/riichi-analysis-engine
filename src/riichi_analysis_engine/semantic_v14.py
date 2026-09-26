"""Convolutional input encoders and one shared task readout, model format v14.

Input/target schemas remain v13. This is a new weight topology, never an implicit
reinterpretation of a historical checkpoint.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor, nn

from .analysis_observation import PLANE_CHANNELS
from .architecture import SemanticModelArchitecture
from .constants import ACTION_SPACE
from .kyoku_outcome import OUTCOME_COUNT
from .physical_tile_features import (
    PHYSICAL_STOCK_FEATURE_NAMES,
    physical_stock_features,
)
from .prediction_values import DORA_VALUES, SCORE_VALUES
from .semantic_model import (
    MaskedCausalEventBlock,
    SemanticInputStem,
    SemanticState,
    TileGraphBlock,
    _sinusoidal_positions,
)

ENTITY_ROLES = ("self", "downstream", "opposite", "upstream", "wall", "global")
HIDDEN_SOURCE_ROLES = ENTITY_ROLES[1:5]


class V14InputStem(SemanticInputStem):
    """Preserve legacy stems while exposing exact v14 physical stock facts."""

    def __init__(self, architecture: SemanticModelArchitecture) -> None:
        super().__init__(architecture, shared_rule_context=True, role_aware_events=True)
        self.stock_attributes = nn.Linear(
            len(PHYSICAL_STOCK_FEATURE_NAMES), architecture.width, bias=False
        )

    def forward(self, observation: Tensor, event_tokens: Tensor, event_mask: Tensor):
        tiles, events, context = super().forward(observation, event_tokens, event_mask)
        tiles = tiles + self.stock_attributes(
            physical_stock_features(observation[:, :PLANE_CHANNELS])
        )
        return tiles, events, context


@dataclass(frozen=True)
class V14State(SemanticState):
    wall: Tensor
    events: Tensor
    event_mask: Tensor

    @property
    def hidden_sources(self) -> Tensor:
        return torch.cat((self.players[:, 1:], self.wall.unsqueeze(1)), dim=1)


class V14Backbone(nn.Module):
    def __init__(self, architecture: SemanticModelArchitecture) -> None:
        super().__init__()
        width = architecture.width
        if architecture.backbone != "cnn":
            raise ValueError("v14 currently supports only the reviewed CNN backbone")
        self.entities = nn.Parameter(torch.empty(len(ENTITY_ROLES), width))
        nn.init.normal_(self.entities, std=width**-0.5)
        self.tile_blocks = nn.ModuleList(
            TileGraphBlock(width, prior_version=architecture.semantic_prior_version)
            for _ in range(architecture.backbone_blocks)
        )
        dilations = ((1, 2), (4, 8), (16, 32))
        self.event_blocks = nn.ModuleList(
            MaskedCausalEventBlock(
                width,
                first_dilation=dilations[i % 3][0],
                second_dilation=dilations[i % 3][1],
            )
            for i in range(architecture.event_blocks)
        )
        if (
            architecture.transformer_tile_prior_blocks
            or architecture.transformer_event_prior_blocks
        ):
            raise ValueError(
                "v14 uses convolutional input encoders; extra transformer priors are unsupported"
            )

    def forward(
        self, tiles: Tensor, events: Tensor, mask: Tensor, context: Tensor
    ) -> tuple[Tensor, Tensor]:
        if (~mask.bool().any(-1)).any():
            raise ValueError("v14 requires at least one public event per sample")
        events = events + _sinusoidal_positions(
            events.shape[1], events.shape[-1], events
        ).unsqueeze(0) * mask.unsqueeze(-1)
        events = events.transpose(1, 2)
        for block in self.event_blocks:
            events = block(events, mask)
        events = events.transpose(1, 2)
        entities = self.entities.unsqueeze(0).expand(len(tiles), -1, -1)
        entities = entities + context.unsqueeze(1)
        for block in self.tile_blocks:
            tiles = block(tiles)
        # No repeatedly injected history and no Transformer state blocks. Both
        # encoded modalities remain unpooled for the single common task reader.
        return torch.cat((tiles, entities), dim=1), events


class V14Decoder(nn.Module):
    """One batched reader for all semantic questions, then small output maps.

    Queries retain their entity/tile anchors through a residual connection. All
    queries read the same unpooled memory; none communicate through another
    task's prediction or get a private attention/FFN stack.
    """

    OUTPUT_WIDTHS: ClassVar[dict[str, int]] = {
        "shanten": 7,
        "furiten_no_yaku": 1,
        "dora_distribution": len(DORA_VALUES),
        "score_distribution": len(SCORE_VALUES),
        "deal_in_tile": 1,
        "hidden_joint_residual": 10,
        "outcome": OUTCOME_COUNT,
        "kyoku_accounts": 5,
        "placement": 24,
        "match_score": 4,
        "policy": 1,
    }

    def __init__(self, architecture: SemanticModelArchitecture) -> None:
        super().__init__()
        width = architecture.width
        self.source_projection = nn.Linear(width, width)
        self.tile_projection = nn.Linear(width, width)
        self.five_fusion = nn.Linear(width * 2, width)
        self.action_keys = nn.Embedding(ACTION_SPACE, width)
        self.task_queries = nn.ParameterDict(
            {name: nn.Parameter(torch.empty(width)) for name in self.OUTPUT_WIDTHS}
        )
        for query in self.task_queries.values():
            nn.init.normal_(query, std=width**-0.5)
        nn.init.normal_(self.action_keys.weight, std=width**-0.5)
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.task_attention = nn.MultiheadAttention(
            width, architecture.attention_heads, batch_first=True
        )
        self.task_ff = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, architecture.decoder_width),
            nn.GELU(),
            nn.Linear(architecture.decoder_width, width),
        )
        self.outputs = nn.ModuleDict(
            {name: nn.Linear(width, size) for name, size in self.OUTPUT_WIDTHS.items()}
        )
        # The open-ended dora bucket needs a tail estimate, not a second query.
        self.dora_tail = nn.Linear(width, 1)
        nn.init.zeros_(self.outputs["hidden_joint_residual"].weight)
        nn.init.zeros_(self.outputs["hidden_joint_residual"].bias)

    def question_seeds(self, state: V14State) -> dict[str, Tensor]:
        """Semantic axes, not arbitrary learned slots: 300 questions in total."""
        opponents = self.source_projection(state.players[:, 1:])
        global_state = self.source_projection(state.global_state)
        tiles = self.tile_projection(state.tiles)
        count_tiles = state.tiles[:, :34].clone()
        count_tiles[:, (4, 13, 22)] = self.five_fusion(
            torch.cat((state.tiles[:, (4, 13, 22)], state.tiles[:, 34:]), dim=-1)
        ).to(count_tiles.dtype)
        # 0..36 are discard identities; 37..45 are non-discard actions.
        action_tiles = torch.cat(
            (tiles, tiles.new_zeros(len(tiles), ACTION_SPACE - 37, tiles.shape[-1])),
            dim=1,
        )
        seeds = {
            name: opponents
            for name in (
                "shanten",
                "furiten_no_yaku",
                "dora_distribution",
                "score_distribution",
            )
        }
        seeds["deal_in_tile"] = opponents[:, :, None] + tiles[:, None, :34]
        seeds["hidden_joint_residual"] = (
            self.source_projection(state.hidden_sources)[:, :, None]
            + self.tile_projection(count_tiles)[:, None]
        )
        for name in ("outcome", "kyoku_accounts", "placement", "match_score"):
            seeds[name] = global_state[:, None]
        seeds["policy"] = (
            self.source_projection(state.players[:, 0])[:, None]
            + self.action_keys.weight[None]
            + action_tiles
        )
        return {name: seed + self.task_queries[name] for name, seed in seeds.items()}

    def task_states(self, state: V14State) -> dict[str, Tensor]:
        seeds = self.question_seeds(state)
        flattened = [
            seed.reshape(len(seed), -1, seed.shape[-1]) for seed in seeds.values()
        ]
        query = torch.cat(flattened, dim=1)
        memory = torch.cat(
            (
                state.tiles,
                state.players,
                state.wall[:, None],
                state.global_state[:, None],
                state.events,
            ),
            dim=1,
        )
        mask = torch.cat(
            (
                torch.ones(len(query), 43, device=query.device, dtype=torch.bool),
                state.event_mask.bool(),
            ),
            dim=1,
        )
        memory = self.memory_norm(memory)
        query = (
            query
            + self.task_attention(
                self.query_norm(query),
                memory,
                memory,
                key_padding_mask=~mask,
                need_weights=False,
            )[0]
        )
        parts = (query + self.task_ff(query)).split(
            [part.shape[1] for part in flattened], dim=1
        )
        return {
            name: part.reshape(seed.shape)
            for (name, seed), part in zip(seeds.items(), parts, strict=True)
        }

    def forward(self, state: V14State) -> dict[str, Tensor]:
        tasks = self.task_states(state)
        outputs = {}
        for name, head in self.outputs.items():
            output = head(tasks[name])
            if name in ("outcome", "kyoku_accounts", "placement", "match_score"):
                output = output.squeeze(1)
            if self.OUTPUT_WIDTHS[name] == 1:
                output = output.squeeze(-1)
            outputs[name] = output
        outputs["dora_tail"] = self.dora_tail(tasks["dora_distribution"]).squeeze(-1)
        return outputs


class SemanticV14Model(nn.Module):
    def __init__(self, architecture: SemanticModelArchitecture) -> None:
        super().__init__()
        self.architecture = architecture
        self.input = V14InputStem(architecture)
        self.backbone = V14Backbone(architecture)
        self.decoder = V14Decoder(architecture)

    def encode(
        self, observation: Tensor, event_tokens: Tensor, event_mask: Tensor
    ) -> V14State:
        tiles, events, context = self.input(observation, event_tokens, event_mask)
        state, events = self.backbone(tiles, events, event_mask, context)
        return V14State(
            tiles=state[:, :37],
            players=state[:, 37:41],
            wall=state[:, 41],
            global_state=state[:, 42],
            decision_context=context,
            events=events,
            event_mask=event_mask,
        )

    def decode(self, state: V14State) -> dict[str, Tensor]:
        return self.decoder(state)

    def forward(
        self, observation: Tensor, event_tokens: Tensor, event_mask: Tensor
    ) -> dict[str, Tensor]:
        return self.decode(self.encode(observation, event_tokens, event_mask))

    def decode_policy(self, state: V14State, observation: Tensor) -> Tensor:
        raise ValueError("v14 policy context changes require exact re-encoding")
