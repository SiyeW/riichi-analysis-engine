"""Explicit entity routing and repeated cross-modal fusion, model format v14.

Input/target schemas remain v13. This is a new weight topology, never an implicit
reinterpretation of a historical checkpoint.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .architecture import SemanticModelArchitecture
from .semantic_model import (
    DualStreamTransformerBlock,
    MaskedCausalEventBlock,
    PairwiseCategoricalLogits,
    SemanticInputStem,
    SemanticState,
    StructuredSemanticDecoder,
    TileGraphBlock,
    _sinusoidal_positions,
)

ENTITY_ROLES = ("self", "downstream", "opposite", "upstream", "wall", "global")
HIDDEN_SOURCE_ROLES = ENTITY_ROLES[1:5]


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
        # CNN: fuse halfway and at the end, with local processing in between.
        # Transformer: local prior + global/event fusion at every block.
        self.fusion_indices = (
            tuple(range(architecture.backbone_blocks))
            if architecture.backbone == "transformer"
            else tuple(
                sorted(
                    {
                        (architecture.backbone_blocks - 1) // 2,
                        architecture.backbone_blocks - 1,
                    }
                )
            )
        )
        self.fusions = nn.ModuleList(
            DualStreamTransformerBlock(
                width,
                architecture.attention_heads,
                architecture.transformer_ff_multiplier
                if architecture.backbone == "transformer"
                else 2,
            )
            for _ in self.fusion_indices
        )
        if (
            architecture.transformer_tile_prior_blocks
            or architecture.transformer_event_prior_blocks
        ):
            raise ValueError(
                "v14 already includes local priors in each stage; extra v12 priors are unsupported"
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
        fusion_index = 0
        for index, block in enumerate(self.tile_blocks):
            tiles = block(tiles)
            if index in self.fusion_indices:
                state = self.fusions[fusion_index](
                    torch.cat((tiles, entities), dim=1), events, mask
                )
                tiles, entities = state[:, :37], state[:, 37:]
                fusion_index += 1
        return torch.cat((tiles, entities), dim=1), events


class V14Decoder(StructuredSemanticDecoder):
    def __init__(self, architecture: SemanticModelArchitecture) -> None:
        super().__init__(architecture, shared_rule_context=True)
        del self.hidden_count
        self.hidden_joint = PairwiseCategoricalLogits(
            architecture.width, architecture.decoder_width, 10
        )
        self.five_fusion = nn.Linear(architecture.width * 2, architecture.width)
        width = architecture.width
        self.task_queries = nn.Parameter(torch.empty(3, width))
        nn.init.normal_(self.task_queries, std=width**-0.5)
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.task_attention = nn.MultiheadAttention(
            width, architecture.attention_heads, batch_first=True
        )
        self.task_ff = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )

    def task_states(self, state: V14State) -> Tensor:
        """Nine parallel reads: each opponent's shanten, furiten and value query."""
        query = state.players[:, 1:, None, :] + self.task_queries[None, None, :, :]
        query = query.flatten(1, 2)
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
        return (query + self.task_ff(query)).reshape(len(query), 3, 3, -1)

    def forward(self, state: V14State) -> dict[str, Tensor]:
        # Reuse the unchanged conditional wait / shanten / value semantics.
        opponents = state.players[:, 1:]
        tasks = self.task_states(state)
        shanten_state, furiten_state, value_state = tasks.unbind(dim=2)
        tiles = state.tiles[:, :34]
        outputs = {
            "shanten": self.shanten(shanten_state),
            "furiten_no_yaku": self.furiten(furiten_state).squeeze(-1),
            "deal_in_tile": self.wait(opponents, tiles),
            "dora_distribution": self.dora(value_state),
            "dora_tail": self.dora_tail(value_state).squeeze(-1),
            "score_distribution": self.score(value_state),
            "outcome": self.outcome(state.global_state),
            "kyoku_accounts": self.kyoku_accounts(state.global_state),
            "placement": self.placement(state.global_state),
            "match_score": self.match_score(state.global_state),
            "policy": self.policy(state.global_state, state.decision_context),
        }
        count_tiles = tiles.clone()
        count_tiles[:, (4, 13, 22)] = self.five_fusion(
            torch.cat((tiles[:, (4, 13, 22)], state.tiles[:, 34:]), dim=-1)
        ).to(count_tiles.dtype)
        outputs["hidden_joint_residual"] = self.hidden_joint(
            state.hidden_sources, count_tiles
        )
        return outputs


class SemanticV14Model(nn.Module):
    def __init__(self, architecture: SemanticModelArchitecture) -> None:
        super().__init__()
        self.architecture = architecture
        self.input = SemanticInputStem(architecture, shared_rule_context=True)
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
