"""Successor to v15: exact public melds and directional tile locality."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .architecture import V15Architecture
from .constants import TILES_37
from .semantic_input import EVENT_CONSUMED_START, MAX_EVENT_CONSUMED
from .semantic_v14 import V14Decoder, V14State
from .semantic_v15 import SemanticV15Model, V15Input, V15SharedBlock
from .v16_candidates import CANDIDATE_FEATURES
from .v16_raw_facts import encode_v16_raw_facts


@dataclass(frozen=True)
class V16State(V14State):
    raw_memory: Tensor
    candidate_features: Tensor | None
    candidate_mask: Tensor | None


class V16Decoder(V14Decoder):
    """The same common reader scores every complete executable candidate."""

    def __init__(self, architecture: V15Architecture) -> None:
        super().__init__(architecture)
        del self.action_keys  # v16 has no coarse 46-class policy identity.
        self.candidate_projection = nn.Linear(
            CANDIDATE_FEATURES, architecture.width, bias=False
        )

    def policy_seeds(self, state: V16State, tiles: Tensor) -> Tensor:
        features = state.candidate_features
        if features is None:
            return tiles.new_zeros(len(tiles), 0, tiles.shape[-1])
        if (
            features.ndim != 3
            or features.shape[0] != len(tiles)
            or features.shape[2] != CANDIDATE_FEATURES
        ):
            raise ValueError("v16 complete candidates have the wrong shape")
        return self.source_projection(state.players[:, 0])[
            :, None
        ] + self.candidate_projection(features.to(tiles.dtype))

    def forward(self, state: V16State) -> dict[str, Tensor]:
        outputs = super().forward(state)
        if state.candidate_features is not None:
            if (
                state.candidate_mask is None
                or state.candidate_mask.shape != outputs["policy"].shape
            ):
                raise ValueError("v16 candidates require a matching mask")
            outputs["policy"] = outputs["policy"].masked_fill(
                ~state.candidate_mask.bool(), -torch.inf
            )
        return outputs


def _tile_neighbours() -> tuple[Tensor, Tensor, Tensor]:
    """Return directed same-suit neighbours and ordinary/red-five links."""

    left = [-1] * 37
    right = [-1] * 37
    red = [-1] * 37
    for suit in range(3):
        start = suit * 9
        for rank in range(9):
            index = start + rank
            if rank > 0:
                left[index] = index - 1
            if rank < 8:
                right[index] = index + 1
        red_index = 34 + suit
        left[red_index] = start + 3
        right[red_index] = start + 5
        red[start + 4] = red_index
        red[red_index] = start + 4
    assert len(TILES_37) == 37
    return tuple(torch.tensor(items, dtype=torch.long) for items in (left, right, red))


class DirectionalTileBlock(nn.Module):
    """One role-aware local update; honor/suit boundaries are hard masks."""

    def __init__(self, width: int) -> None:
        super().__init__()
        left, right, red = _tile_neighbours()
        self.register_buffer("left_index", left.clamp_min(0), persistent=False)
        self.register_buffer("right_index", right.clamp_min(0), persistent=False)
        self.register_buffer("red_index", red.clamp_min(0), persistent=False)
        self.register_buffer("left_mask", (left >= 0).float(), persistent=False)
        self.register_buffer("right_mask", (right >= 0).float(), persistent=False)
        self.register_buffer("red_mask", (red >= 0).float(), persistent=False)
        self.norm = nn.LayerNorm(width)
        self.self_map = nn.Linear(width, width, bias=False)
        self.left_map = nn.Linear(width, width, bias=False)
        self.right_map = nn.Linear(width, width, bias=False)
        self.red_map = nn.Linear(width, width, bias=False)
        self.activation = nn.GELU()

    def forward(self, tiles: Tensor) -> Tensor:
        source = self.norm(tiles)
        update = self.self_map(source)
        for index, mask, projection in (
            (self.left_index, self.left_mask, self.left_map),
            (self.right_index, self.right_mask, self.right_map),
            (self.red_index, self.red_mask, self.red_map),
        ):
            update = update + projection(source[:, index]) * mask[None, :, None]
        return tiles + self.activation(update)


class V16Input(V15Input):
    def __init__(self, width: int) -> None:
        super().__init__(width)
        # Replace position-specific consumed-tile maps with one shared set map.
        self.tile_roles = nn.ModuleList((nn.Linear(width, width, bias=False),))
        self.consumed_role = nn.Linear(width, width, bias=False)
        self.meld_counts = nn.Linear(4, width, bias=False)

    def encode_event_tiles(self, fields: Tensor) -> Tensor:
        main = fields[..., 3]
        consumed = fields[
            ..., EVENT_CONSUMED_START : EVENT_CONSUMED_START + MAX_EVENT_CONSUMED
        ]
        main_value = self.tile_roles[0](self.tile_identity(main)) * (
            main != 0
        ).unsqueeze(-1)
        consumed_value = self.tile_identity(consumed) * (consumed != 0).unsqueeze(-1)
        return main_value + self.consumed_role(consumed_value.sum(dim=-2))

    def forward(
        self,
        observation: Tensor,
        facts: Tensor,
        event_tokens: Tensor,
        event_mask: Tensor,
        meld_counts: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if meld_counts.shape != (len(observation), 4, 37):
            raise ValueError("v16 public meld counts must be batch x 4 x 37")
        if not torch.isfinite(meld_counts).all() or (meld_counts < 0).any():
            raise ValueError("v16 public meld counts must be finite and non-negative")
        raw_memory = encode_v16_raw_facts(
            observation,
            facts,
            event_tokens,
            event_mask,
            meld_counts,
            width=self.tile_identity.embedding_dim,
        )
        memory, mask = super().forward(observation, facts, event_tokens, event_mask)
        tiles = memory[:, :37] + self.meld_counts(meld_counts.transpose(1, 2).float())
        return torch.cat((tiles, memory[:, 37:]), dim=1), mask, raw_memory


class V16SharedBlock(V15SharedBlock):
    def __init__(self, architecture: V15Architecture) -> None:
        super().__init__(architecture)
        self.tile_local = DirectionalTileBlock(architecture.width)

    def forward(self, value: Tensor, mask: Tensor) -> Tensor:
        value = torch.cat((self.tile_local(value[:, :37]), value[:, 37:]), dim=1)
        return super().forward(value, mask)


class SemanticV16Model(SemanticV15Model):
    def __init__(self, architecture: V15Architecture) -> None:
        super().__init__(architecture)
        self.input = V16Input(architecture.width)
        self.blocks = nn.ModuleList(
            V16SharedBlock(architecture) for _ in range(architecture.blocks)
        )
        self.decoder = V16Decoder(architecture)

    def encode(
        self,
        observation: Tensor,
        event_tokens: Tensor,
        event_mask: Tensor,
        facts: Tensor,
        meld_counts: Tensor,
        candidate_features: Tensor | None = None,
        candidate_mask: Tensor | None = None,
    ) -> V16State:
        memory, mask, raw_memory = self.input(
            observation, facts, event_tokens, event_mask, meld_counts
        )
        for block in self.blocks:
            memory = block(memory, mask)
        return V16State(
            tiles=memory[:, :37],
            players=memory[:, 37:41],
            global_state=memory[:, 41],
            wall=memory[:, 42],
            events=memory[:, 43:],
            event_mask=event_mask,
            decision_context=None,
            raw_memory=raw_memory,
            candidate_features=candidate_features,
            candidate_mask=candidate_mask,
        )

    def forward(
        self,
        observation: Tensor,
        event_tokens: Tensor,
        event_mask: Tensor,
        facts: Tensor,
        meld_counts: Tensor,
        candidate_features: Tensor | None = None,
        candidate_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        return self.decode(
            self.encode(
                observation,
                event_tokens,
                event_mask,
                facts,
                meld_counts,
                candidate_features,
                candidate_mask,
            )
        )
