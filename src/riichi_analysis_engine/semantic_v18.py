"""Task-specific query construction with one common, complete memory read."""

import torch
from torch import Tensor, nn

from .architecture import V18Architecture
from .branch_execution import execute_branches
from .semantic_v16 import SemanticV16Model, V16Input, V16SharedBlock, V16State
from .semantic_v17 import V17Decoder
from .v16_raw_facts import OWNER_COUNT


def scale_raw_facts(raw: Tensor) -> Tensor:
    """Equal RMS without centering or discarding absolute field values.

    Every valid row has a one-hot owner code whose sum is exactly one. After
    scaling, dividing the row by its owner-code sum reconstructs the input.
    Masked zero rows stay zero. FP32 accumulation is intentional under AMP.
    """
    value = raw.float()
    scale = value.square().mean(dim=-1, keepdim=True).clamp_min(1e-12).rsqrt()
    return (value * scale).to(raw.dtype)


def restore_raw_facts(scaled: Tensor) -> Tensor:
    """Audit helper; not a learned layer or an inference-time correction."""
    reference = scaled[..., :OWNER_COUNT].sum(dim=-1, keepdim=True)
    return scaled / torch.where(reference != 0, reference, torch.ones_like(reference))


class V18Decoder(V17Decoder):
    def __init__(self, architecture: V18Architecture) -> None:
        super().__init__(architecture)
        del self.task_queries  # Private module identity replaces the small task offset.
        self.query_branches = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(architecture.width),
                    nn.Linear(architecture.width, architecture.decoder_width),
                    nn.GELU(),
                    nn.Linear(architecture.decoder_width, architecture.width),
                )
                for name in self.OUTPUT_WIDTHS
            }
        )

    def question_seeds(self, state: V16State) -> dict[str, Tensor]:
        normalized = {
            name: self.query_norm(anchor)
            for name, anchor in self.question_anchors(state).items()
        }
        processed = execute_branches(
            self.query_branches,
            normalized,
            grouped=self.grouped_branch_execution,
            eligible=self.grouped_branch_tasks,
        )
        return {name: value + processed[name] for name, value in normalized.items()}

    def task_states(self, state: V16State) -> dict[str, Tensor]:
        seeds = self.question_seeds(state)
        flattened = [seed.flatten(1, -2) for seed in seeds.values()]
        query = torch.cat(flattened, dim=1)
        # Use the encoder's tile/player/global/wall/event order for both paths.
        memory = torch.cat(
            (
                state.tiles,
                state.players,
                state.global_state[:, None],
                state.wall[:, None],
                state.events,
            ),
            dim=1,
        )
        if state.raw_memory.shape != memory.shape:
            raise ValueError("raw facts must align with the complete learned memory")
        memory = torch.cat(
            (self.memory_norm(memory), scale_raw_facts(state.raw_memory)), dim=1
        )
        mask = torch.cat(
            (
                torch.ones(len(query), 43, device=query.device, dtype=torch.bool),
                state.event_mask.bool(),
            ),
            dim=1,
        )
        mask = torch.cat((mask, mask), dim=1)
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
        return self.process_task_queries(query, seeds, flattened)


class SemanticV18Model(SemanticV16Model):
    def __init__(self, architecture: V18Architecture) -> None:
        nn.Module.__init__(self)
        self.architecture = architecture
        self.input = V16Input(architecture.width)
        self.blocks = nn.ModuleList(
            V16SharedBlock(architecture) for _ in range(architecture.blocks)
        )
        self.decoder = V18Decoder(architecture)
