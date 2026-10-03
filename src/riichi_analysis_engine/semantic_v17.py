"""V16 input and relational backbone with independent post-read task blocks."""

from torch import Tensor, nn

from .architecture import V17Architecture
from .branch_execution import execute_branches
from .semantic_v16 import SemanticV16Model, V16Decoder, V16Input, V16SharedBlock


class V17Decoder(V16Decoder):
    """One common reader; one equally sized residual block per output task."""

    def __init__(self, architecture: V17Architecture) -> None:
        super().__init__(architecture)
        del self.task_ff
        self.grouped_branch_execution = False
        self.grouped_branch_tasks = None
        self.task_branches = nn.ModuleDict(
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

    def process_task_queries(
        self, query: Tensor, seeds: dict[str, Tensor], flattened: list[Tensor]
    ) -> dict[str, Tensor]:
        parts = query.split([part.shape[1] for part in flattened], dim=1)
        values = dict(zip(seeds, parts, strict=True))
        processed = execute_branches(
            self.task_branches,
            values,
            grouped=self.grouped_branch_execution,
            eligible=self.grouped_branch_tasks,
        )
        return {
            name: (part + processed[name]).reshape(seed.shape)
            for (name, seed), part in zip(seeds.items(), parts, strict=True)
        }


class SemanticV17Model(SemanticV16Model):
    """Reuse the unchanged V16 facts, tile locality and global self-attention."""

    def __init__(self, architecture: V17Architecture) -> None:
        nn.Module.__init__(self)
        self.architecture = architecture
        self.input = V16Input(architecture.width)
        self.blocks = nn.ModuleList(
            V16SharedBlock(architecture) for _ in range(architecture.blocks)
        )
        self.decoder = V17Decoder(architecture)
