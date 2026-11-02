"""A second task-conditioned read, without another private processing stack."""

from torch import Tensor, nn

from .architecture import V19Architecture
from .semantic_v16 import V16Input, V16SharedBlock, V16State
from .semantic_v18 import SemanticV18Model, V18Decoder


class V19Decoder(V18Decoder):
    def __init__(self, architecture: V19Architecture) -> None:
        super().__init__(architecture)
        self.refine_query_norm = nn.LayerNorm(architecture.width)
        self.refine_attention = nn.MultiheadAttention(
            architecture.width, architecture.attention_heads, batch_first=True
        )
        # Only the final projection is zero. Q/K/V remain nonzero so its output
        # projection can learn before gradients open the interior of this branch.
        nn.init.zeros_(self.refine_attention.out_proj.weight)
        nn.init.zeros_(self.refine_attention.out_proj.bias)

    def task_states(self, state: V16State) -> dict[str, Tensor]:
        query, seeds, flattened, memory, mask = self._first_task_read(state)
        query = (
            query
            + self.refine_attention(
                self.refine_query_norm(query),
                memory,
                memory,
                key_padding_mask=~mask,
                need_weights=False,
            )[0]
        )
        return self.process_task_queries(query, seeds, flattened)


class SemanticV19Model(SemanticV18Model):
    def __init__(self, architecture: V19Architecture) -> None:
        nn.Module.__init__(self)
        self.architecture = architecture
        self.input = V16Input(architecture.width)
        self.blocks = nn.ModuleList(
            V16SharedBlock(architecture) for _ in range(architecture.blocks)
        )
        self.decoder = V19Decoder(architecture)
