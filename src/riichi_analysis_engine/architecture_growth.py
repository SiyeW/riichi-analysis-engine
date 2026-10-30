"""Head-aligned, explicitly approximate V18 width/depth warm-start.

This maps model parameters only. Uneven replication changes LayerNorm statistics;
canonical sinusoidal positions also change with width. No optimizer trajectory or
whole-network function preservation is claimed.
"""

import hashlib

import torch
from torch import nn

from .architecture import V18Architecture
from .growth_mapping import (
    Replication,
    expand_linear,
    raw_reader_coefficients,
    split_outgoing_symmetry,
)
from .model import RiichiAnalysisModel, count_parameters
from .v16_raw_facts import MIN_RAW_FACT_WIDTH


def grow_v18(
    parent: RiichiAnalysisModel,
    target: V18Architecture,
    *,
    seed: int = 0,
    symmetry_scale: float = 0.001,
) -> tuple[RiichiAnalysisModel, dict]:
    old = parent.architecture
    if parent.format_version != 18 or not isinstance(old, V18Architecture):
        raise ValueError("growth accepts only explicitly configured V18 parents")
    if target.blocks < old.blocks:
        raise ValueError("growth cannot remove shared blocks")
    if old.width // old.attention_heads != target.width // target.attention_heads:
        raise ValueError("head-aligned growth must preserve attention head dimension")
    if next(parent.parameters()).device.type != "cpu":
        raise ValueError("growth is a CPU artifact operation")
    state = Replication.ordered(old.width, target.width)
    reader = raw_reader_coefficients(state, MIN_RAW_FACT_WIDTH)
    # No mutation of caller RNG, including random neutral-block incoming weights.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = RiichiAnalysisModel(format_version=18, architecture=target)
    source = parent.state_dict()
    if any(not value.isfinite().all() for value in source.values()):
        raise ValueError("parent contains non-finite parameters")
    mapped = {}
    modules = dict(model.named_modules())
    old_modules = dict(parent.named_modules())
    hidden_maps = {}

    def hidden_map(name, source_width, target_width):
        if name not in hidden_maps:
            hidden_maps[name] = Replication.ordered(source_width, target_width)
        result = hidden_maps[name]
        if result.source != source_width or len(result.indices) != target_width:
            raise ValueError("inconsistent feed-forward axis")
        return result

    def put(name, value):
        if name in mapped:
            raise ValueError(f"parameter mapped twice: {name}")
        mapped[name] = value.detach().clone()

    for name, module in old_modules.items():
        new = modules[name]
        prefix = name + "."
        if isinstance(module, nn.MultiheadAttention):
            parts = module.in_proj_weight.detach().chunk(3)
            coeff = reader if name.endswith("decoder.task_attention") else None
            put(
                prefix + "in_proj_weight",
                torch.cat(
                    [
                        expand_linear(
                            part,
                            state,
                            state,
                            column_coefficients=coeff if index else None,
                        )
                        for index, part in enumerate(parts)
                    ]
                ),
            )
            put(
                prefix + "in_proj_bias",
                torch.cat(
                    [
                        part[state.indices]
                        for part in module.in_proj_bias.detach().chunk(3)
                    ]
                ),
            )
        elif isinstance(module, nn.Embedding):
            if module.num_embeddings != new.num_embeddings:
                raise ValueError("growth cannot change categorical identities")
            put(prefix + "weight", module.weight.detach()[:, state.indices])
        elif isinstance(module, nn.LayerNorm):
            put(prefix + "weight", module.weight.detach()[state.indices])
            put(prefix + "bias", module.bias.detach()[state.indices])
        elif isinstance(module, nn.Linear):
            rows = (
                state
                if module.out_features == old.width and new.out_features == target.width
                else Replication.ordered(module.out_features, new.out_features)
            )
            columns = Replication.ordered(module.in_features, new.in_features)
            ff = name.rsplit(".", 1)
            is_ff = (
                ".feed_forward." in name
                or ".query_branches." in name
                or ".task_branches." in name
            )
            if is_ff and ff[1] == "1":
                rows = hidden_map(ff[0], module.out_features, new.out_features)
                columns = state
            elif is_ff and ff[1] == "3":
                rows = state
                columns = hidden_map(ff[0], module.in_features, new.in_features)
            elif name.endswith("decoder.five_fusion"):
                columns = Replication(
                    2 * old.width, torch.cat((state.indices, old.width + state.indices))
                )
            elif module.in_features == old.width and new.in_features == target.width:
                columns = state
            value = expand_linear(module.weight.detach(), rows, columns)
            if is_ff and ff[1] == "3":
                local_seed = int.from_bytes(
                    hashlib.sha256(f"{seed}/{name}".encode()).digest()[:8], "little"
                ) % (2**63)
                value = split_outgoing_symmetry(
                    value, columns, seed=local_seed, relative_scale=symmetry_scale
                )
            put(prefix + "weight", value)
            if module.bias is not None:
                put(prefix + "bias", module.bias.detach()[rows.indices])

    if set(mapped) != set(source):
        raise ValueError(
            f"unmapped parent parameters: {sorted(set(source) - set(mapped))}"
        )
    # New residual blocks start as identity on valid memory. Their incoming
    # projections are nonzero, so outgoing projections can learn on the first step.
    for block in model.semantic_model.blocks[old.blocks :]:
        with torch.no_grad():
            for projection in (
                block.tile_local.self_map,
                block.tile_local.left_map,
                block.tile_local.right_map,
                block.tile_local.red_map,
            ):
                projection.weight.zero_()
            block.attention.out_proj.weight.zero_()
            block.attention.out_proj.bias.zero_()
            block.feed_forward[3].weight.zero_()
            block.feed_forward[3].bias.zero_()
    remaining = model.state_dict()
    for name in remaining.keys() - mapped.keys():
        if not any(
            name.startswith(f"semantic_model.blocks.{index}.")
            for index in range(old.blocks, target.blocks)
        ):
            raise ValueError(f"unexpected new parameter: {name}")
        mapped[name] = remaining[name]
    model.load_state_dict(mapped, strict=True)
    return model, {
        "algorithm": "v18-head-aligned-replication-v1",
        "exactFunctionPreservation": False,
        "limitations": [
            "LayerNorm statistics change under uneven replication",
            "canonical sinusoidal positions depend on the new width",
        ],
        "sourceArchitecture": old.to_dict(),
        "targetArchitecture": target.to_dict(),
        "sourceParameters": count_parameters(parent),
        "targetParameters": count_parameters(model),
        "sourceTensors": len(source),
        "targetTensors": len(mapped),
        "stateMapSha256": state.digest(),
        "activeRawFactWidth": MIN_RAW_FACT_WIDTH,
        "rawAndLearnedReaderProjectionPreservedUnderIdealReplication": True,
        "neutralNewBlocks": list(range(old.blocks, target.blocks)),
        "seed": seed,
        "outgoingZeroSumSymmetryScale": symmetry_scale,
        "optimizerStateMapped": False,
        "resumeAllowed": False,
    }
