"""Optional grouped execution of independent, identically shaped task blocks.

Keep the original modules and state-dict keys. Stack live parameters rather than
caching copies, so backward and optimizer/resume ownership remain unchanged.
"""

from collections import defaultdict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def execute_branches(
    branches: nn.ModuleDict,
    values: dict[str, Tensor],
    *,
    grouped: bool = False,
    eligible: set[str] | None = None,
) -> dict[str, Tensor]:
    if not grouped:
        return {name: branches[name](value) for name, value in values.items()}
    groups = defaultdict(list)
    result = {}
    for name, value in values.items():
        if eligible is not None and name not in eligible:
            result[name] = branches[name](value)
            continue
        groups[tuple(value.shape)].append(name)
    for names in groups.values():
        if len(names) == 1:
            name = names[0]
            result[name] = branches[name](values[name])
            continue
        blocks = [branches[name] for name in names]
        norms = [block[0] for block in blocks]
        if any(norm.eps != norms[0].eps for norm in norms):
            raise ValueError("grouped branches require equal normalization epsilon")
        shape = values[names[0]].shape
        value = torch.stack([values[name] for name in names]).reshape(
            len(names), -1, shape[-1]
        )
        value = F.layer_norm(value, norms[0].normalized_shape, eps=norms[0].eps)
        normalized_dtype = value.dtype
        value = value * torch.stack([norm.weight for norm in norms])[:, None]
        value = (value + torch.stack([norm.bias for norm in norms])[:, None]).to(
            normalized_dtype
        )
        for index in (1, 3):
            layers = [block[index] for block in blocks]
            weights = torch.stack([layer.weight for layer in layers])
            value = torch.bmm(value, weights.transpose(1, 2))
            value = (
                value
                + torch.stack([layer.bias for layer in layers]).to(value.dtype)[:, None]
            )
            if index == 1:
                value = F.gelu(value)
        for name, item in zip(names, value.unbind(), strict=True):
            result[name] = item.reshape(shape)
    return {name: result[name] for name in values}


def supervised_branch_names(active: dict[str, bool]) -> set[str]:
    """Map existing CPU label availability to decoder blocks, not new sampling."""
    from .semantic_v14 import V14Decoder

    return {
        name
        for name in V14Decoder.OUTPUT_WIDTHS
        if active.get(name, active.get("analysis", False))
    }


def packed_gradients_are_finite(
    parameters: list[nn.Parameter], *, chunk_elements: int = 4_194_304
) -> bool:
    """Bounded concatenation reduces small checks; no norm overflow or clipping.

    This diagnostic alternative trades copying for fewer GPU kernel launches.
    It must be benchmarked, not assumed faster than the per-tensor check.
    """
    groups = defaultdict(list)
    for parameter in parameters:
        if parameter.grad is not None:
            gradient = parameter.grad.detach()
            groups[(gradient.device, gradient.dtype)].append(gradient.reshape(-1))
    flags = []
    for gradients in groups.values():
        pending = []
        size = 0
        for gradient in gradients:
            for part in gradient.split(chunk_elements):
                if pending and size + part.numel() > chunk_elements:
                    flags.append(torch.isfinite(torch.cat(pending)).all())
                    pending, size = [], 0
                pending.append(part)
                size += part.numel()
        if pending:
            flags.append(torch.isfinite(torch.cat(pending)).all())
    # Trainers own one device. Preserve support for diagnostic mixed-device lists.
    return all(
        bool(torch.stack(items).all())
        for items in (
            [flag for flag in flags if flag.device == device]
            for device in {flag.device for flag in flags}
        )
    )
