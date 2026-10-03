"""Capability-dispatched AdamW execution, independent of model serialization."""

from collections.abc import Iterable

import torch


def adamw_execution(device: torch.device, requested: str) -> str:
    if requested not in {"auto", "foreach", "fused"}:
        raise ValueError("unsupported AdamW execution mode")
    if requested == "foreach":
        return requested
    if device.type != "cuda":
        if requested == "fused":
            raise ValueError("fused execution requires CUDA")
        return "default"
    # Check the actual installed torch/device combination using disposable state.
    # No model, RNG, training cursor, or existing optimizer is touched.
    parameter = torch.nn.Parameter(torch.ones(1, device=device))
    try:
        probe = torch.optim.AdamW([parameter], fused=True)
        parameter.grad = torch.ones_like(parameter)
        probe.step()
        torch.cuda.synchronize(device)
    except (RuntimeError, TypeError, NotImplementedError):
        if requested == "fused":
            raise
        return "foreach"
    return "fused"


def set_adamw_execution(optimizer: torch.optim.AdamW, execution: str) -> None:
    """Apply AFTER load_state_dict, which otherwise restores the old flags."""
    if execution not in {"default", "foreach", "fused"}:
        raise ValueError("unsupported effective AdamW execution mode")
    for group in optimizer.param_groups:
        group["fused"] = True if execution == "fused" else None
        group["foreach"] = True if execution == "foreach" else None


def make_adamw(
    groups: Iterable, device: torch.device, requested: str = "auto"
) -> tuple[torch.optim.AdamW, str]:
    execution = adamw_execution(device, requested)
    optimizer = torch.optim.AdamW(groups)
    set_adamw_execution(optimizer, execution)
    return optimizer, execution
