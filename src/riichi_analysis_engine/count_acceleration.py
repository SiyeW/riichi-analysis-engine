"""Optional, trainer-owned CUDA replay of the unchanged joint-count projection.

One fixed-capacity graph bounds memory independently of observed batch shapes.
Only independent batch rows are padded; no tile/state dimension is altered.
This is a first-order training accelerator, not part of exported model weights.
"""

import torch
from torch import Tensor
from torch.nn import functional as F

from .joint_counts import (
    JointCountPrediction,
    joint_projection_inputs,
    project_joint_probability,
)


class _ReplayResult(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probability, owner):
        ctx.owner = owner
        ctx.used = False
        return probability.clone()

    @staticmethod
    def backward(ctx, gradient):
        if ctx.used:
            raise RuntimeError(
                "CUDA count replay supports one backward per forward; use eager for retained graphs"
            )
        ctx.used = True
        ctx.owner.pending = False
        return gradient, None


class CudaGraphJointCounts:
    """Use forward/backward pairs serially, as in the single-owner trainer."""

    def __init__(self, capacity: int, device: torch.device):
        if capacity < 1 or device.type != "cuda":
            raise ValueError("CUDA count replay requires a positive capacity and CUDA")
        self.capacity = capacity
        self.device = torch.device(
            "cuda",
            device.index if device.index is not None else torch.cuda.current_device(),
        )
        self.pending = False
        # No model or optimizer state participates in capture. Uniform dummy
        # logits and zero constraints exercise the SAME branch-free operations.
        samples = (
            torch.zeros(capacity, 4, 34, 10, device=device, requires_grad=True),
            torch.zeros(capacity, 34, device=device),
            torch.zeros(capacity, 34, device=device),
            torch.zeros(capacity, 4, device=device),
        )
        with (
            torch.cuda.device(self.device),
            torch.autocast(device_type="cuda", enabled=False),
        ):
            self.replay = torch.cuda.make_graphed_callables(
                project_joint_probability, samples
            )

    def __call__(
        self, residual: Tensor, inventory: Tensor, capacities: Tensor
    ) -> JointCountPrediction:
        if self.pending:
            raise RuntimeError(
                "complete CUDA count backward before another forward; use eager for accumulated graphs"
            )
        if not torch.is_grad_enabled() or not residual.requires_grad:
            raise ValueError("CUDA count replay is for gradient-enabled training only")
        if residual.device != self.device:
            raise ValueError("CUDA count replay device mismatch")
        rows = len(residual)
        if not 0 < rows <= self.capacity:
            raise ValueError("CUDA count replay batch exceeds fixed capacity")
        if inventory.requires_grad or capacities.requires_grad:
            raise ValueError("count constraints must be non-trainable")
        baseline, *inputs = joint_projection_inputs(residual, inventory, capacities)
        padded = [
            F.pad(value, (0, 0) * (value.ndim - 1) + (0, self.capacity - rows))
            for value in inputs
        ]
        # Clone prevents a later replay from overwriting a retained prediction.
        self.pending = True
        probability = _ReplayResult.apply(self.replay(*padded)[:rows], self)
        return JointCountPrediction(probability, baseline)
