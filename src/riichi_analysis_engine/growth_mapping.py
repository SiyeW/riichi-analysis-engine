"""Explicit replication maps used by architecture warm-starts, not optimizers."""

import hashlib
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Replication:
    source: int
    indices: Tensor

    @classmethod
    def ordered(cls, source: int, target: int) -> "Replication":
        if target < source or source <= 0:
            raise ValueError("replication requires positive, non-shrinking dimensions")
        return cls(source, torch.arange(target) % source)

    @property
    def counts(self) -> Tensor:
        return torch.bincount(self.indices, minlength=self.source)

    def coefficients(self) -> Tensor:
        return 1.0 / self.counts[self.indices]

    def digest(self) -> str:
        return hashlib.sha256(str(self.indices.tolist()).encode("ascii")).hexdigest()


def expand_linear(
    weight: Tensor,
    rows: Replication,
    columns: Replication,
    *,
    column_coefficients: Tensor | None = None,
) -> Tensor:
    if weight.shape != (rows.source, columns.source):
        raise ValueError("linear weight does not match its declared axis maps")
    coefficients = (
        columns.coefficients() if column_coefficients is None else column_coefficients
    )
    if coefficients.shape != columns.indices.shape:
        raise ValueError("column coefficients have the wrong shape")
    return weight[rows.indices][:, columns.indices] * coefficients.to(weight)[None]


def raw_reader_coefficients(state: Replication, raw_width: int) -> Tensor:
    """A common K/V map for replicated learned states AND RMS-scaled raw facts.

    Raw facts are zero-padded, not replicated. Their RMS multiplier changes by
    sqrt(new/old width). Set each original active raw column to its reciprocal;
    distribute the remaining learned-path coefficient among its replicas.
    """
    target = len(state.indices)
    if target == state.source:
        return state.coefficients()
    if not torch.equal(state.indices[: state.source], torch.arange(state.source)):
        raise ValueError("raw reader requires original columns first")
    if raw_width > state.source or (state.counts[:raw_width] < 2).any():
        raise ValueError("growth must replicate every active raw-fact column")
    coefficients = state.coefficients()
    original = (state.source / target) ** 0.5
    for index in range(raw_width):
        positions = (state.indices == index).nonzero().flatten()
        coefficients[positions[0]] = original
        coefficients[positions[1:]] = (1 - original) / (len(positions) - 1)
    return coefficients


def split_outgoing_symmetry(
    weight: Tensor, hidden: Replication, *, seed: int, relative_scale: float
) -> Tensor:
    """Zero-sum column perturbation; equal cloned activations still sum equally."""
    if relative_scale < 0 or not torch.isfinite(torch.tensor(relative_scale)):
        raise ValueError("symmetry scale must be finite and non-negative")
    result = weight.clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scale = weight.float().std().item() * relative_scale
    for index in range(hidden.source):
        positions = (hidden.indices == index).nonzero().flatten().tolist()
        if len(positions) > 1:
            noise = (
                torch.randn(
                    len(weight),
                    len(positions) - 1,
                    generator=generator,
                    dtype=weight.dtype,
                )
                * scale
            )
            # Use basic integer slices, not mixed scalar/advanced-index in-place
            # writes. The latter faulted inside the Windows CPU torch kernel.
            for offset, position in enumerate(positions[1:]):
                result[:, position] = result[:, position] + noise[:, offset]
            result[:, positions[0]] -= noise.sum(1)
    return result
