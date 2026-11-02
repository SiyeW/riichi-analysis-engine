"""Explicit identity initialization for a second common task read."""

import torch

from .architecture import V19Architecture
from .model import RiichiAnalysisModel, count_parameters

PREFIX = "semantic_model.decoder."
NEW_PARAMETERS = (
    "refine_query_norm.weight",
    "refine_query_norm.bias",
    "refine_attention.in_proj_weight",
    "refine_attention.in_proj_bias",
    "refine_attention.out_proj.weight",
    "refine_attention.out_proj.bias",
)


def deepen_task_reader(
    parent: RiichiAnalysisModel, *, seed: int = 0
) -> tuple[RiichiAnalysisModel, dict]:
    """Copy all existing tensors; only the new output projection starts at zero."""
    if parent.format_version != 18:
        raise ValueError("reader transfer requires an unchanged V18 parent")
    if next(parent.parameters()).device.type != "cpu":
        raise ValueError("reader transfer is a CPU artifact operation")
    source = parent.state_dict()
    if any(tensor.dtype != torch.float32 for tensor in source.values()):
        raise ValueError("reader transfer requires FP32 parent parameters")
    if any(not tensor.isfinite().all() for tensor in source.values()):
        raise ValueError("reader transfer parent contains non-finite parameters")
    target = V19Architecture(**parent.architecture.to_dict(), readout_stages=2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = RiichiAnalysisModel(format_version=19, architecture=target)
    values = model.state_dict()
    expected_new = {PREFIX + name for name in NEW_PARAMETERS}
    if values.keys() - source.keys() != expected_new or source.keys() - values.keys():
        raise ValueError("reader transfer has an unexpected parameter boundary")
    for name, tensor in source.items():
        if tensor.shape != values[name].shape:
            raise ValueError("reader transfer cannot resize an old tensor: " + name)
        values[name] = tensor.detach().clone()
    copies = {
        "refine_query_norm.weight": "query_norm.weight",
        "refine_query_norm.bias": "query_norm.bias",
        "refine_attention.in_proj_weight": "task_attention.in_proj_weight",
        "refine_attention.in_proj_bias": "task_attention.in_proj_bias",
    }
    for new, old in copies.items():
        values[PREFIX + new] = source[PREFIX + old].detach().clone()
    for suffix in ("weight", "bias"):
        name = PREFIX + "refine_attention.out_proj." + suffix
        values[name] = torch.zeros_like(values[name])
    model.load_state_dict(values, strict=True)
    return model, {
        "type": "identity-task-reader-expansion-v1",
        "sourceFormat": 18,
        "targetFormat": 19,
        "sourceArchitecture": parent.architecture.to_dict(),
        "targetArchitecture": target.to_dict(),
        "sourceParameters": count_parameters(parent),
        "targetParameters": count_parameters(model),
        "oldTensorCopies": list(source),
        "newTensorCopies": {PREFIX + new: PREFIX + old for new, old in copies.items()},
        "zeroTensors": [
            PREFIX + "refine_attention.out_proj." + suffix
            for suffix in ("weight", "bias")
        ],
        "newArchitectureUpdates": 0,
        "identityForFiniteIntermediateValues": True,
        "numericalIdentityNotYetVerified": True,
        "seed": seed,
    }
