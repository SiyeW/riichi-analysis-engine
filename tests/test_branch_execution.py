"""Grouped task execution retains independent parameters and resume ownership."""

import copy

import pytest
import torch
from torch import nn

from riichi_analysis_engine.branch_execution import (
    execute_branches,
    packed_gradients_are_finite,
    supervised_branch_names,
)
from riichi_analysis_engine.train import gradients_are_finite


@pytest.mark.parametrize(
    "device,amp", [("cpu", False), ("cuda", False), ("cuda", True)]
)
def test_full_model_grouped_loss_and_gradient_parity(device, amp):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from test_v17 import training_batch

    from riichi_analysis_engine.architecture import V18Architecture
    from riichi_analysis_engine.losses import BatchSupervision, multitask_loss
    from riichi_analysis_engine.model import RiichiAnalysisModel
    from riichi_analysis_engine.semantic_v17 import V17Decoder
    from riichi_analysis_engine.train import forward_batch

    torch.manual_seed(42)
    cpu = {
        name: value.long()
        if value.dtype in {torch.uint16, torch.uint32, torch.uint64}
        else value
        for name, value in training_batch().items()
    }
    supervision = BatchSupervision.from_cpu(cpu)
    batch = {name: value.to(device) for name, value in cpu.items()}
    first = RiichiAnalysisModel(
        format_version=18,
        architecture=V18Architecture(
            width=128, blocks=1, feed_forward_width=256, decoder_width=256
        ),
    ).to(device)
    second = copy.deepcopy(first)
    for module in second.modules():
        if isinstance(module, V17Decoder):
            module.grouped_branch_execution = True
            module.grouped_branch_tasks = supervised_branch_names(
                dict(supervision.active)
            )
    assert first.state_dict().keys() == second.state_dict().keys()
    outputs, losses = [], []
    for model in (first, second):
        with torch.autocast(device, dtype=torch.float16, enabled=amp):
            output = forward_batch(model, batch)
            total, terms, _, _ = multitask_loss(output, batch, supervision=supervision)
        outputs.append(output)
        losses.append(terms)
        total.backward()
    tolerance = {"rtol": 0.02, "atol": 0.003} if amp else {"rtol": 2e-4, "atol": 2e-5}
    for name in outputs[0]:
        torch.testing.assert_close(outputs[0][name], outputs[1][name], **tolerance)
    for name in losses[0]:
        torch.testing.assert_close(losses[0][name], losses[1][name], **tolerance)
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        if left.grad is None:
            assert right.grad is None
        else:
            torch.testing.assert_close(left.grad, right.grad, **tolerance)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_grouped_branches_outputs_gradients_and_resume(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(19)
    branches = nn.ModuleDict(
        {
            name: nn.Sequential(
                nn.LayerNorm(32), nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32)
            )
            for name in ("a", "b", "c", "d")
        }
    ).to(device)
    other = copy.deepcopy(branches)
    values = {
        name: torch.randn(2, rows, 32, device=device, requires_grad=True)
        for name, rows in zip(branches, (3, 3, 5, 5), strict=True)
    }
    copies = {
        name: value.detach().clone().requires_grad_() for name, value in values.items()
    }
    first = execute_branches(branches, values)
    second = execute_branches(other, copies, grouped=True)
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=2e-5, atol=2e-6)
    sum(value.square().mean() for value in first.values()).backward()
    sum(value.square().mean() for value in second.values()).backward()
    for left, right in zip(branches.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=3e-5, atol=3e-6)
    for name, value in values.items():
        torch.testing.assert_close(value.grad, copies[name].grad, rtol=3e-5, atol=3e-6)
    optimizer = torch.optim.AdamW(other.parameters(), lr=1e-4)
    optimizer.step()
    restored = copy.deepcopy(other)
    resumed = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for model, opt in ((other, optimizer), (restored, resumed)):
        opt.zero_grad(set_to_none=True)
        result = execute_branches(model, copies, grouped=True)
        sum(value.square().mean() for value in result.values()).backward()
        opt.step()
    for left, right in zip(other.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("bad", [None, float("inf"), float("nan")])
def test_packed_finite_preserves_checks_and_gradients(device, bad):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    parameters = [nn.Parameter(torch.zeros(size, device=device)) for size in (0, 7, 13)]
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    if bad is not None:
        parameters[-1].grad[-1] = bad
    before = [parameter.grad.clone() for parameter in parameters]
    assert packed_gradients_are_finite(
        parameters, chunk_elements=8
    ) == gradients_are_finite(parameters)
    for parameter, saved in zip(parameters, before, strict=True):
        torch.testing.assert_close(
            parameter.grad, saved, equal_nan=True, rtol=0, atol=0
        )
    assert packed_gradients_are_finite([])
