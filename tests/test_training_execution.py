"""Acceleration preserves task masks, gradients and optimizer state."""

import copy

import pytest
import torch
from test_v17 import training_batch

from riichi_analysis_engine.architecture import V18Architecture
from riichi_analysis_engine.losses import (
    LOSS_TERMS_V8,
    BatchSupervision,
    LearnedUncertaintyBalancer,
    multitask_loss,
)
from riichi_analysis_engine.model import RiichiAnalysisModel
from riichi_analysis_engine.optimizer_execution import make_adamw, set_adamw_execution
from riichi_analysis_engine.train import forward_batch, validate


def test_distance_validation_keeps_probability_metrics_and_core_selection():
    batch = {
        name: value.long()
        if value.dtype in {torch.uint16, torch.uint32, torch.uint64}
        else value
        for name, value in training_batch().items()
    }
    model = RiichiAnalysisModel(
        format_version=18,
        architecture=V18Architecture(
            width=128, blocks=1, feed_forward_width=256, decoder_width=256
        ),
    )
    balancer = LearnedUncertaintyBalancer(LOSS_TERMS_V8)
    baseline = validate(model, balancer, [batch], torch.device("cpu"), progress_every=0)
    ordered = validate(
        model,
        balancer,
        [batch],
        torch.device("cpu"),
        progress_every=0,
        ordered_distance=True,
    )
    for name, value in baseline.items():
        if name.startswith(("metric/", "Selection/", "null/")):
            assert ordered[name] == value
    for name in (
        "shanten",
        "hidden_allocation",
        "dora_distribution",
        "score_distribution",
        "placement",
    ):
        assert ordered["lossComponent/base/" + name] == baseline[name]
        assert ordered[name] == pytest.approx(
            baseline[name] + ordered["lossComponent/distance/" + name]
        )


@pytest.mark.parametrize("analysis", ["all", "mixed", "none"])
@pytest.mark.parametrize("conditional", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("ordered_distance", [False, True])
def test_cpu_supervision_exact_loss_and_gradient_parity(
    analysis, conditional, device, ordered_distance
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(42)
    # Match PackDataset's storage-dtype normalization. Raw converter arrays
    # contain uint32 provenance; older CUDA/PyTorch cannot boolean-index it.
    batch = {
        name: value.long()
        if value.dtype in {torch.uint16, torch.uint32, torch.uint64}
        else value
        for name, value in training_batch().items()
    }
    if analysis == "mixed":
        batch["analysis_active"][1:] = False
    elif analysis == "none":
        batch["analysis_active"][:] = False
    if conditional:
        batch["shanten"][:] = 0
        batch["winner_mask"][:, 0] = True
        batch["score"][:, 0] = 2000
        batch["current_concealed_dora"][:, 0] = 8
    else:
        batch["shanten"][:] = 2
        batch["winner_mask"][:] = False
    supervision = BatchSupervision.from_cpu(batch)
    batch = {name: value.to(device) for name, value in batch.items()}
    model = RiichiAnalysisModel(
        format_version=18,
        architecture=V18Architecture(
            width=128, blocks=1, feed_forward_width=256, decoder_width=256
        ),
    ).to(device)
    other = copy.deepcopy(model)
    first, losses, active, _ = multitask_loss(
        forward_batch(model, batch), batch, ordered_distance=ordered_distance
    )
    second, accelerated, availability, _ = multitask_loss(
        forward_batch(other, batch),
        batch,
        supervision=supervision,
        ordered_distance=ordered_distance,
    )
    assert active == availability
    for name in losses:
        torch.testing.assert_close(losses[name], accelerated[name], rtol=0, atol=0)
    first.backward()
    second.backward()
    for left, right in zip(model.parameters(), other.parameters(), strict=True):
        if left.grad is None:
            assert right.grad is None
        else:
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


def test_adamw_backend_reapplied_after_restore_without_resetting_moments():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer, mode = make_adamw([parameter], torch.device("cpu"))
    assert mode == "default"
    parameter.grad = torch.tensor([0.1, 0.2])
    optimizer.step()
    state = copy.deepcopy(optimizer.state_dict())
    resumed = torch.nn.Parameter(parameter.detach().clone())
    restored, _ = make_adamw([resumed], torch.device("cpu"), "foreach")
    restored.load_state_dict(state)
    set_adamw_execution(restored, "foreach")
    parameter.grad = torch.tensor([0.3, 0.4])
    resumed.grad = parameter.grad.clone()
    optimizer.step()
    restored.step()
    torch.testing.assert_close(parameter, resumed)
    assert restored.param_groups[0]["foreach"] is True
    assert restored.state[resumed]["step"] == 2
    torch.testing.assert_close(
        optimizer.state[parameter]["exp_avg"], restored.state[resumed]["exp_avg"]
    )


def test_fused_rejected_on_cpu():
    with pytest.raises(ValueError, match="requires CUDA"):
        make_adamw([torch.nn.Parameter(torch.ones(1))], torch.device("cpu"), "fused")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fused_cuda_optimizer_exact_resume():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], device="cuda"))
    optimizer, mode = make_adamw([parameter], torch.device("cuda"), "auto")
    assert mode in {"fused", "foreach"}
    parameter.grad = torch.tensor([0.1, 0.2], device="cuda")
    optimizer.step()
    resumed = torch.nn.Parameter(parameter.detach().clone())
    restored, _ = make_adamw([resumed], torch.device("cuda"), mode)
    restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    set_adamw_execution(restored, mode)
    parameter.grad = torch.tensor([0.3, 0.4], device="cuda")
    resumed.grad = parameter.grad.clone()
    optimizer.step()
    restored.step()
    torch.testing.assert_close(parameter, resumed, rtol=0, atol=0)
    for key, value in optimizer.state[parameter].items():
        torch.testing.assert_close(value, restored.state[resumed][key], rtol=0, atol=0)
