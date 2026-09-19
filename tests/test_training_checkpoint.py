import pytest
import torch

from riichi_analysis_engine.architecture import StructuredModelArchitecture
from riichi_analysis_engine.losses import LOSS_TERMS_V8, LearnedUncertaintyBalancer
from riichi_analysis_engine.model import RiichiAnalysisModel
from riichi_analysis_engine.train import (
    gradients_are_finite,
    resume_training_cursor,
    save_checkpoint,
    single_pass_window,
    step_budget_reached,
)


def test_gradient_finiteness_checks_every_materialized_gradient() -> None:
    finite = torch.nn.Parameter(torch.tensor([1.0]))
    unused = torch.nn.Parameter(torch.tensor([2.0]))
    finite.grad = torch.tensor([3.0])

    assert gradients_are_finite([finite, unused])

    finite.grad = torch.tensor([float("inf")])
    assert not gradients_are_finite([finite, unused])


def test_v8_checkpoint_records_architecture_and_learned_loss_state(tmp_path) -> None:
    architecture = StructuredModelArchitecture(
        shared_channels=16,
        shared_blocks=1,
        family_latent_width=32,
        opponent_latent_width=40,
        policy_latent_width=48,
        opponent_blocks=1,
        hidden_blocks=1,
        value_blocks=1,
        kyoku_blocks=1,
        match_blocks=1,
        policy_blocks=1,
        task_width=24,
        tile_width=8,
        policy_context_channels=16,
        policy_context_blocks=1,
        policy_context_width=16,
        policy_width=24,
    )
    model = RiichiAnalysisModel(format_version=8, architecture=architecture)
    balancer = LearnedUncertaintyBalancer(LOSS_TERMS_V8)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters()},
            {"params": balancer.parameters(), "weight_decay": 0.0},
        ]
    )
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    destination = tmp_path / "checkpoint.pt"

    save_checkpoint(
        destination,
        model,
        optimizer,
        scaler,
        balancer,
        architecture,
        step=10,
        samples_seen=20,
        analysis_samples_seen=12,
        batch_size=2,
        pass_complete=False,
        parameters={},
        datasets={},
        environment={},
        validation=None,
    )

    payload = torch.load(destination, map_location="cpu", weights_only=True)
    assert payload["format"] == "riichi-analysis-model-v8"
    assert payload["modelArchitecture"] == architecture.to_dict()
    assert payload["analysisSamplesSeen"] == 12
    assert payload["lossBalancer"]["terms"] == list(LOSS_TERMS_V8)
    assert payload["trainingCursor"] == {
        "type": "single-pass-v1",
        "nextSample": 20,
        "batchesConsumed": 10,
        "batchSize": 2,
        "complete": False,
    }
    assert "epoch" not in payload
    assert "batchInEpoch" not in payload


def test_step_budget_is_off_until_a_positive_limit_is_reached() -> None:
    assert not step_budget_reached(10, 0)
    assert not step_budget_reached(9, 10)
    assert step_budget_reached(10, 10)


def test_single_pass_budget_is_an_absolute_limit_after_resume() -> None:
    assert single_pass_window(100, 0, 20) == (20, 20)
    assert single_pass_window(100, 13, 20) == (20, 7)
    assert single_pass_window(100, 13, 0) == (100, 87)

    with pytest.raises(RuntimeError, match="increase --max-train-samples"):
        single_pass_window(100, 20, 20)


def test_resume_cursor_requires_the_same_manifests_and_exact_counters() -> None:
    datasets = {
        "train": {"manifestSha256": "train-a"},
        "validation": {"manifestSha256": "validation-a"},
    }
    checkpoint = {
        "step": 2,
        "samplesSeen": 13,
        "datasets": datasets,
        "trainingCursor": {
            "type": "single-pass-v1",
            "nextSample": 13,
            "batchesConsumed": 2,
            "batchSize": 8,
            "complete": False,
        },
    }

    assert resume_training_cursor(checkpoint, datasets, 8) == (13, 2)

    changed = {**datasets, "train": {"manifestSha256": "train-b"}}
    with pytest.raises(RuntimeError, match="different train manifest"):
        resume_training_cursor(checkpoint, changed, 8)

    checkpoint["trainingCursor"]["complete"] = True
    with pytest.raises(RuntimeError, match="already completed"):
        resume_training_cursor(checkpoint, datasets, 8)


def test_legacy_epoch_checkpoint_cannot_silently_resume() -> None:
    with pytest.raises(RuntimeError, match="predates the single-pass cursor"):
        resume_training_cursor(
            {"epoch": 0, "batchInEpoch": 10, "step": 10, "samplesSeen": 80},
            {
                "train": {"manifestSha256": "train"},
                "validation": {"manifestSha256": "validation"},
            },
            8,
        )
