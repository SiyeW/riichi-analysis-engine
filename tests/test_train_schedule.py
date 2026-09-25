import pytest
import torch

from riichi_analysis_engine.train import (
    gradient_total_norm,
    learning_rate_at,
    resolve_learning_rate_schedule,
    shared_gradient_geometry,
    tail_learning_rate_factor,
)


def test_gradient_norm_measurement_does_not_modify_gradients() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([3.0, 4.0])
    before = parameter.grad.clone()

    assert gradient_total_norm([parameter]) == pytest.approx(5.0)
    assert torch.equal(parameter.grad, before)


def test_shared_gradient_geometry_reports_orthogonal_and_opposed_tasks() -> None:
    shared = torch.tensor([[2.0, 3.0]], requires_grad=True)
    losses = {
        "right": shared[:, 0].sum(),
        "up": shared[:, 1].sum(),
        "left": -shared[:, 0].sum(),
        "inactive": shared.sum(),
    }
    metrics = shared_gradient_geometry(
        losses,
        {"right": True, "up": True, "left": True, "inactive": False},
        shared,
    )

    assert metrics["sharedGradientNorm/right"] == pytest.approx(1.0)
    assert metrics["sharedGradientCosine/right__up"] == pytest.approx(0.0)
    assert metrics["sharedGradientCosine/right__left"] == pytest.approx(-1.0)
    assert metrics["sharedGradientCosine/up__left"] == pytest.approx(0.0)
    assert metrics["sharedGradientConflictFraction"] == pytest.approx(1 / 3)
    assert not any("inactive" in name for name in metrics)


def test_the_default_shape_keeps_the_plateau_rate_constant() -> None:
    for step in (0, 1, 10_000):
        assert learning_rate_at(step, 0, 0, 5e-4, 2e-4) == pytest.approx(2e-4)


def test_warmup_reaches_the_peak_and_then_the_plateau() -> None:
    assert learning_rate_at(0, 100, 0, 5e-4, 2e-4) == 0.0
    assert learning_rate_at(50, 100, 0, 5e-4, 2e-4) == pytest.approx(2.5e-4)
    assert learning_rate_at(100, 100, 0, 5e-4, 2e-4) == pytest.approx(5e-4)
    assert learning_rate_at(101, 100, 0, 5e-4, 2e-4) == pytest.approx(2e-4)


def test_cooldown_runs_from_the_peak_down_to_the_plateau() -> None:
    assert learning_rate_at(150, 100, 100, 5e-4, 2e-4) == pytest.approx(3.5e-4)
    assert learning_rate_at(200, 100, 100, 5e-4, 2e-4) == pytest.approx(2e-4)
    assert learning_rate_at(201, 100, 100, 5e-4, 2e-4) == pytest.approx(2e-4)


def test_the_schedule_depends_only_on_the_step() -> None:
    # A resumed run continues at the rate the same step would have had.
    assert learning_rate_at(5_000, 2_000, 500, 4e-4, 2e-4) == learning_rate_at(
        5_000, 2_000, 500, 4e-4, 2e-4
    )


def test_tail_decay_uses_the_authoritative_sample_cursor() -> None:
    assert tail_learning_rate_factor(79, 100, 20, 0.1) == 1.0
    assert tail_learning_rate_factor(80, 100, 20, 0.1) == 1.0
    assert tail_learning_rate_factor(90, 100, 20, 0.1) == pytest.approx(0.55)
    assert tail_learning_rate_factor(100, 100, 20, 0.1) == pytest.approx(0.1)
    assert tail_learning_rate_factor(120, 100, 20, 0.1) == pytest.approx(0.1)


def test_disabled_tail_decay_does_not_change_existing_runs() -> None:
    assert tail_learning_rate_factor(100, 100, 0, 0.1) == 1.0


def test_learning_rate_transition_preserves_tail_and_records_exact_cursor() -> None:
    saved = {
        "type": "sample-tail-linear-v1",
        "warmupSteps": 0,
        "cooldownSteps": 0,
        "peakLearningRate": 1e-5,
        "learningRate": 1e-5,
        "lossBalanceLearningRate": 1e-3,
        "tailDecaySamples": 9_217_570,
        "tailLearningRateFactor": 0.1,
        "sampleLimit": 92_175_696,
    }
    requested = {**saved, "peakLearningRate": 2e-5, "learningRate": 2e-5}
    with pytest.raises(RuntimeError, match="different learning-rate schedule"):
        resolve_learning_rate_schedule(
            requested, saved, next_sample=12_500_016,
            allow_transition=False, validate_only=False,
        )
    transitioned = resolve_learning_rate_schedule(
        requested, saved, next_sample=12_500_016,
        allow_transition=True, validate_only=False,
    )
    assert transitioned["learningRatePhases"] == [
        {"startSample": 0, "learningRate": 1e-5},
        {"startSample": 12_500_016, "learningRate": 2e-5},
    ]
    assert transitioned["sampleLimit"] == saved["sampleLimit"]
    assert transitioned["tailDecaySamples"] == saved["tailDecaySamples"]
    assert resolve_learning_rate_schedule(
        requested, transitioned, next_sample=13_000_016,
        allow_transition=False, validate_only=False,
    ) == transitioned
    with pytest.raises(RuntimeError, match="another schedule field"):
        resolve_learning_rate_schedule(
            {**requested, "tailDecaySamples": 8_000_000}, saved,
            next_sample=12_500_016, allow_transition=True, validate_only=False,
        )
