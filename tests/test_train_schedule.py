import pytest

from riichi_analysis_engine.train import learning_rate_at


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
