import numpy as np
import pytest

from riichi_analysis_engine.kyoku_outcome import (
    OUTCOME_CLASSES,
    OUTCOME_COUNT,
    outcome_marginals,
    outcome_class_index,
)


def test_outcome_space_contains_all_mutually_exclusive_results() -> None:
    assert OUTCOME_COUNT == 33
    assert sum(outcome.kind == "draw" for outcome in OUTCOME_CLASSES) == 1
    assert sum(outcome.kind == "tsumo" for outcome in OUTCOME_CLASSES) == 4
    assert sum(outcome.kind == "ron" for outcome in OUTCOME_CLASSES) == 28


def test_double_ron_is_one_result_with_two_winners() -> None:
    index = outcome_class_index(
        np.asarray([0, 1, 1, 0], dtype=np.uint8),
        np.asarray([-1, 0, 0, -1], dtype=np.int8),
    )
    assert OUTCOME_CLASSES[index].kind == "ron"
    assert OUTCOME_CLASSES[index].winners == (1, 2)
    assert OUTCOME_CLASSES[index].target == 0


def test_marginals_are_derived_from_the_joint_distribution() -> None:
    probabilities = np.full(OUTCOME_COUNT, 1.0 / OUTCOME_COUNT, dtype=np.float32)

    draw, win, deal_in = outcome_marginals(probabilities)

    assert draw == pytest.approx(1 / 33)
    assert win.tolist() == pytest.approx([13 / 33] * 4)
    assert deal_in.tolist() == pytest.approx([7 / 33] * 4)
