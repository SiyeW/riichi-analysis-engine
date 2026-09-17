from __future__ import annotations

import numpy as np
import torch

from riichi_analysis_engine.constants import MORTAL_OBS_CHANNELS, TILE_TYPES
from riichi_analysis_engine.model_input import (
    ANALYSIS_CHANNELS,
    MODEL_INPUT_CHANNELS,
    POLICY_CONTEXT_CHANNELS,
    compose_model_input,
    extract_policy_context,
    range_indices,
    split_model_input,
)


def test_policy_context_has_one_named_non_overlapping_layout() -> None:
    indices = range_indices()
    assert len(indices) == POLICY_CONTEXT_CHANNELS == 156
    assert len(set(indices)) == len(indices)
    assert 718 in indices and 722 in indices
    assert 860 in indices and 868 in indices
    assert 869 not in indices
    assert indices[-1] == MORTAL_OBS_CHANNELS - 1


def test_model_input_round_trip_preserves_both_logical_inputs() -> None:
    mortal = np.arange(MORTAL_OBS_CHANNELS * TILE_TYPES, dtype=np.float32).reshape(
        MORTAL_OBS_CHANNELS, TILE_TYPES
    )
    analysis = np.arange(ANALYSIS_CHANNELS * TILE_TYPES, dtype=np.float32).reshape(
        ANALYSIS_CHANNELS, TILE_TYPES
    )
    policy = extract_policy_context(mortal)
    combined = compose_model_input(analysis, policy)
    assert combined.shape == (MODEL_INPUT_CHANNELS, TILE_TYPES)

    actual_analysis, actual_policy = split_model_input(
        torch.from_numpy(combined).unsqueeze(0)
    )
    np.testing.assert_array_equal(actual_analysis[0].numpy(), analysis)
    np.testing.assert_array_equal(actual_policy[0].numpy(), policy)
