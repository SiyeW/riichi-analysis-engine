"""Versioned ownership boundary for the separated v9 model input."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from torch import Tensor

from .analysis_observation import PLANE_CHANNELS, PLANE_SCHEMA_ID
from .constants import MORTAL_OBS_CHANNELS, TILE_TYPES
from .rule_context import RULE_CONTEXT_CHANNELS, rule_context_metadata

MODEL_INPUT_SCHEMA_ID = "riichi-analysis-model-input-v1"
LEGACY_MODEL_INPUT_SCHEMA_ID = "riichi-analysis-model-input-legacy-v8"
ANALYSIS_SCHEMA_ID = PLANE_SCHEMA_ID
ANALYSIS_CHANNELS = PLANE_CHANNELS

# These are stable semantic ranges in libriichi observation v4.  They are kept
# here, rather than repeated at call sites, and are covered by layout tests.
MORTAL_DORA_SUMMARY = slice(718, 723)
MORTAL_SELF_WAIT_STATE = slice(860, 869)
MORTAL_DECISION_CONTEXT = slice(870, MORTAL_OBS_CHANNELS)
POLICY_CONTEXT_RANGES: tuple[slice, ...] = (
    MORTAL_DORA_SUMMARY,
    MORTAL_SELF_WAIT_STATE,
    MORTAL_DECISION_CONTEXT,
)
POLICY_CONTEXT_CHANNELS = sum(
    int(part.stop) - int(part.start) for part in POLICY_CONTEXT_RANGES
)
MODEL_INPUT_CHANNELS = ANALYSIS_CHANNELS + POLICY_CONTEXT_CHANNELS
POLICY_CONTEXT_START = ANALYSIS_CHANNELS
SHARED_MODEL_INPUT_SCHEMA_ID = "riichi-analysis-model-input-v2"
SHARED_MODEL_INPUT_CHANNELS = ANALYSIS_CHANNELS + RULE_CONTEXT_CHANNELS
RULE_CONTEXT_START = ANALYSIS_CHANNELS


def extract_policy_context(mortal_observation: np.ndarray) -> np.ndarray:
    source = np.asarray(mortal_observation, dtype=np.float32)
    if source.shape != (MORTAL_OBS_CHANNELS, TILE_TYPES):
        raise ValueError(f"wrong Mortal v4 observation shape: {source.shape}")
    result = np.concatenate([source[part] for part in POLICY_CONTEXT_RANGES], axis=0)
    if result.shape != (POLICY_CONTEXT_CHANNELS, TILE_TYPES):
        raise RuntimeError("policy-context extraction produced the wrong shape")
    return result


def compose_model_input(
    analysis_observation: np.ndarray, policy_context: np.ndarray
) -> np.ndarray:
    analysis = np.asarray(analysis_observation, dtype=np.float32)
    policy = np.asarray(policy_context, dtype=np.float32)
    if analysis.shape != (ANALYSIS_CHANNELS, TILE_TYPES):
        raise ValueError(f"wrong analysis observation shape: {analysis.shape}")
    if policy.shape != (POLICY_CONTEXT_CHANNELS, TILE_TYPES):
        raise ValueError(f"wrong policy-context shape: {policy.shape}")
    return np.concatenate((analysis, policy), axis=0)


def compose_shared_model_input(
    analysis_observation: np.ndarray, rule_context: np.ndarray
) -> np.ndarray:
    analysis = np.asarray(analysis_observation, dtype=np.float32)
    rules = np.asarray(rule_context, dtype=np.float32)
    if analysis.shape != (ANALYSIS_CHANNELS, TILE_TYPES):
        raise ValueError(f"wrong analysis observation shape: {analysis.shape}")
    if rules.shape != (RULE_CONTEXT_CHANNELS, TILE_TYPES):
        raise ValueError(f"wrong rule-context shape: {rules.shape}")
    return np.concatenate((analysis, rules), axis=0)


def split_model_input(value: Tensor) -> tuple[Tensor, Tensor]:
    if value.ndim != 3 or tuple(value.shape[1:]) != (
        MODEL_INPUT_CHANNELS,
        TILE_TYPES,
    ):
        raise ValueError(f"wrong v9 model input shape: {tuple(value.shape)}")
    return value[:, :POLICY_CONTEXT_START], value[:, POLICY_CONTEXT_START:]


def range_indices(parts: Sequence[slice] = POLICY_CONTEXT_RANGES) -> tuple[int, ...]:
    """Expose the selected Mortal channels for audit tests and metadata."""

    return tuple(
        index for part in parts for index in range(int(part.start), int(part.stop))
    )


def model_input_metadata() -> dict[str, object]:
    """Return the public, serializable v9 input contract."""

    return {
        "schema": MODEL_INPUT_SCHEMA_ID,
        "analysisSchema": ANALYSIS_SCHEMA_ID,
        "analysisChannels": ANALYSIS_CHANNELS,
        "policyContextSchema": "riichi-analysis-policy-context-v1",
        "policyContextChannels": POLICY_CONTEXT_CHANNELS,
        "channels": MODEL_INPUT_CHANNELS,
        "tileTypes": TILE_TYPES,
        "mortalObservationVersion": 4,
        "mortalPolicyContextRanges": [
            [int(part.start), int(part.stop)] for part in POLICY_CONTEXT_RANGES
        ],
    }


def shared_model_input_metadata() -> dict[str, object]:
    return {
        "schema": SHARED_MODEL_INPUT_SCHEMA_ID,
        "analysisSchema": ANALYSIS_SCHEMA_ID,
        "analysisChannels": ANALYSIS_CHANNELS,
        "ruleContext": rule_context_metadata(),
        "channels": SHARED_MODEL_INPUT_CHANNELS,
        "tileTypes": TILE_TYPES,
    }
