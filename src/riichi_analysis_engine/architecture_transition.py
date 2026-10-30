"""Hash-bound shape changes within one unchanged single-pass data stream."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .corpus_extension import file_sha256

ARCHITECTURE_CURSOR = "single-pass-architecture-v1"
TRANSITION_TYPE = "architecture-warmstart-v1"
BASE_CURSORS = frozenset(
    {
        "single-pass-v1",
        "single-pass-phase-v1",
        "single-pass-ordinal-v1",
        "single-pass-phase-ordinal-v1",
    }
)


def generation_progress(
    transition: dict[str, Any] | None, samples: int, steps: int, analysis: int
) -> dict[str, int]:
    if transition is None:
        return {}
    result = {}
    for name, value, baseline in (
        ("generationSamples", samples, "samples"),
        ("generationStep", steps, "steps"),
        ("generationAnalysisSamples", analysis, "analysis"),
    ):
        start = transition["start"][baseline]
        if type(value) is not int or type(start) is not int or value < start:
            raise RuntimeError("architecture generation counters precede their parent")
        result[name] = value - start
    if result["generationAnalysisSamples"] > result["generationSamples"]:
        raise RuntimeError("architecture generation has excess analysis rows")
    return result


def transition_fields(
    transition: dict[str, Any] | None, samples: int, steps: int, analysis: int
) -> dict[str, Any]:
    if transition is None:
        return {}
    values = generation_progress(transition, samples, steps, analysis)
    return {
        "architectureTransition": transition,
        "modelGeneration": {
            "type": "architecture-generation-v1",
            "initializerSha256": transition["initializer"]["sha256"],
            "samplesSeen": values["generationSamples"],
            "step": values["generationStep"],
            "analysisSamplesSeen": values["generationAnalysisSamples"],
        },
    }


def transition_cursor(
    cursor: dict[str, Any], transition: dict[str, Any] | None
) -> dict[str, Any]:
    if transition is None:
        return cursor
    if cursor.get("type") not in BASE_CURSORS:
        raise RuntimeError("unsupported base cursor for architecture transfer")
    return {**cursor, "type": ARCHITECTURE_CURSOR, "baseType": cursor["type"]}


def base_cursor(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Expose the base convention only after checking the typed generation."""
    cursor = checkpoint.get("trainingCursor")
    if not isinstance(cursor, dict):
        return {}
    transition = checkpoint.get("architectureTransition")
    if transition is None:
        if cursor.get("type") == ARCHITECTURE_CURSOR or "modelGeneration" in checkpoint:
            raise RuntimeError("architecture cursor has no transition history")
        return cursor
    if (
        not isinstance(transition, dict)
        or transition.get("type") != TRANSITION_TYPE
        or cursor.get("type") != ARCHITECTURE_CURSOR
        or cursor.get("baseType") not in BASE_CURSORS
        or checkpoint.get("modelArchitecture") != transition.get("targetArchitecture")
    ):
        raise RuntimeError("architecture transition/cursor/shape disagree")
    expected = transition_fields(
        transition,
        checkpoint["samplesSeen"],
        checkpoint["step"],
        checkpoint["analysisSamplesSeen"],
    )["modelGeneration"]
    if checkpoint.get("modelGeneration") != expected:
        raise RuntimeError("architecture generation counters changed")
    return {**cursor, "type": cursor["baseType"]}


def restore_architecture_transition(
    checkpoint: dict[str, Any],
) -> dict[str, Any] | None:
    base_cursor(checkpoint)
    transition = checkpoint.get("architectureTransition")
    if transition is None:
        return None
    reference = transition.get("proof", {})
    path = Path(reference.get("path", ""))
    if not path.is_file() or file_sha256(path) != reference.get("sha256"):
        raise RuntimeError("architecture transition proof changed")
    proof = json.loads(path.read_bytes())
    if proof != {k: v for k, v in transition.items() if k != "proof"}:
        raise RuntimeError("architecture transition history differs from its proof")
    for key in ("initializer", "parent"):
        artifact = transition[key]
        source = Path(artifact["path"])
        if not source.is_file() or file_sha256(source) != artifact["sha256"]:
            raise RuntimeError("architecture transition artifact changed: " + key)
    for split in ("train", "validation"):
        if (
            checkpoint["datasets"][split]["manifestSha256"]
            != transition["manifests"][split]
        ):
            raise RuntimeError("architecture transition changed data: " + split)
    if checkpoint.get("lossObjective") != transition["lossObjective"]:
        raise RuntimeError("architecture transition changed the loss objective")
    if checkpoint.get("learningRateSchedule") != transition["learningRateSchedule"]:
        raise RuntimeError("architecture transition changed the learning-rate contract")
    return transition
