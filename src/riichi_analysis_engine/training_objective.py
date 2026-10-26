"""Versioned objectives and hash-bound changes at an exact unread cursor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .corpus_extension import file_sha256

PROBABILITY_PROFILE = "probability-v1"
ORDERED_PROFILE = "ordinal-cdf-v1"
PHASE_CURSOR = "single-pass-phase-v1"
ORDERED_CURSOR = "single-pass-ordinal-v1"
ORDERED_PHASE_CURSOR = "single-pass-phase-ordinal-v1"
ORDERED_TERMS = (
    "shanten",
    "hidden_allocation",
    "dora_distribution",
    "score_distribution",
    "placement",
)


def objective_definition(profile: str) -> dict[str, Any]:
    if profile not in {PROBABILITY_PROFILE, ORDERED_PROFILE}:
        raise RuntimeError("unsupported training objective profile")
    result: dict[str, Any] = {"type": "training-objective-v1", "profile": profile}
    if profile == ORDERED_PROFILE:
        result.update(
            distanceCoefficient=1.0,
            scoreUnitPoints=1000.0,
            ordinalUnit=1.0,
            terms=list(ORDERED_TERMS),
            reduction="original-active-label-and-frame-means-v1",
        )
    return result


def objective_cursor(profile: str, *, phase: bool) -> str:
    objective_definition(profile)
    if profile == ORDERED_PROFILE:
        return ORDERED_PHASE_CURSOR if phase else ORDERED_CURSOR
    return PHASE_CURSOR if phase else "single-pass-v1"


def _proof(reference: dict[str, Any], datasets: dict[str, Any]) -> dict[str, Any]:
    path = Path(reference["path"])
    if file_sha256(path) != reference["sha256"]:
        raise RuntimeError("objective transition proof changed")
    proof = json.loads(path.read_bytes())
    if proof.get("type") != "training-objective-transition-v1":
        raise RuntimeError("unsupported objective transition proof")
    for split in ("train", "validation"):
        if proof.get(split + "ManifestSha256") != datasets[split].get("manifestSha256"):
            raise RuntimeError("objective transition changed dataset identity")
    parent = proof.get("parentCheckpoint", {})
    if file_sha256(Path(parent["path"])) != parent.get("sha256"):
        raise RuntimeError("objective transition parent changed")
    if proof.get("fromObjective") != objective_definition(
        PROBABILITY_PROFILE
    ) or proof.get("toObjective") != objective_definition(ORDERED_PROFILE):
        raise RuntimeError("objective transition has a different loss definition")
    return proof


def saved_objective(checkpoint: dict[str, Any]) -> dict[str, Any]:
    objective = checkpoint.get(
        "lossObjective", objective_definition(PROBABILITY_PROFILE)
    )
    if not isinstance(objective, dict) or objective != objective_definition(
        objective.get("profile")
    ):
        raise RuntimeError("checkpoint objective definition is incompatible")
    cursor = checkpoint.get("trainingCursor")
    if isinstance(cursor, dict) and cursor.get("type") != objective_cursor(
        objective["profile"], phase=checkpoint.get("trainingPhase") is not None
    ):
        raise RuntimeError("checkpoint cursor and objective version disagree")
    transition = checkpoint.get("objectiveTransition")
    if transition is not None:
        if objective["profile"] != ORDERED_PROFILE or not isinstance(transition, dict):
            raise RuntimeError("invalid objective transition history")
        proof = _proof(transition["proof"], checkpoint["datasets"])
        parent = proof["parentCheckpoint"]
        if transition["parent"] != parent:
            raise RuntimeError("objective transition boundary history changed")
        if (
            parent["samples"] > checkpoint["samplesSeen"]
            or parent["step"] > checkpoint["step"]
        ):
            raise RuntimeError("objective transition lies beyond the unread cursor")
    return objective


def resolve_objective(
    requested: str | None,
    checkpoint: dict[str, Any] | None,
    checkpoint_path: Path | None,
    datasets: dict[str, Any],
    proof_path: Path | None,
    *,
    source_revision: str | None,
    validate_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    """Inherit by default; an explicit change never resets optimizer or counters."""
    if checkpoint is None:
        if proof_path is not None:
            raise RuntimeError("objective transition requires a resume checkpoint")
        return objective_definition(requested or PROBABILITY_PROFILE), None, False
    original = saved_objective(checkpoint)
    objective = objective_definition(requested or original["profile"])
    history = checkpoint.get("objectiveTransition")
    if objective == original:
        if proof_path is not None:
            raise RuntimeError("objective proof supplied without an objective change")
        return objective, history, False
    if validate_only or proof_path is None or checkpoint_path is None:
        raise RuntimeError(
            "changing objective requires an explicit unread-cursor transition proof"
        )
    if (
        original["profile"] != PROBABILITY_PROFILE
        or objective["profile"] != ORDERED_PROFILE
    ):
        raise RuntimeError("unsupported objective transition")
    reference = {"path": str(proof_path.resolve()), "sha256": file_sha256(proof_path)}
    proof = _proof(reference, datasets)
    parent = proof["parentCheckpoint"]
    cursor = checkpoint.get("trainingCursor", {})
    if (
        parent["sha256"] != file_sha256(checkpoint_path)
        or Path(parent["path"]).resolve() != checkpoint_path.resolve()
        or parent["samples"] != checkpoint["samplesSeen"]
        or parent["step"] != checkpoint["step"]
        or cursor.get("nextSample") != checkpoint["samplesSeen"]
        or cursor.get("batchesConsumed") != checkpoint["step"]
        or cursor.get("complete") is not False
        or proof.get("sourceRevision") != source_revision
        or proof.get("parentSourceRevision")
        != checkpoint.get("environment", {}).get("sourceRevision")
    ):
        raise RuntimeError(
            "objective proof does not bind the exact unread checkpoint/source"
        )
    for split in ("train", "validation"):
        if (
            checkpoint["datasets"][split]["manifestSha256"]
            != datasets[split]["manifestSha256"]
        ):
            raise RuntimeError("objective transition cannot change data")
    return objective, {"proof": reference, "parent": parent}, True
