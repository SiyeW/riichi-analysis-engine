"""Hash-bound, audited phase changes without discarding optimizer history.

An active cursor is local to one immutable manifest. Cumulative counters and the
parent checkpoint are explicit, rather than pretending that a new corpus is the
unfinished tail of the old one. External audit artifacts stay outside the repo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .corpus_extension import file_sha256
from .training_objective import ORDERED_PHASE_CURSOR, PHASE_CURSOR

PHASE_FORMAT = "audited-training-phase-v1"


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeError(f"invalid phase counter: {name}")
    return value


def phase_totals(
    phase: dict[str, Any] | None, samples: int, steps: int, analysis: int
) -> dict[str, int]:
    """Add immutable parent offsets; never infer updates from samples/batch."""
    offsets = phase["offsets"] if phase is not None else {}
    return {
        "cumulativeSamplesSeen": _integer(samples, "samples")
        + _integer(offsets.get("samples", 0), "sample offset"),
        "cumulativeStep": _integer(steps, "steps")
        + _integer(offsets.get("steps", 0), "step offset"),
        "cumulativeAnalysisSamplesSeen": _integer(analysis, "analysis")
        + _integer(offsets.get("analysis", 0), "analysis offset"),
    }


def phase_checkpoint_fields(
    phase: dict[str, Any] | None, samples: int, steps: int, analysis: int
) -> dict[str, Any]:
    if phase is None:
        return {}
    return {"trainingPhase": phase, **phase_totals(phase, samples, steps, analysis)}


def validate_phase_counters(checkpoint: dict[str, Any]) -> dict[str, int]:
    phase = checkpoint.get("trainingPhase")
    if not isinstance(phase, dict) or not isinstance(phase.get("offsets"), dict):
        raise TypeError("phase has no valid cumulative counter offsets")
    totals = phase_totals(
        phase,
        checkpoint["samplesSeen"],
        checkpoint["step"],
        checkpoint["analysisSamplesSeen"],
    )
    if any(checkpoint.get(key) != value for key, value in totals.items()):
        raise RuntimeError("phase cumulative counters do not match the active cursor")
    return totals


def _artifact(reference: object, *, binary: bool = False) -> Any:
    if not isinstance(reference, dict):
        raise TypeError("phase proof has an invalid artifact reference")
    path = Path(str(reference.get("path", "")))
    digest = reference.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or file_sha256(path) != digest:
        raise RuntimeError(f"phase proof artifact changed: {path}")
    return None if binary else json.loads(path.read_bytes())


def _evidence(
    proof_path: Path, datasets: dict[str, dict[str, object]], parent_train: str
) -> tuple[dict[str, Any], set[str]]:
    proof = json.loads(proof_path.read_bytes())
    if (
        proof.get("type") != PHASE_FORMAT
        or not isinstance(proof.get("phaseId"), str)
        or not proof["phaseId"]
    ):
        raise RuntimeError("unsupported training phase proof")
    train = datasets["train"]
    validation = datasets["validation"]
    if proof.get("trainManifestSha256") != train.get("manifestSha256") or proof.get(
        "validationManifestSha256"
    ) != validation.get("manifestSha256"):
        raise RuntimeError("phase proof uses different train/validation manifests")
    if train.get("manifestSha256") == parent_train:
        raise RuntimeError("a new phase cannot replay the parent manifest")
    _artifact(proof.get("parentCheckpoint"), binary=True)
    artifacts = proof.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TypeError("phase proof has no independent audit chain")
    audit = _artifact(artifacts.get("parentAudit"))
    provenance = _artifact(artifacts.get("provenance"))
    novelty = _artifact(artifacts.get("novelty"))
    storage = _artifact(artifacts.get("storage"))
    terminal = _artifact(artifacts.get("terminal"))
    hashes = audit.get("identity", {}).get("inputHashes", {})
    if (
        audit.get("status") != "complete"
        or hashes.get("train-packs") != parent_train
        or hashes.get("holdout-packs") != validation.get("manifestSha256")
    ):
        raise RuntimeError("phase audit does not cover the parent exposure/holdout")
    if provenance.get("parentAuditSha256") != artifacts["parentAudit"]["sha256"]:
        raise RuntimeError("phase provenance has a different parent audit")
    digest = train["manifestSha256"]
    samples = train["samples"]
    if (
        provenance.get("manifestSha256") != digest
        or provenance.get("trainingStarted") is not False
    ):
        raise RuntimeError("phase provenance does not certify this new dataset")
    if (
        novelty.get("status") != "complete"
        or novelty.get("verified") is not True
        or novelty.get("manifestSha256") != digest
        or novelty.get("samples") != samples
        or novelty.get("sourceIdentities") != train.get("sourceGames")
    ):
        raise RuntimeError("independent phase novelty audit is incomplete")
    for key in (
        "oldInputOverlaps",
        "duplicateInputs",
        "holdoutSourceIdIntersection",
        "holdoutPhysicalSourceIntersection",
    ):
        if novelty.get(key) != 0:
            raise RuntimeError(f"phase novelty boundary failed: {key}")
    if (
        novelty.get("stagePackedExactKeyEquality") is not True
        or novelty.get("stageKeyIndexSha256") != novelty.get("packedKeyIndexSha256")
        or not novelty.get("stageKeyIndexSha256")
    ):
        raise RuntimeError("phase row identities were not conserved")
    if (
        storage.get("verified") is not True
        or storage.get("samples") != samples
        or storage.get("adjacentSameGamePairs") != 0
        or storage.get("storage", {}).get("samples") != samples
    ):
        raise RuntimeError("full phase storage/order audit is incomplete")
    loader = storage.get("loader", {})
    if (
        loader.get("samples", 0) <= 0
        or loader.get("emptyBatches") != 0
        or loader.get("shortBatches") != 0
    ):
        raise RuntimeError("phase loader audit is incomplete")
    if (
        terminal.get("status") != "accepted"
        or terminal.get("manifestSha256") != digest
        or terminal.get("samples") != samples
        or terminal.get("canonicalKeyIndexSha256") != novelty["stageKeyIndexSha256"]
        or terminal.get("zeroResidualOwners") is not True
    ):
        raise RuntimeError("phase terminal acceptance is incomplete")
    return proof, {str(hashes["train-packs"])}


def restore_training_phase(
    checkpoint: dict[str, Any], datasets: dict[str, dict[str, object]]
) -> dict[str, Any] | None:
    phase = checkpoint.get("trainingPhase")
    if phase is None:
        if checkpoint.get("trainingCursor", {}).get("type") in {
            PHASE_CURSOR,
            ORDERED_PHASE_CURSOR,
        } or any(
            name in checkpoint
            for name in (
                "cumulativeSamplesSeen",
                "cumulativeStep",
                "cumulativeAnalysisSamplesSeen",
            )
        ):
            raise RuntimeError("phase cursor has no parent history")
        return None
    if (
        not isinstance(phase, dict)
        or phase.get("type") != PHASE_FORMAT
        or checkpoint.get("trainingCursor", {}).get("type")
        not in {PHASE_CURSOR, ORDERED_PHASE_CURSOR}
    ):
        raise RuntimeError("invalid training phase history/cursor")
    validate_phase_counters(checkpoint)
    reference = phase.get("proof")
    _artifact(reference)
    proof, covered = _evidence(
        Path(reference["path"]), datasets, phase["parent"]["trainManifestSha256"]
    )
    if (
        phase.get("phaseId") != proof["phaseId"]
        or phase["parent"]["checkpointSha256"] != proof["parentCheckpoint"]["sha256"]
    ):
        raise RuntimeError("phase parent checkpoint/identity changed")
    if proof.get("parentCounters") != {
        key: phase["parent"][key] for key in ("samples", "steps", "analysis")
    }:
        raise RuntimeError("phase parent counters changed")
    if set(phase.get("exposureManifests", [])) != covered | {
        str(datasets["train"]["manifestSha256"])
    }:
        raise RuntimeError("phase proof leaves historical exposure uncovered")
    if any(
        phase["offsets"][key] != phase["parent"][key]
        for key in ("samples", "steps", "analysis")
    ):
        raise RuntimeError("phase parent offsets changed")
    return phase


def begin_training_phase(
    proof_path: Path,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    datasets: dict[str, dict[str, object]],
    batch_size: int,
) -> dict[str, Any]:
    cursor = checkpoint.get("trainingCursor", {})
    parent_datasets = checkpoint.get("datasets", {})
    previous = restore_training_phase(checkpoint, parent_datasets)
    if (
        cursor.get("type") not in {"single-pass-v1", PHASE_CURSOR}
        or cursor.get("complete") is not True
    ):
        raise RuntimeError("new phase requires a completed exact parent cursor")
    samples = _integer(checkpoint.get("samplesSeen"), "parent samples")
    steps = _integer(checkpoint.get("step"), "parent steps")
    analysis = _integer(checkpoint.get("analysisSamplesSeen"), "parent analysis")
    if (
        samples <= 0
        or cursor.get("nextSample") != samples
        or cursor.get("batchesConsumed") != steps
        or analysis > samples
        or samples != parent_datasets["train"].get("samples")
    ):
        raise RuntimeError("parent pass counters are inconsistent or incomplete")
    if cursor.get("batchSize") != batch_size:
        raise RuntimeError("new phase changed the parent batch size")
    if checkpoint.get("resumeAllowed", True) is not True or any(
        not isinstance(checkpoint.get(name), dict)
        for name in ("optimizer", "scaler", "randomState", "lossBalancer")
    ):
        raise RuntimeError("new phase requires the full resumable training state")
    if (
        parent_datasets["validation"]["manifestSha256"]
        != datasets["validation"]["manifestSha256"]
    ):
        raise RuntimeError("new phase changed the validation manifest")
    for key in (
        "modelInputSchema",
        "observationChannels",
        "eventMemorySchema",
        "trainingTargetSchema",
    ):
        if parent_datasets["train"].get(key) != datasets["train"].get(key):
            raise RuntimeError(f"new phase changed the training contract: {key}")
    old_digest = str(parent_datasets["train"]["manifestSha256"])
    proof, covered = _evidence(proof_path, datasets, old_digest)
    if proof["parentCheckpoint"]["sha256"] != file_sha256(checkpoint_path):
        raise RuntimeError("phase proof starts from a different parent checkpoint")
    prior = set(previous["exposureManifests"]) if previous else {old_digest}
    if not prior.issubset(covered) or str(datasets["train"]["manifestSha256"]) in prior:
        raise RuntimeError("phase proof leaves historical exposure uncovered")
    totals = phase_totals(previous, samples, steps, analysis)
    parent = {
        "checkpointSha256": proof["parentCheckpoint"]["sha256"],
        "trainManifestSha256": old_digest,
        "samples": totals["cumulativeSamplesSeen"],
        "steps": totals["cumulativeStep"],
        "analysis": totals["cumulativeAnalysisSamplesSeen"],
        "sourceRevision": checkpoint.get("environment", {}).get("sourceRevision"),
    }
    if proof.get("parentCounters") != {
        key: parent[key] for key in ("samples", "steps", "analysis")
    }:
        raise RuntimeError("phase proof uses different parent counters")
    return {
        "type": PHASE_FORMAT,
        "phaseId": proof["phaseId"],
        "proof": {"path": str(proof_path.resolve()), "sha256": file_sha256(proof_path)},
        "parent": parent,
        "offsets": {key: parent[key] for key in ("samples", "steps", "analysis")},
        "exposureManifests": sorted(prior | {str(datasets["train"]["manifestSha256"])}),
        "parentLearningRateSchedule": checkpoint.get("learningRateSchedule"),
    }
