from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch
from test_dataset import pack_directory
from test_training_single_pass import run_training

from riichi_analysis_engine import export_weights, train
from riichi_analysis_engine.corpus_extension import file_sha256
from riichi_analysis_engine.dataset import PackDataset
from riichi_analysis_engine.training_phase import (
    PHASE_CURSOR,
    PHASE_FORMAT,
    begin_training_phase,
    phase_checkpoint_fields,
    restore_training_phase,
)


def assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_nested_equal(a, b)
    else:
        assert actual == expected


def accepted_proof_fixture(
    root: Path, parent_path: Path, parent: dict, datasets: dict
) -> Path:
    """A synthetic trusted-auditor boundary, not a corpus novelty audit."""
    root.mkdir(parents=True, exist_ok=True)

    def write(name, value):
        path = root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return {"path": str(path), "sha256": file_sha256(path)}

    old = parent["datasets"]["train"]["manifestSha256"]
    val = datasets["validation"]["manifestSha256"]
    new = datasets["train"]["manifestSha256"]
    count = datasets["train"]["samples"]
    audit = write(
        "audit.json",
        {
            "status": "complete",
            "identity": {"inputHashes": {"train-packs": old, "holdout-packs": val}},
        },
    )
    provenance = write(
        "provenance.json",
        {
            "parentAuditSha256": audit["sha256"],
            "manifestSha256": new,
            "trainingStarted": False,
        },
    )
    novelty = write(
        "novelty.json",
        {
            "status": "complete",
            "verified": True,
            "manifestSha256": new,
            "samples": count,
            "sourceIdentities": datasets["train"]["sourceGames"],
            "oldInputOverlaps": 0,
            "duplicateInputs": 0,
            "holdoutSourceIdIntersection": 0,
            "holdoutPhysicalSourceIntersection": 0,
            "stagePackedExactKeyEquality": True,
            "stageKeyIndexSha256": "e" * 64,
            "packedKeyIndexSha256": "e" * 64,
        },
    )
    storage = write(
        "storage.json",
        {
            "verified": True,
            "samples": count,
            "adjacentSameGamePairs": 0,
            "storage": {"samples": count},
            "loader": {"samples": 8, "emptyBatches": 0, "shortBatches": 0},
        },
    )
    terminal = write(
        "terminal.json",
        {
            "status": "accepted",
            "manifestSha256": new,
            "samples": count,
            "canonicalKeyIndexSha256": "e" * 64,
            "zeroResidualOwners": True,
        },
    )
    proof = {
        "type": PHASE_FORMAT,
        "phaseId": "next-phase",
        "trainManifestSha256": new,
        "validationManifestSha256": val,
        "parentCheckpoint": {
            "path": str(parent_path),
            "sha256": file_sha256(parent_path),
        },
        "parentCounters": {
            "samples": parent.get("cumulativeSamplesSeen", parent["samplesSeen"]),
            "steps": parent.get("cumulativeStep", parent["step"]),
            "analysis": parent.get(
                "cumulativeAnalysisSamplesSeen", parent["analysisSamplesSeen"]
            ),
        },
        "artifacts": {
            "parentAudit": audit,
            "provenance": provenance,
            "novelty": novelty,
            "storage": storage,
            "terminal": terminal,
        },
    }
    return Path(write("proof.json", proof)["path"])


@pytest.fixture
def boundary(tmp_path):
    parent_file = tmp_path / "parent.pt"
    parent_file.write_bytes(b"synthetic-parent")
    old = {"manifestSha256": "a" * 64, "samples": 16, "sourceGames": 2}
    val = {"manifestSha256": "b" * 64}
    parent = {
        "step": 2,
        "samplesSeen": 16,
        "analysisSamplesSeen": 12,
        "trainingCursor": {
            "type": "single-pass-v1",
            "nextSample": 16,
            "batchesConsumed": 2,
            "complete": True,
            "batchSize": 8,
        },
        "datasets": {"train": old, "validation": val},
        "optimizer": {},
        "scaler": {},
        "randomState": {},
        "lossBalancer": {},
    }
    datasets = {
        "train": {**old, "manifestSha256": "c" * 64, "samples": 32},
        "validation": val,
    }
    proof = accepted_proof_fixture(tmp_path / "proof", parent_file, parent, datasets)
    return parent_file, parent, datasets, proof


def test_new_phase_keeps_parent_immutable_and_separates_local_totals(boundary):
    path, parent, datasets, proof = boundary
    before = copy.deepcopy(parent)
    phase = begin_training_phase(proof, path, parent, datasets, 8)
    assert parent == before
    fields = phase_checkpoint_fields(phase, 8, 1, 5)
    assert fields["cumulativeSamplesSeen"] == 24
    assert fields["cumulativeStep"] == 3
    assert fields["cumulativeAnalysisSamplesSeen"] == 17
    child = {
        **parent,
        "samplesSeen": 8,
        "step": 1,
        "analysisSamplesSeen": 5,
        "datasets": datasets,
        "trainingCursor": {"type": PHASE_CURSOR},
        **fields,
    }
    assert restore_training_phase(child, datasets) == phase
    child["cumulativeSamplesSeen"] += 1
    with pytest.raises(RuntimeError, match="cumulative counters"):
        restore_training_phase(child, datasets)


@pytest.mark.parametrize(
    "change,match",
    [
        ("incomplete", "completed exact"),
        ("batch", "batch size"),
        ("validation", "validation manifest"),
        ("schema", "training contract"),
        ("inference", "full resumable"),
        ("counters", "counters"),
    ],
)
def test_phase_entry_rejects_unproven_changes(boundary, change, match):
    path, parent, datasets, proof = boundary
    if change == "incomplete":
        parent["trainingCursor"]["complete"] = False
    if change == "batch":
        parent["trainingCursor"]["batchSize"] = 16
    if change == "validation":
        datasets["validation"] = {"manifestSha256": "d" * 64}
    if change == "schema":
        datasets["train"]["observationChannels"] = 99
    if change == "inference":
        parent["resumeAllowed"] = False
    if change == "counters":
        parent["samplesSeen"] = 15
    with pytest.raises(RuntimeError, match=match):
        begin_training_phase(proof, path, parent, datasets, 8)


def test_mutated_artifact_and_wrong_parent_are_rejected(boundary, tmp_path):
    path, parent, datasets, proof = boundary
    other = tmp_path / "other.pt"
    other.write_bytes(b"wrong-parent")
    with pytest.raises(RuntimeError, match="different parent checkpoint"):
        begin_training_phase(proof, other, parent, datasets, 8)
    (proof.parent / "novelty.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact changed"):
        begin_training_phase(proof, path, parent, datasets, 8)


def test_nonzero_exposure_even_with_refreshed_hash_is_rejected(boundary):
    path, parent, datasets, proof = boundary
    report = proof.parent / "novelty.json"
    value = json.loads(report.read_text())
    value["oldInputOverlaps"] = 1
    report.write_text(json.dumps(value), encoding="utf-8")
    document = json.loads(proof.read_text())
    document["artifacts"]["novelty"]["sha256"] = file_sha256(report)
    proof.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RuntimeError, match="oldInputOverlaps"):
        begin_training_phase(proof, path, parent, datasets, 8)


def test_same_manifest_cannot_be_renamed_as_a_new_phase(boundary):
    path, parent, datasets, proof = boundary
    datasets["train"]["manifestSha256"] = parent["datasets"]["train"]["manifestSha256"]
    document = json.loads(proof.read_text())
    document["trainManifestSha256"] = datasets["train"]["manifestSha256"]
    proof.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RuntimeError, match="replay the parent"):
        begin_training_phase(proof, path, parent, datasets, 8)


def test_missing_phase_history_cannot_silently_drop_cumulative_counters(boundary):
    _, parent, datasets, _ = boundary
    parent["cumulativeSamplesSeen"] = 20
    with pytest.raises(RuntimeError, match="no parent history"):
        restore_training_phase(parent, datasets)


def test_real_phase_resume_has_no_repeated_samples_and_keeps_export_totals(
    tmp_path, monkeypatch
):
    parent_packs = pack_directory(tmp_path / "parent-data", training_targets=True)
    parent_path = run_training(
        monkeypatch, parent_packs, tmp_path / "parent-run", max_samples=0
    )
    parent_hash = file_sha256(parent_path)
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    new_packs = pack_directory(
        tmp_path / "new-data", games=2, chunks=2, samples=8, training_targets=True
    )
    datasets = {
        "train": train.dataset_metadata(new_packs),
        "validation": train.dataset_metadata(parent_packs),
    }
    proof = accepted_proof_fixture(tmp_path / "proof", parent_path, parent, datasets)
    with pytest.raises(RuntimeError, match="output directory must be empty"):
        run_training(
            monkeypatch,
            new_packs,
            tmp_path / "parent-run",
            max_samples=0,
            resume=parent_path,
            start_new_phase=proof,
            validation_packs=parent_packs,
        )
    expected = [
        int(tag)
        for batch in PackDataset(new_packs, batch_size=8)
        for tag in batch["sample_tag"]
    ]
    consumed = []
    original = train.forward_batch
    original_optimizer = train.make_adamw
    original_restore = train.restore_random_state
    optimizers = []
    rng_calls = []

    def optimizer_spy(*args, **kwargs):
        result = original_optimizer(*args, **kwargs)
        optimizers.append(result[0])
        return result

    def rng_spy(state):
        if not rng_calls:
            assert_nested_equal(state, parent["randomState"])
        rng_calls.append(True)
        original_restore(state)

    def observe(model, batch):
        assert "source_game" not in batch
        if model.training:
            if not consumed:
                assert_nested_equal(model.state_dict(), parent["model"])
                assert_nested_equal(
                    optimizers[-1].state_dict()["state"], parent["optimizer"]["state"]
                )
            consumed.extend(batch["sample_tag"].tolist())
        return original(model, batch)

    monkeypatch.setattr(train, "forward_batch", observe)
    monkeypatch.setattr(train, "make_adamw", optimizer_spy)
    monkeypatch.setattr(train, "restore_random_state", rng_spy)
    first = run_training(
        monkeypatch,
        new_packs,
        tmp_path / "phase-a",
        max_samples=0,
        resume=parent_path,
        start_new_phase=proof,
        max_steps=2,
        warmup_steps=4,
        warmup_start_factor=0.1,
        validation_packs=parent_packs,
        validation_group_report=tmp_path / "groups.json",
    )
    saved = torch.load(first, map_location="cpu", weights_only=True)
    initialized = torch.load(
        tmp_path / "phase-a" / "phase-initial.pt", map_location="cpu", weights_only=True
    )
    assert (
        initialized["samplesSeen"] == 0
        and initialized["cumulativeSamplesSeen"] == parent["samplesSeen"]
    )
    assert_nested_equal(initialized["model"], parent["model"])
    assert_nested_equal(initialized["optimizer"]["state"], parent["optimizer"]["state"])
    assert_nested_equal(initialized["scaler"], parent["scaler"])
    assert saved["trainingCursor"]["type"] == PHASE_CURSOR
    assert (
        saved["samplesSeen"] == 16
        and saved["cumulativeSamplesSeen"] == parent["samplesSeen"] + 16
    )
    assert saved["optimizer"]["param_groups"][0]["lr"] == pytest.approx(
        2e-4 * (0.1 + 0.9 / 4)
    )
    assert saved["optimizer"]["param_groups"][1]["lr"] == pytest.approx(
        1e-3 * (0.1 + 0.9 / 4)
    )
    second = run_training(
        monkeypatch,
        new_packs,
        tmp_path / "phase-b",
        max_samples=0,
        resume=first,
        max_steps=4,
        warmup_steps=4,
        warmup_start_factor=0.1,
        validation_packs=parent_packs,
    )
    end = torch.load(second, map_location="cpu", weights_only=True)
    assert consumed == expected
    assert len(rng_calls) == 2
    assert (
        float(next(iter(end["optimizer"]["state"].values()))["step"])
        == float(next(iter(parent["optimizer"]["state"].values()))["step"]) + 4
    )
    pointer = json.loads((tmp_path / "phase-b" / "latest_checkpoint.json").read_text())
    assert pointer["cumulativeSamplesSeen"] == end["cumulativeSamplesSeen"]
    assert end["environment"]["resumeParent"]["checkpointSha256"] == file_sha256(first)
    assert end["cumulativeStep"] == parent["step"] + 4
    assert end["cumulativeAnalysisSamplesSeen"] == parent["analysisSamplesSeen"] + 32
    assert file_sha256(parent_path) == parent_hash
    assert (
        json.loads((tmp_path / "groups.json").read_text())["manifestSha256"]
        == datasets["validation"]["manifestSha256"]
    )
    destination = tmp_path / "inference.pt"
    monkeypatch.setattr(sys, "argv", ["export", str(second), str(destination)])
    export_weights.main()
    exported = torch.load(destination, map_location="cpu", weights_only=True)
    assert exported["training"]["samplesSeen"] == end["cumulativeSamplesSeen"]
    assert exported["training"]["phaseSamplesSeen"] == end["samplesSeen"]


def test_phase_stopped_before_first_update_can_resume_at_zero(tmp_path, monkeypatch):
    old = pack_directory(tmp_path / "old", training_targets=True)
    parent_path = run_training(monkeypatch, old, tmp_path / "parent", max_samples=0)
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    new = pack_directory(
        tmp_path / "new", games=2, chunks=2, samples=8, training_targets=True
    )
    datasets = {
        "train": train.dataset_metadata(new),
        "validation": train.dataset_metadata(old),
    }
    proof = accepted_proof_fixture(tmp_path / "proof", parent_path, parent, datasets)
    stop = tmp_path / "stop.request"
    stop.touch()
    with pytest.raises(SystemExit) as stopped:
        run_training(
            monkeypatch,
            new,
            tmp_path / "before-update",
            max_samples=0,
            resume=parent_path,
            start_new_phase=proof,
            validation_packs=old,
            stop_file=stop,
        )
    assert stopped.value.code == 130
    checkpoint_path = train.resolve_resume_path(tmp_path / "before-update")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["samplesSeen"] == 0
    assert checkpoint["cumulativeSamplesSeen"] == parent["samplesSeen"]
    stop.unlink()
    result = run_training(
        monkeypatch,
        new,
        tmp_path / "after-update",
        max_samples=0,
        resume=checkpoint_path,
        validation_packs=old,
        max_steps=1,
    )
    resumed = torch.load(result, map_location="cpu", weights_only=True)
    assert (
        resumed["samplesSeen"] == 8 and resumed["cumulativeStep"] == parent["step"] + 1
    )
