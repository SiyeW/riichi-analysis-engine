import copy
import json
import sys

import pytest
import torch
from test_dataset import pack_directory
from test_training_phase import assert_nested_equal

from riichi_analysis_engine import export_weights, train
from riichi_analysis_engine.corpus_extension import file_sha256
from riichi_analysis_engine.train import resume_training_cursor
from riichi_analysis_engine.training_objective import (
    ORDERED_CURSOR,
    ORDERED_PROFILE,
    PROBABILITY_PROFILE,
    objective_definition,
    resolve_objective,
    saved_objective,
)


def transition_case(tmp_path):
    datasets = {
        "train": {"manifestSha256": "train"},
        "validation": {"manifestSha256": "holdout"},
    }
    parent_path = tmp_path / "parent.pt"
    parent_path.write_bytes(b"immutable optimizer checkpoint")
    parent = {
        "samplesSeen": 32,
        "step": 2,
        "datasets": datasets,
        "environment": {"sourceRevision": "old"},
        "trainingCursor": {
            "type": "single-pass-v1",
            "nextSample": 32,
            "batchesConsumed": 2,
            "batchSize": 16,
            "complete": False,
        },
    }
    proof = {
        "type": "training-objective-transition-v1",
        "parentCheckpoint": {
            "path": str(parent_path),
            "sha256": file_sha256(parent_path),
            "samples": 32,
            "step": 2,
        },
        "trainManifestSha256": "train",
        "validationManifestSha256": "holdout",
        "fromObjective": objective_definition(PROBABILITY_PROFILE),
        "toObjective": objective_definition(ORDERED_PROFILE),
        "sourceRevision": "new",
        "parentSourceRevision": "old",
    }
    proof_path = tmp_path / "objective-proof.json"
    proof_path.write_text(json.dumps(proof), encoding="utf-8")
    return datasets, parent_path, parent, proof, proof_path


def test_legacy_defaults_and_new_training_are_explicit():
    objective, history, changed = resolve_objective(
        None, None, None, {}, None, source_revision=None
    )
    assert (
        objective == objective_definition(PROBABILITY_PROFILE)
        and history is None
        and not changed
    )
    assert saved_objective({}) == objective
    assert (
        resolve_objective(ORDERED_PROFILE, None, None, {}, None, source_revision=None)[
            0
        ]["profile"]
        == ORDERED_PROFILE
    )


def test_exact_unread_transition_and_resume_keep_boundary(tmp_path):
    datasets, path, parent, _, proof_path = transition_case(tmp_path)
    objective, history, changed = resolve_objective(
        ORDERED_PROFILE, parent, path, datasets, proof_path, source_revision="new"
    )
    assert changed and history["parent"]["samples"] == 32
    assert parent["samplesSeen"] == 32 and "lossObjective" not in parent
    new = {
        **copy.deepcopy(parent),
        "lossObjective": objective,
        "objectiveTransition": history,
    }
    new["trainingCursor"]["type"] = ORDERED_CURSOR
    assert resume_training_cursor(new, datasets, 16) == (32, 2)
    assert resolve_objective(None, new, path, datasets, None, source_revision="new")[
        0:2
    ] == (objective, history)
    with pytest.raises(RuntimeError, match="transition"):
        resolve_objective(
            PROBABILITY_PROFILE, new, path, datasets, None, source_revision="new"
        )
    assert ORDERED_CURSOR not in {"single-pass-v1", "single-pass-phase-v1"}


@pytest.mark.parametrize(
    "case",
    [
        "proof_missing",
        "validation_only",
        "source",
        "cursor",
        "complete",
        "data",
        "parent_hash",
        "definition",
    ],
)
def test_transition_rejects_unbound_or_concurrent_changes(tmp_path, case):
    datasets, path, parent, proof, proof_path = transition_case(tmp_path)
    kwargs = {"source_revision": "new"}
    if case == "proof_missing":
        proof_path = None
    if case == "validation_only":
        kwargs["validate_only"] = True
    if case == "source":
        kwargs["source_revision"] = "other"
    if case == "cursor":
        parent["trainingCursor"]["nextSample"] = 16
    if case == "complete":
        parent["trainingCursor"]["complete"] = True
    if case == "data":
        datasets = copy.deepcopy(datasets)
        datasets["train"]["manifestSha256"] = "different"
    if case == "parent_hash":
        path.write_bytes(b"changed")
    if case == "definition":
        proof["toObjective"]["distanceCoefficient"] = 0.5
        proof_path.write_text(json.dumps(proof), encoding="utf-8")
    with pytest.raises(RuntimeError):
        resolve_objective(ORDERED_PROFILE, parent, path, datasets, proof_path, **kwargs)


def test_resuming_checks_objective_definition_cursor_and_proof_identity(tmp_path):
    datasets, path, parent, _, proof_path = transition_case(tmp_path)
    objective, history, _ = resolve_objective(
        ORDERED_PROFILE, parent, path, datasets, proof_path, source_revision="new"
    )
    new = {
        **copy.deepcopy(parent),
        "lossObjective": objective,
        "objectiveTransition": history,
    }
    with pytest.raises(RuntimeError, match="cursor"):
        saved_objective(new)
    new["trainingCursor"]["type"] = ORDERED_CURSOR
    assert saved_objective(new) == objective
    new["lossObjective"]["scoreUnitPoints"] = 100
    with pytest.raises(RuntimeError, match="definition"):
        saved_objective(new)
    new["lossObjective"] = objective_definition(ORDERED_PROFILE)
    proof_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="proof changed"):
        saved_objective(new)


def test_real_entry_transition_keeps_state_unread_suffix_and_inference_export(
    tmp_path, monkeypatch
):
    packs = pack_directory(
        tmp_path / "data",
        games=2,
        chunks=2,
        samples=8,
        training_targets=True,
        model_format=13,
        semantic_history=True,
    )
    common = [
        "train",
        "--train",
        str(packs),
        "--validation",
        str(packs),
        "--device",
        "cpu",
        "--batch-size",
        "8",
        "--model-format",
        "14",
        "--semantic-width",
        "16",
        "--semantic-stem-width",
        "24",
        "--semantic-event-width",
        "12",
        "--semantic-backbone-blocks",
        "2",
        "--semantic-event-blocks",
        "1",
        "--semantic-decoder-width",
        "20",
        "--semantic-attention-heads",
        "4",
        "--max-validation-samples",
        "8",
        "--checkpoint-every-samples",
        "0",
    ]
    monkeypatch.setattr(train, "open_dashboard", lambda _run: None)
    consumed = []
    forward = train.forward_batch

    def observe(model, batch):
        if model.training:
            consumed.extend(batch["sample_tag"].tolist())
        return forward(model, batch)

    monkeypatch.setattr(train, "forward_batch", observe)

    def run(name, extra):
        folder = tmp_path / name
        monkeypatch.setattr(sys, "argv", [*common, "--run", str(folder), *extra])
        train.main()
        return train.resolve_resume_path(folder)

    parent_path = run("parent", ["--max-steps", "1"])
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    proof = {
        "type": "training-objective-transition-v1",
        "parentCheckpoint": {
            "path": str(parent_path),
            "sha256": file_sha256(parent_path),
            "samples": 8,
            "step": 1,
        },
        "trainManifestSha256": parent["datasets"]["train"]["manifestSha256"],
        "validationManifestSha256": parent["datasets"]["validation"]["manifestSha256"],
        "fromObjective": objective_definition(PROBABILITY_PROFILE),
        "toObjective": objective_definition(ORDERED_PROFILE),
        "sourceRevision": parent["environment"]["sourceRevision"],
        "parentSourceRevision": parent["environment"]["sourceRevision"],
    }
    proof_path = tmp_path / "real-objective-proof.json"
    proof_path.write_text(json.dumps(proof), encoding="utf-8")
    with pytest.raises(RuntimeError, match="output directory must be empty"):
        run(
            "parent",
            [
                "--resume",
                str(parent_path),
                "--max-steps",
                "2",
                "--loss-profile",
                ORDERED_PROFILE,
                "--loss-transition-proof",
                str(proof_path),
            ],
        )
    middle = run(
        "changed",
        [
            "--resume",
            str(parent_path),
            "--max-steps",
            "2",
            "--loss-profile",
            ORDERED_PROFILE,
            "--loss-transition-proof",
            str(proof_path),
        ],
    )
    initial = torch.load(
        tmp_path / "changed/objective-initial.pt", map_location="cpu", weights_only=True
    )
    for name in ["model", "scaler", "lossBalancer"]:
        assert_nested_equal(initial[name], parent[name])
    assert_nested_equal(initial["optimizer"]["state"], parent["optimizer"]["state"])
    assert initial["samplesSeen"] == 8 and initial["step"] == 1
    assert initial["trainingCursor"]["type"] == ORDERED_CURSOR
    final_path = run("finished", ["--resume", str(middle)])
    final = torch.load(final_path, map_location="cpu", weights_only=True)
    expected = [
        int(tag)
        for batch in train.PackDataset(packs, batch_size=8)
        for tag in batch["sample_tag"]
    ]
    assert consumed == expected
    assert final["samplesSeen"] == 32 and final["step"] == 4
    assert final["trainingCursor"]["complete"]
    assert final["lossObjective"] == objective_definition(ORDERED_PROFILE)
    assert "lossComponent/distance/shanten" in final["validation"]
    output = tmp_path / "inference.pt"
    monkeypatch.setattr(sys, "argv", ["export", str(final_path), str(output)])
    export_weights.main()
    exported = torch.load(output, map_location="cpu", weights_only=True)
    assert_nested_equal(exported["model"], final["model"])
    assert exported["training"]["lossObjective"] == final["lossObjective"]
