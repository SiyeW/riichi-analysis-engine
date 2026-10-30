import sys

import numpy as np
import pytest
import torch
from riichi_analysis_engine import export_weights, train
from riichi_analysis_engine.architecture import V18Architecture
from riichi_analysis_engine.architecture_transition import (
    ARCHITECTURE_CURSOR,
    base_cursor,
    restore_architecture_transition,
)
from riichi_analysis_engine.corpus_extension import file_sha256
from riichi_analysis_engine.grow_weights import publish_initialization
from riichi_analysis_engine.losses import LOSS_TERMS_V8, LearnedUncertaintyBalancer
from riichi_analysis_engine.model import RiichiAnalysisModel, count_parameters
from riichi_analysis_engine.prepare_architecture_checkpoint import prepare_checkpoint
from riichi_analysis_engine.training_objective import (
    ORDERED_PROFILE,
    objective_definition,
)
from test_training_phase import assert_nested_equal
from test_v17 import converted_fixture


def checkpoint_case(tmp_path):
    arch = V18Architecture(
        width=128,
        blocks=1,
        attention_heads=8,
        feed_forward_width=256,
        decoder_width=256,
    )
    model = RiichiAnalysisModel(format_version=18, architecture=arch)
    balancer = LearnedUncertaintyBalancer(LOSS_TERMS_V8)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": 1e-5, "weight_decay": 1e-4},
            {"params": balancer.parameters(), "lr": 1e-4, "weight_decay": 0},
        ]
    )
    next(model.parameters()).grad = torch.ones_like(next(model.parameters()))
    balancer.log_variance.grad = torch.ones_like(balancer.log_variance)
    optimizer.step()
    parent = tmp_path / "parent.pt"
    train.save_checkpoint(
        parent,
        model,
        optimizer,
        torch.amp.GradScaler("cuda", enabled=False),
        balancer,
        arch,
        step=1,
        samples_seen=16,
        analysis_samples_seen=8,
        batch_size=16,
        pass_complete=False,
        parameters=count_parameters(model),
        datasets={
            "train": {"manifestSha256": "train", "samples": 64},
            "validation": {"manifestSha256": "validation"},
        },
        environment={"sourceRevision": "parent"},
        validation=None,
        loss_objective=objective_definition(ORDERED_PROFILE),
        learning_rate_schedule={"type": "test-schedule"},
    )
    initial = tmp_path / "initializer.pt"
    target = V18Architecture(
        width=128,
        blocks=2,
        attention_heads=8,
        feed_forward_width=256,
        decoder_width=256,
    )
    publish_initialization(
        parent, initial, target, expected_parent_sha256=file_sha256(parent)
    )
    return parent, initial


def bootstrap(parent, initial, output):
    return prepare_checkpoint(
        initial,
        parent,
        output,
        initializer_sha256=file_sha256(initial),
        parent_sha256=file_sha256(parent),
        source_revision="new-source",
    )


def test_warmstart_resets_only_model_moments_and_keeps_exact_unread_history(tmp_path):
    parent_path, initial_path = checkpoint_case(tmp_path)
    before = file_sha256(parent_path)
    out = tmp_path / "warm.pt"
    receipt = bootstrap(parent_path, initial_path, out)
    parent = torch.load(parent_path, weights_only=True)
    initial = torch.load(initial_path, weights_only=True)
    value = torch.load(out, weights_only=True)
    assert file_sha256(parent_path) == before
    assert_nested_equal(value["model"], initial["model"])
    for name in (
        "lossBalancer",
        "scaler",
        "randomState",
        "datasets",
        "learningRateSchedule",
    ):
        assert_nested_equal(value[name], parent[name])
    assert value["step"] == 1 and value["samplesSeen"] == 16
    assert value["trainingCursor"]["type"] == ARCHITECTURE_CURSOR
    assert base_cursor(value)["type"] == parent["trainingCursor"]["type"]
    assert (
        value["modelGeneration"]["step"] == value["modelGeneration"]["samplesSeen"] == 0
    )
    new_balance_id = value["optimizer"]["param_groups"][1]["params"][0]
    old_balance_id = parent["optimizer"]["param_groups"][1]["params"][0]
    assert list(value["optimizer"]["state"]) == [new_balance_id]
    assert_nested_equal(
        value["optimizer"]["state"][new_balance_id],
        parent["optimizer"]["state"][old_balance_id],
    )
    assert receipt["newModelAdamStates"] == 0
    assert restore_architecture_transition(value) == value["architectureTransition"]
    assert train.resume_training_cursor(value, value["datasets"], 16) == (16, 1)
    with pytest.raises(FileExistsError):
        bootstrap(parent_path, initial_path, out)


@pytest.mark.parametrize(
    "changed", ["generation", "shape", "loss", "dataset", "schedule", "proof", "parent"]
)
def test_transition_is_fail_closed(tmp_path, changed):
    parent, initial = checkpoint_case(tmp_path)
    output = tmp_path / "warm.pt"
    bootstrap(parent, initial, output)
    value = torch.load(output, weights_only=True)
    if changed == "generation":
        value["modelGeneration"]["samplesSeen"] = 1
    elif changed == "shape":
        value["modelArchitecture"]["blocks"] = 9
    elif changed == "loss":
        value["lossObjective"]["distanceCoefficient"] = 2
    elif changed == "dataset":
        value["datasets"]["train"]["manifestSha256"] = "different"
    elif changed == "schedule":
        value["learningRateSchedule"]["type"] = "different"
    elif changed == "proof":
        (output.with_suffix(".pt.transition.json")).write_text("{}", encoding="utf-8")
    else:
        parent.write_bytes(b"changed-parent")
    with pytest.raises((RuntimeError, ValueError)):
        restore_architecture_transition(value)


def make_packs(tmp_path):
    from riichi_analysis_engine.packing import (
        MANIFEST_FORMAT,
        build_pack,
        pack_slots,
        plan_corpus,
        seam_game,
        staged_games,
        write_manifest,
    )
    from riichi_analysis_engine.storage import (
        V16_TRAINING_TARGET_SCHEMA_ID,
        save_chunk_archive,
    )

    converted = converted_fixture()
    stage, packs = tmp_path / "stage", tmp_path / "packs"
    for index in range(4):
        arrays = dict(converted.arrays)
        arrays["sample_tag"] = (
            np.arange(len(arrays["policy"]), dtype=np.int64) + index * 100
        )
        save_chunk_archive(
            stage / f"game-{index:06d}.zip",
            arrays,
            1,
            event_catalog=converted.event_catalog,
            training_target_schema=V16_TRAINING_TARGET_SCHEMA_ID,
        )
    paths = staged_games(stage)
    plan = plan_corpus(paths, 17)
    entries = [
        build_pack(paths, packs, plan, slot, 17, seam_game(plan, slot))[0]
        for slot in pack_slots(plan["length"], 32)
    ]
    write_manifest(
        packs / "manifest.json",
        {
            "format": MANIFEST_FORMAT,
            "seed": 17,
            "samples": int(plan["length"].sum()),
            "sourceGames": 4,
            "chunks": len(plan["length"]),
            "packs": entries,
            **{
                key: entries[0].get(key)
                for key in (
                    "modelInputSchema",
                    "observationChannels",
                    "eventMemorySchema",
                    "trainingTargetSchema",
                )
            },
        },
    )
    return packs


def test_actual_cli_stops_resumes_without_replay_and_exports_generation(
    tmp_path, monkeypatch
):
    packs = make_packs(tmp_path)
    monkeypatch.setattr(train, "open_dashboard", lambda _run: None)
    seen = []
    forward = train.forward_batch

    def observed(model, batch):
        if model.training:
            seen.extend(batch["sample_tag"].tolist())
        return forward(model, batch)

    monkeypatch.setattr(train, "forward_batch", observed)
    stop = tmp_path / "stop.request"
    common = [
        "train",
        "--train",
        str(packs),
        "--validation",
        str(packs),
        "--model-format",
        "18",
        "--device",
        "cpu",
        "--batch-size",
        "1",
        "--semantic-width",
        "128",
        "--semantic-attention-heads",
        "8",
        "--v15-feed-forward-width",
        "256",
        "--semantic-decoder-width",
        "256",
        "--learning-rate",
        "1e-5",
        "--loss-profile",
        ORDERED_PROFILE,
        "--max-validation-samples",
        "2",
        "--checkpoint-every-samples",
        "0",
        "--stop-file",
        str(stop),
    ]

    def run(name, blocks, parent=None, cap=0, interrupted=False):
        argv = [
            *common,
            "--run",
            str(tmp_path / name),
            "--v15-blocks",
            str(blocks),
            "--max-steps",
            str(cap),
        ]
        if parent:
            argv += ["--resume", str(parent)]
        monkeypatch.setattr(sys, "argv", argv)
        if interrupted:
            with pytest.raises(SystemExit) as error:
                train.main()
            assert error.value.code == 130
        else:
            train.main()
        return train.resolve_resume_path(tmp_path / name)

    parent = run("old", 1, cap=1)
    target = V18Architecture(
        width=128,
        blocks=2,
        attention_heads=8,
        feed_forward_width=256,
        decoder_width=256,
    )
    initial = tmp_path / "initializer.pt"
    publish_initialization(
        parent, initial, target, expected_parent_sha256=file_sha256(parent)
    )
    seed = tmp_path / "seed.pt"
    bootstrap(parent, initial, seed)
    stop.touch()
    paused = run("paused", 2, seed, interrupted=True)
    paused_value = torch.load(paused, weights_only=True)
    assert paused_value["samplesSeen"] == 1
    assert paused_value["modelGeneration"]["samplesSeen"] == 0
    stop.unlink()
    middle = run("gate", 2, paused, cap=3)
    middle_value = torch.load(middle, weights_only=True)
    assert middle_value["modelGeneration"]["samplesSeen"] == 2
    assert middle_value["modelGeneration"]["step"] == 2
    finished = run("finished", 2, middle)
    value = torch.load(finished, weights_only=True)
    expected = [
        int(tag)
        for batch in train.PackDataset(packs, batch_size=1)
        for tag in batch["sample_tag"]
    ]
    assert seen == expected
    assert value["trainingCursor"]["complete"]
    assert value["modelGeneration"]["samplesSeen"] == len(expected) - 1
    assert value["architectureTransition"] == middle_value["architectureTransition"]
    exported = tmp_path / "inference.pt"
    monkeypatch.setattr(sys, "argv", ["export", str(finished), str(exported)])
    export_weights.main()
    result = torch.load(exported, weights_only=True)
    assert_nested_equal(result["model"], value["model"])
    assert result["training"]["modelGeneration"] == value["modelGeneration"]
    assert "parentCheckpointSha256" in result["training"]["architectureTransfer"]


def test_architecture_cursor_cannot_drop_its_history():
    with pytest.raises(RuntimeError, match="no transition"):
        base_cursor({"trainingCursor": {"type": ARCHITECTURE_CURSOR}})
