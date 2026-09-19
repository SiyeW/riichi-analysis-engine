from __future__ import annotations

import json
import shutil
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from test_dataset import pack_directory

from riichi_analysis_engine import train
from riichi_analysis_engine.model_input import LEGACY_MODEL_INPUT_SCHEMA_ID


@pytest.fixture()
def scratch() -> Iterator[Path]:
    root = (
        Path(__file__).resolve().parents[1]
        / "runs"
        / "test-single-pass"
        / uuid.uuid4().hex
    )
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run_training(
    monkeypatch: pytest.MonkeyPatch,
    packs: Path,
    run: Path,
    *,
    max_samples: int,
    resume: Path | None = None,
    checkpoint_every_samples: int = 0,
    validate_only: bool = False,
    model_format: int = 8,
    max_analysis_samples: int = 0,
    tail_decay_samples: int = 0,
    tail_learning_rate_factor: float = 0.1,
) -> Path:
    arguments = [
        "riichi-analysis-train",
        "--train",
        str(packs),
        "--validation",
        str(packs),
        "--run",
        str(run),
        "--batch-size",
        "8",
        "--max-train-samples",
        str(max_samples),
        "--max-analysis-samples",
        str(max_analysis_samples),
        "--max-validation-samples",
        "8",
        "--checkpoint-every-samples",
        str(checkpoint_every_samples),
        "--tail-decay-samples",
        str(tail_decay_samples),
        "--tail-learning-rate-factor",
        str(tail_learning_rate_factor),
        "--device",
        "cpu",
        "--model-format",
        str(model_format),
        "--shared-channels",
        "16",
        "--shared-blocks",
        "1",
        "--family-latent-width",
        "32",
        "--opponent-blocks",
        "1",
        "--hidden-blocks",
        "1",
        "--value-blocks",
        "1",
        "--kyoku-blocks",
        "1",
        "--match-blocks",
        "1",
        "--policy-blocks",
        "1",
        "--task-width",
        "24",
        "--tile-width",
        "8",
        "--policy-context-channels",
        "16",
        "--policy-context-blocks",
        "1",
        "--policy-context-width",
        "16",
        "--policy-width",
        "24",
    ]
    if resume is not None:
        arguments.extend(["--resume", str(resume)])
    if validate_only:
        arguments.append("--validate-only")
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(train, "open_dashboard", lambda _run: None)
    train.main()
    pointer = train.resolve_resume_path(run)
    assert pointer.is_file()
    return pointer


def load_cursor(checkpoint: Path) -> dict[str, object]:
    return torch.load(checkpoint, map_location="cpu", weights_only=True)[
        "trainingCursor"
    ]


def test_v9_training_smoke_uses_the_versioned_input_contract(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packs = pack_directory(
        scratch, games=2, chunks=1, samples=8, training_targets=True, model_format=9
    )
    checkpoint_path = run_training(
        monkeypatch,
        packs,
        scratch / "v9-run",
        max_samples=8,
        model_format=9,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["format"] == "riichi-analysis-model-v9"
    assert checkpoint["modelInput"]["schema"] == "riichi-analysis-model-input-v1"
    assert checkpoint["trainingCursor"]["nextSample"] == 8
    assert checkpoint["analysisSamplesSeen"] == 8


def test_v10_training_smoke_requires_canonical_analysis_rows(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packs = pack_directory(
        scratch, games=2, chunks=1, samples=8, training_targets=True, model_format=10
    )
    checkpoint_path = run_training(
        monkeypatch,
        packs,
        scratch / "v10-run",
        max_samples=8,
        model_format=10,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["format"] == "riichi-analysis-model-v10"
    assert checkpoint["modelInput"]["schema"] == "riichi-analysis-model-input-v1"
    assert checkpoint["trainingCursor"]["nextSample"] == 8
    assert checkpoint["analysisSamplesSeen"] == 8


def test_v10_can_stop_on_a_canonical_analysis_sample_budget(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packs = pack_directory(scratch, training_targets=True, model_format=10)
    checkpoint_path = run_training(
        monkeypatch,
        packs,
        scratch / "v10-analysis-budget",
        max_samples=0,
        max_analysis_samples=13,
        model_format=10,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["samplesSeen"] == 16
    assert checkpoint["analysisSamplesSeen"] == 16
    assert checkpoint["trainingCursor"]["complete"] is False
    metrics = [
        json.loads(line)
        for line in (scratch / "v10-analysis-budget" / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert metrics[-1]["stopReason"] == "max-analysis-samples"


def test_v9_training_rejects_legacy_packs_before_reading_batches(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packs = pack_directory(scratch, training_targets=True, model_format=8)

    with pytest.raises(RuntimeError, match="different model-input contract"):
        run_training(
            monkeypatch,
            packs,
            scratch / "wrong-input-run",
            max_samples=8,
            model_format=9,
        )

    manifest = json.loads((packs / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["modelInputSchema"] == LEGACY_MODEL_INPUT_SCHEMA_ID


def test_training_resumes_forward_and_completes_one_pass(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packs = pack_directory(scratch, training_targets=True)
    run = scratch / "run"

    first = run_training(monkeypatch, packs, run, max_samples=13)
    assert load_cursor(first) == {
        "type": "single-pass-v1",
        "nextSample": 13,
        "batchesConsumed": 2,
        "batchSize": 8,
        "complete": False,
    }

    second = run_training(monkeypatch, packs, run, max_samples=29, resume=first)
    assert load_cursor(second) == {
        "type": "single-pass-v1",
        "nextSample": 29,
        "batchesConsumed": 4,
        "batchSize": 8,
        "complete": False,
    }

    complete = run_training(monkeypatch, packs, run, max_samples=0, resume=second)
    assert load_cursor(complete) == {
        "type": "single-pass-v1",
        "nextSample": 128,
        "batchesConsumed": 17,
        "batchSize": 8,
        "complete": True,
    }

    with pytest.raises(RuntimeError, match="already completed"):
        run_training(monkeypatch, packs, run, max_samples=0, resume=complete)


def test_training_checkpoint_is_durable_before_validation(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packs = pack_directory(scratch, training_targets=True)
    run = scratch / "run"

    def interrupt_validation(*_args: object, **_kwargs: object) -> dict[str, float]:
        pointer = json.loads(
            (run / "latest_checkpoint.json").read_text(encoding="utf-8")
        )
        assert pointer["samplesSeen"] == 16
        assert pointer["validationComplete"] is False
        checkpoint = torch.load(pointer["path"], map_location="cpu", weights_only=True)
        assert checkpoint["trainingCursor"]["nextSample"] == 16
        assert checkpoint["validation"] is None
        raise train.TrainingInterrupted

    monkeypatch.setattr(train, "validate", interrupt_validation)
    with pytest.raises(SystemExit) as stopped:
        run_training(monkeypatch, packs, run, max_samples=16)

    assert stopped.value.code == 130
    pointer = json.loads((run / "latest_checkpoint.json").read_text(encoding="utf-8"))
    assert pointer["validationComplete"] is False


def test_validation_can_resume_without_replaying_training_samples(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packs = pack_directory(scratch, training_targets=True)
    interrupted_run = scratch / "interrupted"
    validation_run = scratch / "validation"
    original_validate = train.validate

    def interrupt_validation(*_args: object, **_kwargs: object) -> dict[str, float]:
        raise train.TrainingInterrupted

    monkeypatch.setattr(train, "validate", interrupt_validation)
    with pytest.raises(SystemExit) as stopped:
        run_training(monkeypatch, packs, interrupted_run, max_samples=16)
    assert stopped.value.code == 130

    checkpoint = train.resolve_resume_path(interrupted_run)
    cursor_before = load_cursor(checkpoint)
    monkeypatch.setattr(train, "validate", original_validate)
    validated = run_training(
        monkeypatch,
        packs,
        validation_run,
        max_samples=16,
        resume=checkpoint,
        validate_only=True,
    )

    assert load_cursor(validated) == cursor_before
    pointer = json.loads(
        (validation_run / "latest_checkpoint.json").read_text(encoding="utf-8")
    )
    assert pointer["validationComplete"] is True
    records = [
        json.loads(line)
        for line in (validation_run / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["phase"] for record in records] == ["validation"]
    assert records[0]["stopReason"] == "validation-only"


def test_interrupt_saves_the_next_unread_sample_and_can_resume(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packs = pack_directory(scratch, training_targets=True)
    run = scratch / "run"
    installed_handler: dict[str, object] = {}
    original_loss = train.multitask_loss
    interrupted = False

    def capture_handler(_signal: int, handler: object) -> object:
        installed_handler["handler"] = handler
        return handler

    def interrupt_after_first_loss(*args: object, **kwargs: object):
        nonlocal interrupted
        result = original_loss(*args, **kwargs)
        if not interrupted:
            interrupted = True
            handler = installed_handler["handler"]
            assert callable(handler)
            handler()
        return result

    monkeypatch.setattr(train.signal, "signal", capture_handler)
    monkeypatch.setattr(train, "multitask_loss", interrupt_after_first_loss)
    with pytest.raises(SystemExit) as stopped:
        run_training(monkeypatch, packs, run, max_samples=32)

    assert stopped.value.code == 130
    interrupted_checkpoint = train.resolve_resume_path(run)
    assert interrupted_checkpoint.name == "interrupted.pth"
    assert load_cursor(interrupted_checkpoint) == {
        "type": "single-pass-v1",
        "nextSample": 8,
        "batchesConsumed": 1,
        "batchSize": 8,
        "complete": False,
    }

    monkeypatch.setattr(train, "multitask_loss", original_loss)
    resumed = run_training(
        monkeypatch,
        packs,
        run,
        max_samples=32,
        resume=interrupted_checkpoint,
    )
    assert load_cursor(resumed)["nextSample"] == 32


def test_rolling_checkpoints_follow_sample_thresholds_not_step_numbers(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packs = pack_directory(scratch, training_targets=True)
    run = scratch / "run"

    run_training(
        monkeypatch,
        packs,
        run,
        max_samples=29,
        checkpoint_every_samples=10,
    )

    rolling = sorted(path.name for path in run.glob("ckpt-*.pt"))
    assert rolling == ["ckpt-000000002.pt", "ckpt-000000003.pt"]
    assert load_cursor(run / rolling[0])["nextSample"] == 16
    assert load_cursor(run / rolling[1])["nextSample"] == 24


def test_terminal_sample_boundary_is_not_saved_twice(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packs = pack_directory(scratch, training_targets=True)
    run = scratch / "run"

    final_checkpoint = run_training(
        monkeypatch,
        packs,
        run,
        max_samples=32,
        checkpoint_every_samples=16,
    )

    assert final_checkpoint.name == "checkpoint-step-4.pt"
    assert sorted(path.name for path in run.glob("ckpt-*.pt")) == ["ckpt-000000002.pt"]


def test_tail_decay_is_applied_and_recorded_as_a_resume_contract(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packs = pack_directory(scratch, training_targets=True)
    checkpoint_path = run_training(
        monkeypatch,
        packs,
        scratch / "tail-run",
        max_samples=16,
        tail_decay_samples=8,
        tail_learning_rate_factor=0.5,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["learningRateSchedule"]["sampleLimit"] == 16
    assert checkpoint["learningRateSchedule"]["tailDecaySamples"] == 8
    assert checkpoint["optimizer"]["param_groups"][0]["lr"] == pytest.approx(1e-4)
    assert checkpoint["optimizer"]["param_groups"][1]["lr"] == pytest.approx(5e-4)
