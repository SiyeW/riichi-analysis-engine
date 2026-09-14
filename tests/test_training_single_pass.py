from __future__ import annotations

import json
import sys
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from test_dataset import pack_directory

from riichi_analysis_engine import train


@pytest.fixture()
def scratch() -> Iterator[Path]:
    root = Path(__file__).resolve().parents[1] / "runs" / "test-single-pass" / uuid.uuid4().hex
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
        "--max-validation-samples",
        "8",
        "--checkpoint-every-samples",
        str(checkpoint_every_samples),
        "--device",
        "cpu",
        "--analysis-channels",
        "16",
        "--analysis-blocks",
        "1",
        "--analysis-latent-width",
        "32",
        "--state-width",
        "24",
        "--future-width",
        "20",
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
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(train, "open_dashboard", lambda _run: None)
    train.main()
    pointer = train.resolve_resume_path(run)
    assert pointer.is_file()
    return pointer


def load_cursor(checkpoint: Path) -> dict[str, object]:
    return torch.load(checkpoint, map_location="cpu", weights_only=True)["trainingCursor"]


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
        pointer = json.loads((run / "latest_checkpoint.json").read_text(encoding="utf-8"))
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
    assert sorted(path.name for path in run.glob("ckpt-*.pt")) == [
        "ckpt-000000002.pt"
    ]
