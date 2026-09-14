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
        "0",
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
