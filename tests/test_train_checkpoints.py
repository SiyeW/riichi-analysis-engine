import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from riichi_analysis_engine.train import (
    _label_entropy,
    add_core_selection_metrics,
    capture_random_state,
    prune_numbered_checkpoints,
    restore_random_state,
    resolve_resume_path,
    source_metadata,
    write_dashboard,
    write_pointer,
)


class RecordingWriter:
    """Stands in for a TensorBoard writer and remembers what it was given."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, value, step))


def test_source_metadata_uses_recorded_revision_for_a_clean_source_archive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "f" * 40
    monkeypatch.setenv("RIICHI_ANALYSIS_SOURCE_REVISION", revision)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(128, "git")),
    )

    assert source_metadata() == {"sourceRevision": revision, "sourceDirty": False}


def test_source_metadata_rejects_an_invalid_recorded_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RIICHI_ANALYSIS_SOURCE_REVISION", "not-a-commit")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(128, "git")),
    )

    assert source_metadata() == {"sourceRevision": None, "sourceDirty": None}


def test_label_entropy_of_a_uniform_distribution_is_log_of_the_classes() -> None:
    assert _label_entropy(np.zeros(4, dtype=np.int64)) == 0.0
    assert _label_entropy(np.asarray([3, 3, 3, 3])) == pytest.approx(np.log(4))
    assert _label_entropy(np.asarray([4, 0, 0, 0])) == 0.0


def test_core_selection_metrics_require_both_losses() -> None:
    opponent_only = {"shanten": 1.0}
    add_core_selection_metrics(opponent_only, {"policy": 2.0, "shanten": 2.0})
    assert "Selection/core_score" not in opponent_only

    all_tasks = {"policy": 1.0, "shanten": 0.5}
    add_core_selection_metrics(all_tasks, {"policy": 2.0, "shanten": 1.0})
    assert all_tasks["Selection/core_score"] == pytest.approx(0.5)
    assert all_tasks["Selection/core_skill"] == pytest.approx(0.5)


def test_dashboard_keeps_distinct_metrics_with_the_same_value() -> None:
    writer = RecordingWriter()
    write_dashboard(
        writer,
        {
            "policy": 1.5,
            "shanten": 1.5,
            "metric/policyAccuracy": 0.5,
            "Selection/core_score": 0.9,
            "empty": float("nan"),
        },
        step=100,
    )

    tags = [tag for tag, _, _ in writer.scalars]
    assert tags == [
        "Validation/policy",
        "Validation/shanten",
        "Metrics/policyAccuracy",
        "Selection/core_score",
    ]
    assert all(step == 100 for _, _, step in writer.scalars)


def test_a_run_without_tensorboard_still_validates() -> None:
    write_dashboard(None, {"policy": 1.0}, step=1)


def test_random_state_round_trips_all_cpu_generators(tmp_path: Path) -> None:
    random.seed(17)
    np.random.seed(23)
    torch.manual_seed(29)
    state_path = tmp_path / "random-state.pt"
    torch.save(capture_random_state(), state_path)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    expected = (random.random(), np.random.random(), torch.rand(3))

    random.random()
    np.random.random()
    torch.rand(3)
    restore_random_state(state)

    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_pointer_only_moves_forward(tmp_path: Path) -> None:
    write_pointer(tmp_path, tmp_path / "ckpt-000000010.pt", step=10, samplesSeen=320)
    write_pointer(tmp_path, tmp_path / "ckpt-000000004.pt", step=4, samplesSeen=128)

    payload = json.loads((tmp_path / "latest_checkpoint.json").read_text(encoding="utf-8"))
    assert payload["step"] == 10
    assert payload["path"].endswith("ckpt-000000010.pt")


def test_rolling_checkpoints_are_pruned_but_named_ones_are_kept(tmp_path: Path) -> None:
    for step in (3, 6, 9, 12):
        (tmp_path / f"ckpt-{step:09d}.pt").write_bytes(b"x")
    (tmp_path / "best-core.pth").write_bytes(b"x")
    (tmp_path / "interrupted.pth").write_bytes(b"x")

    prune_numbered_checkpoints(tmp_path, keep=2)

    remaining = sorted(path.name for path in tmp_path.glob("*.pt")) + sorted(
        path.name for path in tmp_path.glob("*.pth")
    )
    assert remaining == ["ckpt-000000009.pt", "ckpt-000000012.pt", "best-core.pth", "interrupted.pth"]


def test_resume_resolves_a_pointer_a_rolling_file_or_a_named_one(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt-000000010.pt"
    checkpoint.write_bytes(b"x")
    write_pointer(tmp_path, checkpoint, step=10)
    assert resolve_resume_path(tmp_path) == checkpoint

    (tmp_path / "latest_checkpoint.json").unlink()
    (tmp_path / "ckpt-000000004.pt").write_bytes(b"x")
    assert resolve_resume_path(tmp_path).name == "ckpt-000000010.pt"

    for path in tmp_path.glob("ckpt-*.pt"):
        path.unlink()
    (tmp_path / "best-core.pth").write_bytes(b"x")
    assert resolve_resume_path(tmp_path).name == "best-core.pth"
    assert resolve_resume_path(checkpoint) == checkpoint if checkpoint.exists() else True

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        resolve_resume_path(empty)
