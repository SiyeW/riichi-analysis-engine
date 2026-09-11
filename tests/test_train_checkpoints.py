import json
from pathlib import Path

import numpy as np
import pytest

from riichi_analysis_engine.train import (
    _label_entropy,
    prune_numbered_checkpoints,
    resolve_resume_path,
    write_dashboard,
    write_pointer,
)


class RecordingWriter:
    """Stands in for a TensorBoard writer and remembers what it was given."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, value, step))


def test_label_entropy_of_a_uniform_distribution_is_log_of_the_classes() -> None:
    assert _label_entropy(np.zeros(4, dtype=np.int64)) == 0.0
    assert _label_entropy(np.asarray([3, 3, 3, 3])) == pytest.approx(np.log(4))
    assert _label_entropy(np.asarray([4, 0, 0, 0])) == 0.0


def test_dashboard_drops_values_that_repeat_or_do_not_apply() -> None:
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
    assert tags == ["Validation/policy", "Metrics/policyAccuracy", "Selection/core_score"]
    assert all(step == 100 for _, _, step in writer.scalars)


def test_a_run_without_tensorboard_still_validates() -> None:
    write_dashboard(None, {"policy": 1.0}, step=1)


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
