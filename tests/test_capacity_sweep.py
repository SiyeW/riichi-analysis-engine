from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "capacity_sweep.py"
SPEC = importlib.util.spec_from_file_location("capacity_sweep", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
capacity_sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capacity_sweep
SPEC.loader.exec_module(capacity_sweep)


def test_capacity_grid_is_strictly_increasing_and_keeps_completed_run_names() -> None:
    points = capacity_sweep.CAPACITY_V1
    assert [point.key for point in points] == [
        "0.94M",
        "1.15M",
        "1.42M",
        "2.01M",
        "2.80M",
        "3.81M",
        "5.03M",
        "6.45M",
        "8.25M",
        "10.26M",
        "12.67M",
        "15.47M",
        "18.76M",
        "22.32M",
        "26.61M",
    ]
    assert points[6].run_name == "v6-small-g1024-s5000-b32"
    assert points[8].run_name == "v6-medium-g1024-s5000-b32"
    assert points[10].run_name == "v6-default-g1024-s5000-b32"


def test_completed_validation_requires_the_controlled_step_budget(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\\n".join(
            [
                json.dumps({"phase": "validation", "step": 4999}),
                json.dumps(
                    {"phase": "validation", "step": 5000, "metric/policyAccuracy": 0.6}
                ),
            ]
        )
        + "\\n",
        encoding="utf-8",
    )
    assert capacity_sweep.completed_validation(tmp_path, 5000) == {
        "phase": "validation",
        "step": 5000,
        "metric/policyAccuracy": 0.6,
    }
    assert capacity_sweep.completed_validation(tmp_path, 6000) is None


def test_train_command_keeps_all_comparison_controls(tmp_path: Path) -> None:
    args = type(
        "Arguments",
        (),
        {
            "train": tmp_path / "train",
            "validation": tmp_path / "validation",
            "batch_size": 32,
            "shuffle_buffer_samples": 1024,
            "checkpoint_every": 1000,
            "device": "cuda",
            "seed": 20252026,
        },
    )()
    point = capacity_sweep.CAPACITY_V1[0]
    command = capacity_sweep.make_train_command(
        Path("python"), point, tmp_path / "run", args, max_steps=5000
    )
    assert command[command.index("--max-steps") + 1] == "5000"
    assert command[command.index("--batch-size") + 1] == "32"
    assert command[command.index("--shuffle-buffer-samples") + 1] == "1024"
    assert command[command.index("--seed") + 1] == "20252026"
    assert command[command.index("--analysis-channels") + 1] == "48"