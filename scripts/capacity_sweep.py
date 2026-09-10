"""Run a resumable, controlled model-capacity comparison.

This deliberately lives outside the training package: it orchestrates several
ordinary ``riichi_analysis_engine.train`` processes without becoming another
training implementation. The data locations are supplied at launch and never
need to be written into the repository.

Each capacity point uses the same data, seed, batch size, optimization settings
and training-step budget. A one-step preflight executes the complete loss and
optimizer path first, so a GPU memory ceiling is recorded instead of silently
turning a larger architecture into a different experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


SWEEP_FORMAT = "riichi-analysis-capacity-sweep-v1"


@dataclass(frozen=True)
class CapacityPoint:
    """One architecture in the monotonic capacity exploration."""

    key: str
    run_name: str
    analysis_channels: int
    analysis_blocks: int
    analysis_latent_width: int
    state_width: int
    future_width: int
    policy_context_channels: int
    policy_context_blocks: int
    policy_context_width: int
    policy_width: int

    def train_arguments(self) -> list[str]:
        return [
            "--analysis-channels",
            str(self.analysis_channels),
            "--analysis-blocks",
            str(self.analysis_blocks),
            "--analysis-latent-width",
            str(self.analysis_latent_width),
            "--state-width",
            str(self.state_width),
            "--future-width",
            str(self.future_width),
            "--policy-context-channels",
            str(self.policy_context_channels),
            "--policy-context-blocks",
            str(self.policy_context_blocks),
            "--policy-context-width",
            str(self.policy_context_width),
            "--policy-width",
            str(self.policy_width),
        ]


# The three existing five-thousand-step runs retain their original directory
# names, allowing a resumed sweep to include them without rerunning them.
# The points are deliberately denser near the small-model boundary, then widen
# progressively while the 3 GB GPU determines the upper usable boundary.
CAPACITY_V1: tuple[CapacityPoint, ...] = (
    CapacityPoint("0.94M", "capacity-0p94m-g1024-s5000-b32", 48, 9, 192, 192, 160, 24, 1, 112, 160),
    CapacityPoint("1.15M", "capacity-1p15m-g1024-s5000-b32", 56, 10, 224, 224, 192, 28, 1, 120, 192),
    CapacityPoint("1.42M", "capacity-1p42m-g1024-s5000-b32", 64, 12, 256, 256, 224, 32, 2, 128, 224),
    CapacityPoint("2.01M", "capacity-2p01m-g1024-s5000-b32", 80, 15, 320, 320, 256, 40, 2, 144, 256),
    CapacityPoint("2.80M", "capacity-2p80m-g1024-s5000-b32", 96, 18, 384, 384, 320, 48, 2, 160, 320),
    CapacityPoint("3.81M", "capacity-3p81m-g1024-s5000-b32", 112, 21, 448, 448, 384, 56, 3, 176, 384),
    CapacityPoint("5.03M", "v6-small-g1024-s5000-b32", 128, 24, 512, 512, 448, 64, 3, 192, 448),
    CapacityPoint("6.45M", "capacity-6p45m-g1024-s5000-b32", 144, 27, 576, 576, 480, 72, 3, 208, 480),
    CapacityPoint("8.25M", "v6-medium-g1024-s5000-b32", 160, 30, 640, 640, 544, 80, 4, 224, 544),
    CapacityPoint("10.26M", "capacity-10p26m-g1024-s5000-b32", 176, 33, 704, 704, 576, 88, 4, 240, 576),
    CapacityPoint("12.67M", "v6-default-g1024-s5000-b32", 192, 36, 768, 768, 640, 96, 4, 256, 640),
    CapacityPoint("15.47M", "capacity-15p47m-g1024-s5000-b32", 208, 39, 832, 832, 704, 104, 4, 288, 704),
    CapacityPoint("18.76M", "capacity-18p76m-g1024-s5000-b32", 224, 42, 896, 896, 768, 112, 5, 320, 768),
    CapacityPoint("22.32M", "capacity-22p32m-g1024-s5000-b32", 240, 45, 960, 960, 800, 120, 5, 352, 800),
    CapacityPoint("26.61M", "capacity-26p61m-g1024-s5000-b32", 256, 48, 1024, 1024, 896, 128, 6, 384, 896),
)


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.append(json.loads(line))
    return result


def completed_validation(run_directory: Path, max_steps: int) -> dict[str, Any] | None:
    """Return the final comparable validation record, if one is complete."""

    for record in reversed(read_json_lines(run_directory / "metrics.jsonl")):
        if record.get("phase") == "validation" and record.get("step") == max_steps:
            return record
    return None


def read_config(run_directory: Path) -> dict[str, Any]:
    path = run_directory / "config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def peak_memory_mib(run_directory: Path) -> float | None:
    values = [
        float(record["peakAllocatedMiB"])
        for record in read_json_lines(run_directory / "metrics.jsonl")
        if record.get("phase") == "train" and "peakAllocatedMiB" in record
    ]
    return max(values) if values else None


def make_train_command(
    python: Path,
    point: CapacityPoint,
    run_directory: Path,
    args: argparse.Namespace,
    *,
    max_steps: int,
    max_validation_samples: int = 0,
) -> list[str]:
    command = [
        str(python),
        "-m",
        "riichi_analysis_engine.train",
        "--train",
        str(args.train),
        "--validation",
        str(args.validation),
        "--run",
        str(run_directory),
        "--epochs",
        "1",
        "--batch-size",
        str(args.batch_size),
        "--max-steps",
        str(max_steps),
        "--shuffle-buffer-samples",
        str(args.shuffle_buffer_samples),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        *point.train_arguments(),
    ]
    if max_validation_samples:
        command.extend(["--max-validation-samples", str(max_validation_samples)])
    return command


def run_process(command: list[str], log_path: Path) -> tuple[int, float]:
    """Stream one train process to both its durable log and the console."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        exit_code = process.wait()
    return exit_code, time.perf_counter() - started


def is_cuda_memory_failure(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace").lower()
    return "cuda out of memory" in text or "outofmemoryerror" in text


def result_row(
    point: CapacityPoint,
    run_directory: Path,
    *,
    status: str,
    duration_seconds: float | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    config = read_config(run_directory)
    validation = completed_validation(run_directory, config.get("max_steps", 0))
    return {
        "capacityPoint": point.key,
        "runName": point.run_name,
        "status": status,
        "parameters": config.get("parameters", {}).get("total"),
        "architecture": config.get("modelArchitecture", asdict(point)),
        "durationSeconds": duration_seconds,
        "peakAllocatedMiB": peak_memory_mib(run_directory),
        "validation": validation,
        "detail": detail,
    }


def write_summary(destination: Path, *, args: argparse.Namespace, rows: Iterable[dict[str, Any]]) -> None:
    payload = {
        "format": SWEEP_FORMAT,
        "comparison": {
            "train": str(args.train),
            "validation": str(args.validation),
            "batchSize": args.batch_size,
            "maxSteps": args.max_steps,
            "shuffleBufferSamples": args.shuffle_buffer_samples,
            "seed": args.seed,
            "device": args.device,
        },
        "points": list(rows),
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def selected_points(keys: list[str]) -> tuple[CapacityPoint, ...]:
    if not keys:
        return CAPACITY_V1
    available = {point.key: point for point in CAPACITY_V1}
    unknown = sorted(set(keys) - set(available))
    if unknown:
        raise ValueError(f"unknown capacity point(s): {', '.join(unknown)}")
    return tuple(available[key] for key in keys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--shuffle-buffer-samples", type=int, default=1024)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20252026)
    parser.add_argument(
        "--point",
        action="append",
        default=[],
        help="Capacity label to run; repeat to select several.",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_steps <= 0:
        raise ValueError("batch size and max steps must be positive")
    if args.shuffle_buffer_samples < args.batch_size:
        raise ValueError("shuffle buffer must hold at least one batch")
    if not args.python.is_file():
        raise FileNotFoundError(f"Python executable not found: {args.python}")
    if not args.train.is_dir() or not args.validation.is_dir():
        raise FileNotFoundError("train and validation must be converted dataset directories")

    points = selected_points(args.point)
    args.runs_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.runs_root / "capacity-sweep-v1-summary.json"
    rows: list[dict[str, Any]] = []
    blocked_by_memory = False

    print(
        json.dumps(
            {
                "phase": "capacity-sweep-start",
                "points": [point.key for point in points],
                "maxSteps": args.max_steps,
                "batchSize": args.batch_size,
                "seed": args.seed,
            },
            ensure_ascii=False,
        )
    )

    for point in points:
        run_directory = args.runs_root / point.run_name
        existing = completed_validation(run_directory, args.max_steps)
        if existing is not None:
            rows.append(result_row(point, run_directory, status="existing"))
            write_summary(summary_path, args=args, rows=rows)
            print(
                json.dumps(
                    {
                        "phase": "capacity-sweep-skip",
                        "point": point.key,
                        "reason": "complete",
                    }
                )
            )
            continue
        if blocked_by_memory:
            rows.append(
                result_row(
                    point,
                    run_directory,
                    status="not-run",
                    detail="larger than CUDA memory boundary",
                )
            )
            write_summary(summary_path, args=args, rows=rows)
            continue

        if args.dry_run:
            print(
                json.dumps(
                    {
                        "phase": "capacity-sweep-plan",
                        "point": point.key,
                        "command": make_train_command(
                            args.python, point, run_directory, args, max_steps=args.max_steps
                        ),
                    }
                )
            )
            rows.append(result_row(point, run_directory, status="planned"))
            continue

        if not args.skip_preflight:
            preflight = args.runs_root / ".capacity-preflight" / point.run_name
            preflight_log = preflight / "process.log"
            print(json.dumps({"phase": "capacity-sweep-preflight", "point": point.key}))
            preflight_code, _ = run_process(
                make_train_command(
                    args.python,
                    point,
                    preflight,
                    args,
                    max_steps=1,
                    max_validation_samples=args.batch_size,
                ),
                preflight_log,
            )
            if preflight_code != 0:
                if is_cuda_memory_failure(preflight_log):
                    blocked_by_memory = True
                    rows.append(
                        result_row(
                            point,
                            run_directory,
                            status="memory-limit",
                            detail="full training step did not fit CUDA memory",
                        )
                    )
                    write_summary(summary_path, args=args, rows=rows)
                    print(json.dumps({"phase": "capacity-sweep-memory-limit", "point": point.key}))
                    continue
                raise RuntimeError(f"preflight failed for {point.key}; see {preflight_log}")

        process_log = run_directory / "process.log"
        print(json.dumps({"phase": "capacity-sweep-train", "point": point.key}))
        exit_code, duration_seconds = run_process(
            make_train_command(args.python, point, run_directory, args, max_steps=args.max_steps),
            process_log,
        )
        if exit_code != 0:
            if is_cuda_memory_failure(process_log):
                blocked_by_memory = True
                status = "memory-limit"
                detail = "full training run did not fit CUDA memory"
            else:
                raise RuntimeError(f"training failed for {point.key}; see {process_log}")
        else:
            status = "complete"
            detail = None
        rows.append(
            result_row(
                point,
                run_directory,
                status=status,
                duration_seconds=duration_seconds,
                detail=detail,
            )
        )
        write_summary(summary_path, args=args, rows=rows)

    write_summary(summary_path, args=args, rows=rows)
    print(json.dumps({"phase": "capacity-sweep-complete", "summary": str(summary_path)}))


if __name__ == "__main__":
    main()