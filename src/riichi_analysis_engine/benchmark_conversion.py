"""Benchmark a small, reproducible slice of dataset conversion.

The benchmark runs ordinary converter subprocesses against the same manifest
range, verifies that every case produced the same games and samples, and writes
one machine-readable report.  It never changes the source manifest or an
existing conversion output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .storage import read_chunk_archive_meta

BENCHMARK_FORMAT = "riichi-analysis-conversion-benchmark-v1"
DEFAULT_CASES = ("1:1", "2:1", "4:1", "2:6")


@dataclass(frozen=True)
class BenchmarkCase:
    workers: int
    compression_level: int

    @property
    def key(self) -> str:
        return f"workers-{self.workers}-compression-{self.compression_level}"


def parse_case(value: str) -> BenchmarkCase:
    try:
        workers_text, compression_text = value.split(":", 1)
        workers = int(workers_text)
        compression_level = int(compression_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("case must be WORKERS:COMPRESSION") from error
    if workers <= 0:
        raise argparse.ArgumentTypeError("case workers must be positive")
    if not 0 <= compression_level <= 9:
        raise argparse.ArgumentTypeError("case compression must be in 0..9")
    return BenchmarkCase(workers, compression_level)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _case_metrics(output: Path) -> dict[str, Any]:
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"converter did not write {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    games = sorted(output.glob("game-*.zip"))
    metas = [read_chunk_archive_meta(path) for path in games]
    return {
        "converterStatus": summary.get("status"),
        "completedGames": summary.get("completedGames"),
        "failures": summary.get("failures"),
        "samples": sum(int(meta["samples"]) for meta in metas),
        "bytes": sum(
            path.stat().st_size for path in output.rglob("*") if path.is_file()
        ),
        "chunkSamples": sorted({int(meta["chunkSamples"]) for meta in metas}),
        "compressionLevels": sorted(
            {int(meta["compressionLevel"]) for meta in metas}
        ),
    }


def validate_comparable(rows: list[dict[str, Any]], expected_games: int) -> None:
    if not rows:
        raise ValueError("conversion benchmark produced no cases")
    for row in rows:
        if row.get("status") != "complete":
            raise ValueError(f"benchmark case did not complete: {row.get('case')}")
        if row.get("converterStatus") != "complete" or row.get("failures"):
            raise ValueError(f"converter reported a failure: {row.get('case')}")
        if row.get("completedGames") != expected_games:
            raise ValueError(f"benchmark case converted the wrong game count: {row.get('case')}")
    sample_counts = {int(row["samples"]) for row in rows}
    if len(sample_counts) != 1:
        raise ValueError("benchmark cases produced different sample counts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mortal-python-root", type=Path, required=True)
    parser.add_argument("--max-games", type=int, default=4)
    parser.add_argument("--start-game", type=int, default=0)
    parser.add_argument("--chunk-samples", type=int, default=16)
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        help=(
            "WORKERS:COMPRESSION; repeat to choose cases "
            f"(default: {', '.join(DEFAULT_CASES)})"
        ),
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="retain successful staged-game outputs instead of only the report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_games <= 0:
        raise ValueError("max-games must be positive")
    if args.start_game < 0:
        raise ValueError("start-game must not be negative")
    if args.chunk_samples <= 0:
        raise ValueError("chunk-samples must be positive")
    manifest = args.manifest.resolve(strict=True)
    mortal_python_root = args.mortal_python_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"benchmark output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "benchmark.json"
    cases = args.case or [parse_case(value) for value in DEFAULT_CASES]
    if len({case.key for case in cases}) != len(cases):
        raise ValueError("benchmark cases must be unique")
    report: dict[str, Any] = {
        "format": BENCHMARK_FORMAT,
        "status": "running",
        "machine": {
            "node": platform.node(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logicalCpuCount": os.cpu_count(),
            "python": sys.version,
        },
        "sourceRevision": _source_revision(),
        "manifest": str(manifest),
        "manifestSha256": _sha256(manifest),
        "startGame": args.start_game,
        "maxGames": args.max_games,
        "chunkSamples": args.chunk_samples,
        "cases": [],
    }
    _write_report(report_path, report)
    try:
        for case in cases:
            output = output_root / case.key
            command = [
                sys.executable,
                "-m",
                "riichi_analysis_engine.convert",
                "--manifest",
                str(manifest),
                "--output",
                str(output),
                "--mortal-python-root",
                str(mortal_python_root),
                "--start-game",
                str(args.start_game),
                "--max-games",
                str(args.max_games),
                "--chunk-samples",
                str(args.chunk_samples),
                "--workers",
                str(case.workers),
                "--compression-level",
                str(case.compression_level),
            ]
            print(f"benchmarking {case.key}", flush=True)
            started = time.perf_counter()
            completed = subprocess.run(command, check=False)
            duration = time.perf_counter() - started
            row: dict[str, Any] = {
                "case": case.key,
                "workers": case.workers,
                "compressionLevel": case.compression_level,
                "durationSeconds": duration,
                "exitCode": completed.returncode,
                "status": "failed" if completed.returncode else "complete",
            }
            if completed.returncode == 0:
                row.update(_case_metrics(output))
                if not args.keep_outputs:
                    shutil.rmtree(output)
            report["cases"].append(row)
            _write_report(report_path, report)
            if completed.returncode:
                raise RuntimeError(f"converter failed for {case.key}")
        validate_comparable(report["cases"], args.max_games)
        report["status"] = "complete"
    except BaseException:
        report["status"] = "failed"
        _write_report(report_path, report)
        raise
    _write_report(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
