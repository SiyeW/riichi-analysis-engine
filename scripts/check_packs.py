"""Check a pack directory before a long training run.

Packing already audits what it wrote. This command adds the checks that only
make sense from outside the writer: that the staged games add up to the same
sample count, that the training loader sees whole batches and no empty ones,
and how many different games a window of the order actually contains. It exits
non-zero on the first violation, so a pipeline can refuse to start a long run
on a corpus that is not what it claims to be.

    python scripts/check_packs.py --packs data/processed-v4/packs-train-2025 \
        --stage data/processed-v4/staged-train-2025 --batch-size 64
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
from pathlib import Path

import numpy as np

from riichi_analysis_engine.dataset import PackDataset, read_manifest
from riichi_analysis_engine.packing import audit_packs, read_plan, staged_games
from riichi_analysis_engine.storage import read_chunk_archive_meta, read_packed_shard


def build_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--plan", type=Path, help="defaults to the plan inside the pack directory")
    parser.add_argument("--stage", type=Path, help="staged games, to check conservation from the source")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window", type=int, default=0, help="window for game diversity (default: batch size)")
    parser.add_argument("--windows", type=int, default=4096, help="how many windows to sample")
    parser.add_argument("--loader-samples", type=int, default=4096, help="samples read through the loader")
    parser.add_argument(
        "--verify-storage",
        action="store_true",
        help="read every pack and check each one is internally consistent",
    )
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def check_one_pack(path: Path) -> dict[str, object]:
    """Check one written pack against the layout its own arrays claim."""

    packed = read_packed_shard(path)
    samples = len(packed["obs_offsets"]) - 1
    if len(packed["obs_values"]) != int(packed["obs_offsets"][-1]):
        raise SystemExit(f"{path.name} holds {len(packed['obs_values'])} values for its offsets")
    for name in ("obs_nonzero", "obs_nonone", "action_mask"):
        if len(packed[name]) != samples:
            raise SystemExit(f"{path.name} holds {len(packed[name])} {name} for {samples} samples")
    for name, value in packed.items():
        if name.startswith("obs_") or value.ndim == 0 or len(value) == samples:
            continue
        raise SystemExit(f"{path.name} holds {name} with {len(value)} rows for {samples} samples")
    return {"pack": path.name, "samples": samples, "values": len(packed["obs_values"])}


def verify_storage(root: Path, manifest: dict[str, object], workers: int) -> dict[str, object]:
    """Read every pack and report what the whole corpus holds."""

    paths = [root / str(entry["pack"]) for entry in manifest["packs"]]
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(check_one_pack, paths))
    else:
        results = [check_one_pack(path) for path in paths]
    return {
        "packs": len(results),
        "samples": sum(int(result["samples"]) for result in results),
        "values": sum(int(result["values"]) for result in results),
    }


def game_stream(root: Path, manifest: dict[str, object]) -> np.ndarray:
    """The source game of every sample, in the order training will read it."""

    return np.concatenate(
        [
            np.load(root / str(entry["pack"]), allow_pickle=False)["source_game"]
            for entry in manifest["packs"]
        ]
    )


def window_diversity(stream: np.ndarray, window: int, wanted: int) -> dict[str, object]:
    """How many different games one window of the order holds, sampled evenly."""

    starts = np.linspace(0, len(stream) - window, num=min(wanted, max(1, len(stream) - window + 1)))
    counts = [len(np.unique(stream[int(start) : int(start) + window])) for start in starts]
    return {
        "window": window,
        "windowsSampled": len(counts),
        "minGamesPerWindow": min(counts),
        "medianGamesPerWindow": statistics.median(counts),
        "maxGamesPerWindow": max(counts),
    }


def longest_run(stream: np.ndarray) -> int:
    """The most consecutive samples that share a source game."""

    if len(stream) == 0:
        return 0
    breaks = np.flatnonzero(np.diff(stream) != 0) + 1
    return int(np.diff(np.concatenate(([0], breaks, [len(stream)]))).max())


def check_loader(root: Path, batch_size: int, samples: int) -> dict[str, object]:
    """Read a bounded prefix through the training loader and check the batches."""

    sizes: list[int] = []
    total = 0
    for batch in PackDataset(root, batch_size=batch_size, max_samples=samples):
        length = len(batch["policy"])
        sizes.append(length)
        total += length
    return {
        "batchSize": batch_size,
        "batches": len(sizes),
        "samples": total,
        "shortBatches": sum(1 for size in sizes if size != batch_size),
        "emptyBatches": sum(1 for size in sizes if size == 0),
    }


def main() -> None:
    arguments = build_arguments()
    manifest = read_manifest(arguments.packs)
    plan = read_plan(arguments.plan or arguments.packs / "plan.npz")
    audit = audit_packs(arguments.packs, manifest, plan)

    stream = game_stream(arguments.packs, manifest)
    if len(stream) != int(manifest["samples"]):
        raise SystemExit(f"packs hold {len(stream)} samples, the manifest says {manifest['samples']}")
    if int(stream.max()) + 1 != int(manifest["sourceGames"]):
        raise SystemExit("the packs hold a different number of source games than the manifest says")

    report: dict[str, object] = {
        "packs": len(manifest["packs"]),
        "samples": len(stream),
        "sourceGames": int(manifest["sourceGames"]),
        "audit": audit,
        "longestRunOfOneGame": longest_run(stream),
        "adjacentSameGamePairs": int(np.count_nonzero(np.diff(stream) == 0)),
        "window": window_diversity(
            stream, arguments.window or arguments.batch_size, arguments.windows
        ),
    }
    if report["longestRunOfOneGame"] != 1:
        raise SystemExit(f"a game owns {report['longestRunOfOneGame']} consecutive samples")

    if arguments.stage is not None:
        games = staged_games(arguments.stage)
        samples = sum(int(read_chunk_archive_meta(path)["samples"]) for path in games)
        chunks = sum(len(read_chunk_archive_meta(path)["chunkLengths"]) for path in games)
        if len(games) != int(manifest["sourceGames"]):
            raise SystemExit(f"the stage holds {len(games)} games, the packs say {manifest['sourceGames']}")
        if samples != len(stream):
            raise SystemExit(f"the stage holds {samples} samples, the packs hold {len(stream)}")
        report["stage"] = {"games": len(games), "chunks": chunks, "samples": samples}

    report["loader"] = check_loader(
        arguments.packs, arguments.batch_size, arguments.loader_samples
    )
    if report["loader"]["emptyBatches"]:
        raise SystemExit("the loader yielded an empty batch")
    if report["loader"]["shortBatches"] > 1:
        raise SystemExit(f"the loader yielded {report['loader']['shortBatches']} short batches")

    if arguments.verify_storage:
        storage = verify_storage(arguments.packs, manifest, arguments.workers)
        if storage["samples"] != len(stream):
            raise SystemExit(
                f"reading every pack found {storage['samples']} samples, the manifest says {len(stream)}"
            )
        report["storage"] = storage

    report["verified"] = True
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
