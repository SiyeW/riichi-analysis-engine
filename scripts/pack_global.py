"""Pack a staged corpus into globally mixed training packs.

This is the step between conversion and training. Conversion writes one
self-contained archive per game; training wants one long order in which
neighbouring samples come from different games. Packing is deliberately a
separate, restartable pass: it reads the staged corpus, writes the plan, the
packs and a manifest, and the result can be audited without touching the
staged data again.

The manifest and the plan are pure functions of the staged corpus and the
seed, so the same inputs always produce the same files. That is what lets the
audit be a real check rather than a restatement of what was just written.

    python scripts/pack_global.py --stage data/processed/staged-2025 \
        --output data/processed/packs-2025 --seed 20252026
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

from riichi_analysis_engine.packing import (
    MANIFEST_FORMAT,
    audit_packs,
    build_pack,
    pack_slots,
    plan_corpus,
    read_plan,
    seam_game,
    staged_games,
    write_manifest,
    write_plan,
)

# Worker processes receive the plan once, in their initializer, instead of
# once per pack: on the full corpus the plan alone is a few hundred megabytes.
_WORKER: dict[str, object] = {}


def build_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True, help="directory of game-*.zip archives")
    parser.add_argument("--output", type=Path, required=True, help="directory to write packs into")
    parser.add_argument("--seed", type=int, default=20252026)
    parser.add_argument(
        "--game-offset",
        type=int,
        default=0,
        help="number this stage's games after the games of the stages before it",
    )
    parser.add_argument(
        "--first-forbidden-game",
        type=int,
        default=-1,
        help="source game the first sample must avoid, for a build split in pieces",
    )
    parser.add_argument(
        "--pack-samples",
        type=int,
        default=65536,
        help="approximate sample count of one pack; a pack never splits a chunk",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--limit-games",
        type=int,
        default=0,
        help="use only the first N staged games, for a trial run",
    )
    parser.add_argument(
        "--reuse-plan",
        action="store_true",
        help="reuse an existing plan instead of rebuilding it from the stage",
    )
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def _initialize_worker(
    games: list[Path], plan: dict, output: Path, seed: int, forbidden_first: int | None
) -> None:
    _WORKER.update(
        games=games, plan=plan, output=output, seed=seed, forbidden_first=forbidden_first
    )


def _build_slot(slot) -> tuple[int, dict[str, object]]:
    plan = _WORKER["plan"]
    # A pack starts where the previous one ended, so the seam between two packs
    # cannot join two samples of the same game either.
    forbidden = seam_game(plan, slot)
    if slot.index == 0:
        forbidden = _WORKER["forbidden_first"]
    meta, _ = build_pack(
        _WORKER["games"], _WORKER["output"], plan, slot, _WORKER["seed"], forbidden
    )
    return slot.index, meta


def report_progress(index: int, total: int, started: float) -> None:
    if (index + 1) % 16 and index + 1 != total:
        return
    elapsed = time.perf_counter() - started
    print(
        f"pack {index + 1}/{total} ({elapsed:.0f}s elapsed, {elapsed / (index + 1):.2f}s/pack)",
        file=sys.stderr,
    )


def main() -> None:
    arguments = build_arguments()
    arguments.output.mkdir(parents=True, exist_ok=True)
    plan_path = arguments.output / "plan.npz"
    started = time.perf_counter()

    if arguments.audit_only:
        plan = read_plan(plan_path)
        manifest = json.loads((arguments.output / "manifest.json").read_text(encoding="utf-8"))
        report = audit_packs(arguments.output, manifest, plan)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    games = staged_games(arguments.stage)
    if arguments.limit_games:
        games = games[: arguments.limit_games]
    if arguments.reuse_plan and plan_path.exists():
        plan = read_plan(plan_path)
        print(f"reused {plan_path} with {len(plan['length'])} chunks", file=sys.stderr)
    else:
        plan = plan_corpus(games, arguments.seed, arguments.game_offset)
        write_plan(plan_path, plan)
        print(
            f"planned {len(plan['length'])} chunks over {len(games)} games "
            f"({int(plan['length'].sum())} samples) in {time.perf_counter() - started:.1f}s",
            file=sys.stderr,
        )

    first_forbidden = (
        arguments.first_forbidden_game if arguments.first_forbidden_game >= 0 else None
    )
    slots = pack_slots(plan["length"], arguments.pack_samples)
    print(f"writing {len(slots)} packs", file=sys.stderr)
    entries: list[dict[str, object]] = []
    if arguments.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=arguments.workers,
            initializer=_initialize_worker,
            initargs=(games, plan, arguments.output, arguments.seed, first_forbidden),
        ) as pool:
            for index, meta in pool.map(_build_slot, slots):
                entries.append(meta)
                report_progress(index, len(slots), started)
    else:
        _initialize_worker(games, plan, arguments.output, arguments.seed, first_forbidden)
        for slot in slots:
            index, meta = _build_slot(slot)
            entries.append(meta)
            report_progress(index, len(slots), started)

    entries.sort(key=lambda entry: str(entry["pack"]))
    manifest: dict[str, object] = {
        "format": MANIFEST_FORMAT,
        "stage": str(arguments.stage),
        "seed": arguments.seed,
        "gameOffset": arguments.game_offset,
        "packSamples": arguments.pack_samples,
        "samples": int(plan["length"].sum()),
        "chunks": len(plan["length"]),
        "sourceGames": int(plan["source_game"].max()) + 1,
        "packs": entries,
    }
    write_manifest(arguments.output / "manifest.json", manifest)
    report = audit_packs(arguments.output, manifest, plan)
    manifest["audit"] = report
    write_manifest(arguments.output / "manifest.json", manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
