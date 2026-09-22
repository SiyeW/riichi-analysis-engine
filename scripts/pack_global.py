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

    python scripts/pack_global.py --stage data/processed/staged-train \
        --output data/processed/packs-train --seed 314159
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

import numpy as np

from riichi_analysis_engine.packing import (
    MANIFEST_FORMAT,
    PackSlot,
    audit_packs,
    build_pack,
    nonadjacent_order,
    pack_slots,
    plan_corpus,
    read_plan,
    seam_game,
    staged_games,
    write_manifest,
    write_plan,
)
from riichi_analysis_engine.storage import read_packed_shard

# Worker processes receive the plan once, in their initializer, instead of
# once per pack: on the full corpus the plan alone is a few hundred megabytes.
_WORKER: dict[str, object] = {}


def build_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", type=Path, required=True, help="directory of game-*.zip archives"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="directory to write packs into"
    )
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument(
        "--game-offset",
        type=int,
        default=0,
        help="add this value to the game numbers already present in the archive names",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "verify and reuse complete pack files in an interrupted output directory; "
            "requires its existing plan.npz"
        ),
    )
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def _initialize_worker(
    games: list[Path], plan: dict, output: Path, seed: int, forbidden_first: int | None
) -> None:
    _WORKER.update(
        games=games,
        plan=plan,
        output=output,
        seed=seed,
        forbidden_first=forbidden_first,
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


def _existing_pack_meta(
    output: Path,
    plan: dict,
    slot: PackSlot,
    seed: int,
    first_forbidden: int | None,
) -> dict[str, object] | None:
    """Validate one atomically published pack before an interrupted build reuses it."""

    path = output / f"pack-{slot.index:05d}.npz"
    if not path.exists():
        return None
    packed = read_packed_shard(path)
    sample_count = int(plan["length"][slot.start : slot.stop].sum())
    if len(packed["source_game"]) != sample_count:
        raise ValueError(
            f"{path.name} has {len(packed['source_game'])} samples, expected {sample_count}"
        )
    pack_indices = np.unique(packed["pack_index"])
    if pack_indices.tolist() != [slot.index]:
        raise ValueError(f"{path.name} declares pack indices {pack_indices.tolist()}")
    assigned_games = np.repeat(
        plan["source_game"][slot.start : slot.stop],
        plan["length"][slot.start : slot.stop],
    )
    actual_games = packed["source_game"]
    forbidden = seam_game(plan, slot)
    if slot.index == 0:
        forbidden = first_forbidden
    order = nonadjacent_order(
        assigned_games,
        seed=int(np.random.SeedSequence([seed, slot.index]).generate_state(1)[0]),
        forbidden_first=forbidden,
        last_game=int(assigned_games[-1]),
    )
    if not np.array_equal(actual_games, assigned_games[order]):
        raise ValueError(f"{path.name} does not match the deterministic pack plan")
    meta: dict[str, object] = {
        "pack": path.name,
        "samples": len(actual_games),
        "chunks": slot.stop - slot.start,
        "sourceGames": len(np.unique(actual_games)),
        "firstGame": int(actual_games[0]),
        "lastGame": int(actual_games[-1]),
        "modelInputSchema": packed["model_input_schema"].item(),
        "observationChannels": int(packed["obs_channels"].item()),
    }
    if "event_memory_schema" in packed:
        meta["eventMemorySchema"] = packed["event_memory_schema"].item()
    if "training_target_schema" in packed:
        meta["trainingTargetSchema"] = packed["training_target_schema"].item()
    return meta


def main() -> None:
    arguments = build_arguments()
    plan_path = arguments.output / "plan.npz"
    started = time.perf_counter()

    if arguments.audit_only:
        plan = read_plan(plan_path)
        manifest = json.loads(
            (arguments.output / "manifest.json").read_text(encoding="utf-8")
        )
        report = audit_packs(arguments.output, manifest, plan)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if arguments.resume:
        if not arguments.output.is_dir() or not plan_path.is_file():
            raise ValueError(
                "--resume requires an existing output directory and plan.npz"
            )
        temporary = sorted(arguments.output.glob("*.tmp"))
        if temporary:
            raise ValueError(
                "resume output contains temporary files: "
                + ", ".join(path.name for path in temporary)
            )
    else:
        if arguments.output.exists() and any(arguments.output.iterdir()):
            raise ValueError(
                "output directory is not empty; use --resume after an interruption"
            )
        arguments.output.mkdir(parents=True, exist_ok=True)

    games = staged_games(arguments.stage)
    if arguments.limit_games:
        games = games[: arguments.limit_games]
    if arguments.resume or (arguments.reuse_plan and plan_path.exists()):
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
    expected_names = {f"pack-{slot.index:05d}.npz" for slot in slots}
    unexpected = sorted(
        path.name
        for path in arguments.output.glob("pack-*.npz")
        if path.name not in expected_names
    )
    if unexpected:
        raise ValueError(
            "output contains packs outside this plan: " + ", ".join(unexpected)
        )
    existing: dict[int, dict[str, object]] = {}
    if arguments.resume:
        for slot in slots:
            meta = _existing_pack_meta(
                arguments.output, plan, slot, arguments.seed, first_forbidden
            )
            if meta is not None:
                existing[slot.index] = meta
        print(f"verified {len(existing)} existing packs", file=sys.stderr)
    pending = [slot for slot in slots if slot.index not in existing]
    entries: list[dict[str, object]] = list(existing.values())
    if arguments.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=arguments.workers,
            initializer=_initialize_worker,
            initargs=(games, plan, arguments.output, arguments.seed, first_forbidden),
        ) as pool:
            for index, meta in pool.map(_build_slot, pending):
                entries.append(meta)
                report_progress(index, len(slots), started)
    else:
        _initialize_worker(
            games, plan, arguments.output, arguments.seed, first_forbidden
        )
        for slot in pending:
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
        "sourceGames": len(set(plan["source_game"].tolist())),
        "firstSourceGame": int(plan["source_game"].min()),
        "lastSourceGame": int(plan["source_game"].max()),
        "packs": entries,
        "modelInputSchema": entries[0]["modelInputSchema"],
        "observationChannels": entries[0]["observationChannels"],
        "eventMemorySchema": entries[0].get("eventMemorySchema"),
        "trainingTargetSchema": entries[0].get("trainingTargetSchema"),
    }
    write_manifest(arguments.output / "manifest.json", manifest)
    report = audit_packs(arguments.output, manifest, plan)
    manifest["audit"] = report
    write_manifest(arguments.output / "manifest.json", manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
