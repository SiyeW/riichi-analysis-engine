from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .constants import OBS_VERSION, relative_players
from .observations import add_all_player_ranks
from .replay import (
    FRAME_EVENTS,
    ExactTargetTracker,
    FullState,
    action_label,
    annotate_game,
    passive_perspective,
    read_events,
    rotated_future,
)
from .storage import read_chunk_archive_meta, save_chunk_archive
from .score_state import relative_scores


def read_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or "manifest" not in rows[0]:
        raise ValueError(f"{path} does not start with manifest metadata")
    return rows[0]["manifest"], rows[1:]


def _sample_targets(
    perspective: int,
    exact_targets: np.ndarray,
    full_state: FullState,
    future: dict[str, np.ndarray | int],
    annotation: Any,
) -> dict[str, np.ndarray | int]:
    row = exact_targets[perspective]
    shanten = np.empty(3, dtype=np.uint8)
    furiten = np.empty(3, dtype=np.uint8)
    deal_in = np.empty((3, 34), dtype=np.uint8)
    for opponent in range(3):
        base = opponent * 8
        shanten[opponent] = int(np.argmax(row[base : base + 7]))
        furiten[opponent] = int(row[base + 7] > 0.5)
        deal_in[opponent] = row[24 + opponent * 34 : 24 + (opponent + 1) * 34] > 0.5
    absolute_opponents = relative_players(perspective)
    concealed = np.stack(
        [full_state.concealed_counts(player) for player in absolute_opponents], axis=0
    )
    concealed_red = np.stack(
        [full_state.concealed_red_counts(player) for player in absolute_opponents], axis=0
    )
    winner_mask = annotation.win[list(absolute_opponents)].astype(np.uint8, copy=False)
    return {
        "shanten": shanten,
        "furiten_no_yaku": furiten,
        "deal_in_tile": deal_in,
        "concealed_count": concealed,
        "concealed_red_count": concealed_red,
        "wall_count": full_state.wall.copy(),
        "wall_red_count": full_state.wall_red.copy(),
        "dora": future["dora"],
        "score": future["score"],
        "winner_mask": winner_mask,
        "draw": future["draw"],
        "win": future["win"],
        "deal_in_player": future["deal_in_player"],
        "outcome": future["outcome"],
        "kyoku_delta": future["kyoku_delta"],
        "placement": future["placement"],
        "match_score": future["match_score"],
    }


def convert_game(
    events: list[dict[str, Any]],
    source_id: str,
    *,
    player_state_type: Any,
) -> dict[str, np.ndarray]:
    annotations = annotate_game(events)
    full_state = FullState()
    exact_tracker = ExactTargetTracker()
    states = [player_state_type(player) for player in range(4)]
    samples: dict[str, list[Any]] = {}
    kyoku_index = -1

    def append_sample(
        event_index: int,
        perspective: int,
        exact_targets: np.ndarray,
        *,
        policy: int,
        kan_select: bool,
    ) -> None:
        observation, action_mask = states[perspective].encode_obs(OBS_VERSION, kan_select)
        observation = add_all_player_ranks(
            observation, relative_scores(full_state.scores, perspective)
        )
        if policy >= 0 and not bool(action_mask[policy]):
            raise ValueError(
                f"illegal policy label {policy} at {source_id}:{event_index}, "
                f"seat={perspective}, mask={np.flatnonzero(action_mask).tolist()}"
            )
        annotation = annotations[event_index]
        future = rotated_future(annotation, full_state.scores, perspective)
        targets = _sample_targets(
            perspective, exact_targets, full_state, future, annotation
        )
        values: dict[str, Any] = {
            "obs": np.asarray(observation, dtype=np.float16),
            "action_mask": np.asarray(action_mask, dtype=bool),
            "policy": np.int8(policy),
            "perspective": np.uint8(perspective),
            "event_index": np.int32(event_index),
            "kyoku_index": np.int32(kyoku_index),
            **targets,
        }
        for name, value in values.items():
            samples.setdefault(name, []).append(value)

    for index, event in enumerate(events):
        if event["type"] == "start_kyoku":
            kyoku_index += 1
        event_json = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        candidates = [state.update(event_json) for state in states]
        full_state.process(event)
        exact_targets = exact_tracker.process(event, states)
        if event["type"] not in FRAME_EVENTS:
            continue
        if exact_targets is None:
            raise RuntimeError(f"missing exact targets at {source_id}:{index}")

        sampled: set[int] = set()
        for perspective, cans in enumerate(candidates):
            _obs, mask = states[perspective].encode_obs(OBS_VERSION, False)
            if not bool(mask.any()):
                continue
            policy, kan_tile = action_label(
                perspective, states[perspective], cans, events, index
            )
            if policy is None:
                continue
            append_sample(
                index,
                perspective,
                exact_targets,
                policy=policy,
                kan_select=False,
            )
            sampled.add(perspective)
            if kan_tile is not None:
                append_sample(
                    index,
                    perspective,
                    exact_targets,
                    policy=kan_tile,
                    kan_select=True,
                )

        passive = passive_perspective(source_id, index)
        if passive not in sampled:
            append_sample(
                index,
                passive,
                exact_targets,
                policy=-1,
                kan_select=False,
            )

    if not samples:
        raise ValueError(f"game produced no samples: {source_id}")
    return {name: np.asarray(values) for name, values in samples.items()}


def convert_record_to_archive(
    record_index: int,
    record: dict[str, str],
    output: str,
    mortal_python_root: str,
    overwrite: bool,
    chunk_samples: int,
) -> tuple[int, str, int, str | None]:
    """Stage one game as independently compressed sample chunks."""

    destination = Path(output) / f"game-{record_index:06d}.zip"
    if destination.exists() and not overwrite:
        try:
            meta = read_chunk_archive_meta(destination)
            return record_index, record["sourceId"], int(meta["samples"]), None
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            destination.unlink()
    if mortal_python_root not in sys.path:
        sys.path.insert(0, mortal_python_root)
    from libriichi.state import PlayerState

    try:
        events = read_events(record["path"])
        arrays = convert_game(
            events,
            record["sourceId"],
            player_state_type=PlayerState,
        )
        meta = save_chunk_archive(destination, arrays, chunk_samples)
        return record_index, record["sourceId"], int(meta["samples"]), None
    except Exception as error:  # noqa: BLE001 -- one malformed game must not stop a batch
        return record_index, record["sourceId"], 0, repr(error)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert mjai logs into staged games of compressed sample chunks."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mortal-python-root",
        type=Path,
        required=True,
        help="directory holding libriichi, usually <runtime>/mortal",
    )
    parser.add_argument("--label-source-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--chunk-samples", type=int, default=16)
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--start-game", type=int, default=0)
    parser.add_argument("--end-game", type=int, default=0)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--summary-name", default="summary.json")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    mortal_root = str(args.mortal_python_root.resolve())
    if mortal_root not in sys.path:
        sys.path.insert(0, mortal_root)
    from libriichi.state import PlayerState

    metadata, records = read_manifest(args.manifest)
    start_game = args.start_game
    end_game = args.end_game or len(records)
    if not 0 <= start_game <= end_game <= len(records):
        raise ValueError("game range is outside the manifest")
    if args.max_games > 0:
        end_game = min(end_game, start_game + args.max_games)
    indexed_records = list(enumerate(records[start_game:end_game], start=start_game))
    if args.reverse:
        indexed_records.reverse()
    args.output.mkdir(parents=True, exist_ok=True)
    converted_games = 0
    converted_samples = 0
    failures: list[dict[str, str]] = []
    start = time.perf_counter()
    output_root = str(args.output.resolve())
    jobs = [
        (index, record, output_root, mortal_root, args.overwrite, args.chunk_samples)
        for index, record in indexed_records
    ]

    def collect(result: tuple[int, str, int, str | None]) -> None:
        nonlocal converted_games, converted_samples
        _index, source_id, samples, error = result
        if error is None:
            converted_games += 1
            converted_samples += samples
        else:
            failures.append({"sourceId": source_id, "error": error})
            print(f"FAILED {source_id}: {error}", file=sys.stderr)

    def report(completed: int) -> None:
        print(
            f"converted {completed}/{len(indexed_records)} games; "
            f"samples={converted_samples}; failures={len(failures)}"
        )

    if args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(convert_record_to_archive, *job) for job in jobs]
            for completed, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                collect(future.result())
                if completed == 1 or completed % 25 == 0 or completed == len(indexed_records):
                    report(completed)
    else:
        for completed, job in enumerate(jobs, start=1):
            collect(convert_record_to_archive(*job))
            if completed % 25 == 0 or completed == len(indexed_records):
                report(completed)

    summary = {
        "format": "riichi-analysis-staged-games-v1",
        "manifest": metadata,
        "requestedGames": len(indexed_records),
        "convertedGames": converted_games,
        "convertedSamples": converted_samples,
        "chunkSamples": args.chunk_samples,
        "workers": args.workers,
        "failures": failures,
        "elapsedSeconds": time.perf_counter() - start,
    }
    (args.output / args.summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
