from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .analysis_observation import IncrementalTilePlaneEncoder
from .analysis_state import PublicHistoryState
from .constants import OBS_VERSION, relative_players
from .model_input import (
    MODEL_INPUT_SCHEMA_ID,
    compose_model_input,
    extract_policy_context,
)
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
from .semantic_input import EVENT_MEMORY_SCHEMA_ID, PublicEventHistoryEncoder
from .storage import STAGED_GAME_FORMAT, read_chunk_archive_meta, save_chunk_archive


@dataclass(frozen=True)
class ConvertedGame:
    arrays: dict[str, np.ndarray]
    event_catalog: np.ndarray


def read_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or "manifest" not in rows[0]:
        raise ValueError(f"{path} does not start with manifest metadata")
    return rows[0]["manifest"], rows[1:]


def preflight_conversion(
    manifest_path: Path,
    output: Path,
    *,
    start_game: int,
    end_game: int,
    max_games: int,
) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, object]]:
    """Validate conversion identity and paths without creating any output."""

    metadata, records = read_manifest(manifest_path)
    if not isinstance(metadata, dict):
        raise TypeError("manifest metadata must be an object")
    source_ids: set[str] = set()
    source_order: list[str] = []
    missing: list[str] = []
    zip_members: dict[Path, set[str]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"manifest record {index} must be an object")
        source_id = record.get("sourceId")
        source = record.get("path")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"manifest record {index} has no sourceId")
        if source_id in source_ids:
            raise ValueError(f"manifest repeats sourceId {source_id!r}")
        source_ids.add(source_id)
        source_order.append(source_id)
        if not isinstance(source, str) or not source:
            raise ValueError(f"manifest record {index} has no source path")
        if source.startswith("zip://"):
            locator = source[6:]
            if "!" not in locator:
                raise ValueError(f"manifest record {index} has an invalid zip path")
            archive_name, member = locator.split("!", 1)
            if not member:
                raise ValueError(f"manifest record {index} has no zip member")
            physical = Path(archive_name)
            zip_members.setdefault(physical, set()).add(member)
        else:
            physical = Path(source)
        if not physical.is_file():
            missing.append(str(physical))
    if missing:
        example = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"manifest references {len(missing)} missing source files; first: {example}"
        )
    for archive_path, expected_members in zip_members.items():
        with zipfile.ZipFile(archive_path) as archive:
            absent = expected_members.difference(archive.namelist())
        if absent:
            example = min(absent)
            raise FileNotFoundError(
                f"manifest references {len(absent)} missing members in "
                f"{archive_path}; first: {example}"
            )
    selected_count = metadata.get("selectedCount")
    if selected_count is not None and int(selected_count) != len(records):
        raise ValueError("manifest selectedCount does not match its records")
    selection_hash = metadata.get("selectionHash")
    actual_hash = hashlib.sha256("\n".join(source_order).encode()).hexdigest()
    if selection_hash is not None and selection_hash != actual_hash:
        raise ValueError("manifest selectionHash does not match its source order")
    stop = end_game or len(records)
    if not 0 <= start_game <= stop <= len(records):
        raise ValueError("game range is outside the manifest")
    if max_games < 0:
        raise ValueError("maximum games must be non-negative")
    if max_games > 0:
        stop = min(stop, start_game + max_games)
    resolved_output = output.resolve()
    if resolved_output.exists() and not resolved_output.is_dir():
        raise NotADirectoryError(f"conversion output is not a directory: {resolved_output}")
    write_boundary = resolved_output if resolved_output.exists() else resolved_output.parent
    if not write_boundary.exists():
        ancestor = write_boundary
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        if not os.access(ancestor, os.W_OK):
            raise PermissionError(f"output ancestor is not writable: {ancestor}")
    elif not os.access(write_boundary, os.W_OK):
        raise PermissionError(f"output boundary is not writable: {write_boundary}")
    return metadata, records, {
        "format": "riichi-analysis-conversion-preflight-v1",
        "manifest": str(manifest_path.resolve()),
        "output": str(resolved_output),
        "manifestGames": len(records),
        "selectedGames": stop - start_game,
        "startGame": start_game,
        "endGame": stop,
        "sourceIdsUnique": True,
        "sourceFilesPresent": True,
        "writesPerformed": False,
    }


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
        [full_state.concealed_red_counts(player) for player in absolute_opponents],
        axis=0,
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
        "system_total": np.int32(full_state.scores.sum() + full_state.kyotaku * 1_000),
    }


def convert_game(
    events: list[dict[str, Any]],
    source_id: str,
    *,
    player_state_type: Any,
) -> ConvertedGame:
    annotations = annotate_game(events)
    full_state = FullState()
    public_state = PublicHistoryState()
    analysis_encoder = IncrementalTilePlaneEncoder()
    exact_tracker = ExactTargetTracker()
    event_history = PublicEventHistoryEncoder()
    states = [player_state_type(player) for player in range(4)]
    samples: dict[str, list[Any]] = {}
    kyoku_index = -1
    history_start = -1
    history_length = 0

    def append_sample(
        event_index: int,
        event: dict[str, Any],
        perspective: int,
        exact_targets: np.ndarray,
        mortal_observation: Any,
        action_mask: Any,
        *,
        policy: int,
        analysis_active: bool,
    ) -> None:
        observation = compose_model_input(
            analysis_encoder.encode(event, perspective),
            extract_policy_context(mortal_observation),
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
            "analysis_active": np.bool_(analysis_active),
            "perspective": np.uint8(perspective),
            "event_index": np.int32(event_index),
            "kyoku_index": np.int32(kyoku_index),
            "history_start": np.uint32(history_start),
            "history_length": np.uint16(history_length),
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
        public_state.process(event)
        exact_targets = exact_tracker.process(event, states)
        if event["type"] not in FRAME_EVENTS:
            continue
        history_start, history_length = event_history.advance(event)
        analysis_encoder.advance(event, public_state)
        if exact_targets is None:
            raise RuntimeError(f"missing exact targets at {source_id}:{index}")

        sampled: set[int] = set()
        encoded: dict[int, tuple[Any, Any]] = {}
        analysis_perspective = passive_perspective(source_id, index)
        for perspective, cans in enumerate(candidates):
            mortal_observation, mask = states[perspective].encode_obs(
                OBS_VERSION, False
            )
            encoded[perspective] = (mortal_observation, mask)
            if not bool(mask.any()):
                continue
            policy, kan_tile = action_label(
                perspective, states[perspective], cans, events, index
            )
            if policy is None:
                continue
            append_sample(
                index,
                event,
                perspective,
                exact_targets,
                mortal_observation,
                mask,
                policy=policy,
                analysis_active=perspective == analysis_perspective,
            )
            sampled.add(perspective)
            if kan_tile is not None:
                kan_observation, kan_mask = states[perspective].encode_obs(
                    OBS_VERSION, True
                )
                append_sample(
                    index,
                    event,
                    perspective,
                    exact_targets,
                    kan_observation,
                    kan_mask,
                    policy=kan_tile,
                    analysis_active=False,
                )

        if analysis_perspective not in sampled:
            mortal_observation, mask = encoded[analysis_perspective]
            append_sample(
                index,
                event,
                analysis_perspective,
                exact_targets,
                mortal_observation,
                mask,
                policy=-1,
                analysis_active=True,
            )

    if not samples:
        raise ValueError(f"game produced no samples: {source_id}")
    arrays = {name: np.asarray(values) for name, values in samples.items()}
    frame_indices = arrays["event_index"]
    active_indices = frame_indices[arrays["analysis_active"]]
    frames, active_counts = np.unique(active_indices, return_counts=True)
    all_frames = np.unique(frame_indices)
    if not np.array_equal(frames, all_frames) or np.any(active_counts != 1):
        raise RuntimeError(
            f"analysis supervision is not one row per frame in {source_id}"
        )
    return ConvertedGame(arrays=arrays, event_catalog=event_history.array())


def convert_record_to_archive(
    record_index: int,
    record: dict[str, str],
    output: str,
    mortal_python_root: str,
    overwrite: bool,
    chunk_samples: int,
    compression_level: int,
) -> tuple[int, str, int, str | None]:
    """Stage one game as independently compressed sample chunks."""

    destination = Path(output) / f"game-{record_index:06d}.zip"
    if destination.exists() and not overwrite:
        try:
            meta = read_chunk_archive_meta(destination)
            if (
                meta.get("format") == STAGED_GAME_FORMAT
                and meta.get("modelInputSchema") == MODEL_INPUT_SCHEMA_ID
                and meta.get("eventMemorySchema") == EVENT_MEMORY_SCHEMA_ID
            ):
                return record_index, record["sourceId"], int(meta["samples"]), None
            destination.unlink()
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            destination.unlink()
    if mortal_python_root not in sys.path:
        sys.path.insert(0, mortal_python_root)
    from libriichi.state import PlayerState

    try:
        events = read_events(record["path"])
        converted = convert_game(
            events,
            record["sourceId"],
            player_state_type=PlayerState,
        )
        meta = save_chunk_archive(
            destination,
            converted.arrays,
            chunk_samples,
            event_catalog=converted.event_catalog,
            compression_level=compression_level,
        )
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
    parser.add_argument("--compression-level", type=int, choices=range(10), default=1)
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--start-game", type=int, default=0)
    parser.add_argument("--end-game", type=int, default=0)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--summary-name", default="summary.json")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate the manifest, runtime and output boundary without writing",
    )
    args = parser.parse_args()

    mortal_root = str(args.mortal_python_root.resolve())
    if mortal_root not in sys.path:
        sys.path.insert(0, mortal_root)
    # Import before spawning workers so a broken private training runtime fails
    # once, at the command boundary, instead of once per submitted game.
    __import__("libriichi.state")

    metadata, records, preflight = preflight_conversion(
        args.manifest,
        args.output,
        start_game=args.start_game,
        end_game=args.end_game,
        max_games=args.max_games,
    )
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return
    start_game = int(preflight["startGame"])
    end_game = int(preflight["endGame"])
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
        (
            index,
            record,
            output_root,
            mortal_root,
            args.overwrite,
            args.chunk_samples,
            args.compression_level,
        )
        for index, record in indexed_records
    ]

    summary_path = args.output / args.summary_name

    def write_summary(status: str, completed: int) -> dict[str, Any]:
        summary = {
            "format": "riichi-analysis-staged-games-v1",
            "status": status,
            "manifest": metadata,
            "requestedGames": len(indexed_records),
            "completedGames": completed,
            "convertedGames": converted_games,
            "convertedSamples": converted_samples,
            "chunkSamples": args.chunk_samples,
            "compressionLevel": args.compression_level,
            "workers": args.workers,
            "failures": failures,
            "elapsedSeconds": time.perf_counter() - start,
        }
        temporary = summary_path.with_name(
            f".{summary_path.name}.{os.getpid()}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, summary_path)
        finally:
            temporary.unlink(missing_ok=True)
        return summary

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
        write_summary("running", completed)

    completed = 0
    try:
        if args.workers > 1:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers
            ) as pool:
                job_iter = iter(jobs)
                pending: set[
                    concurrent.futures.Future[tuple[int, str, int, str | None]]
                ] = set()

                def fill_pending() -> None:
                    while len(pending) < args.workers * 2:
                        try:
                            job = next(job_iter)
                        except StopIteration:
                            break
                        pending.add(pool.submit(convert_record_to_archive, *job))

                fill_pending()
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        collect(future.result())
                        completed += 1
                        if completed == 1 or completed % 25 == 0:
                            report(completed)
                    fill_pending()
                if completed and completed % 25:
                    report(completed)
        else:
            for completed, job in enumerate(jobs, start=1):
                collect(convert_record_to_archive(*job))
                if completed % 25 == 0 or completed == len(indexed_records):
                    report(completed)
    except KeyboardInterrupt:
        write_summary("interrupted", completed)
        raise

    summary = write_summary("failed" if failures else "complete", completed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
