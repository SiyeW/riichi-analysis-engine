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
from .frame_sampling import SAMPLING_STRATA, FrameSamplingPlan
from .model_input import (
    MODEL_INPUT_SCHEMA_ID,
    SHARED_MODEL_INPUT_SCHEMA_ID,
    compose_model_input,
    compose_shared_model_input,
    extract_policy_context,
)
from .replay import (
    FRAME_EVENTS,
    WORKER_EVENT_ARCHIVES,
    ExactTargetTracker,
    FullState,
    HiddenBaselineAnchorTracker,
    action_label,
    annotate_game,
    passive_perspective,
    read_events,
    rotated_future,
)
from .rule_certainties import PublicRuleState
from .rule_context import encode_rule_context
from .semantic_input import EVENT_MEMORY_SCHEMA_ID, PublicEventHistoryEncoder
from .storage import (
    STAGED_GAME_FORMAT,
    TRAINING_TARGET_SCHEMA_ID,
    read_chunk_archive_meta,
    save_chunk_archive,
)


@dataclass(frozen=True)
class ConvertedGame:
    arrays: dict[str, np.ndarray]
    event_catalog: np.ndarray
    frame_counts: dict[str, dict[str, int]]


_ACTION_CANDIDATE_FLAGS = (
    "can_discard",
    "can_riichi",
    "can_chi_low",
    "can_chi_mid",
    "can_chi_high",
    "can_pon",
    "can_daiminkan",
    "can_ankan",
    "can_kakan",
    "can_tsumo_agari",
    "can_ron_agari",
    "can_ryukyoku",
    "can_pass",
)


def _candidate_has_legal_action(candidates: Any) -> bool:
    """Check action availability before materializing Mortal's observation."""

    for name in _ACTION_CANDIDATE_FLAGS:
        value = getattr(candidates, name, False)
        if callable(value):
            value = value()
        if bool(value):
            return True
    return False


def _analysis_perspective(
    source_id: str,
    event_index: int,
    baseline_anchors: np.ndarray | None = None,
) -> int:
    """Select analysis supervision without consulting future labels.

    If any perspective is still in the exact no-information state, retain one
    of those views so the scarce analytic anchors are not lost merely because
    the ordinary passive-perspective hash selected another seat.
    """

    selection = passive_perspective(source_id, event_index)
    if baseline_anchors is None:
        return selection
    anchored = np.flatnonzero(baseline_anchors)
    if not len(anchored):
        return selection
    return int(anchored[selection % len(anchored)])


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
        raise NotADirectoryError(
            f"conversion output is not a directory: {resolved_output}"
        )
    write_boundary = (
        resolved_output if resolved_output.exists() else resolved_output.parent
    )
    if not write_boundary.exists():
        ancestor = write_boundary
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        if not os.access(ancestor, os.W_OK):
            raise PermissionError(f"output ancestor is not writable: {ancestor}")
    elif not os.access(write_boundary, os.W_OK):
        raise PermissionError(f"output boundary is not writable: {write_boundary}")
    return (
        metadata,
        records,
        {
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
        },
    )


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
    model_format: int = 13,
    sampling_plan: FrameSamplingPlan | None = None,
) -> ConvertedGame:
    annotations = annotate_game(events)
    full_state = FullState()
    public_state = PublicHistoryState()
    rule_state = PublicRuleState()
    analysis_encoder = IncrementalTilePlaneEncoder()
    exact_tracker = ExactTargetTracker()
    hidden_anchor_tracker = HiddenBaselineAnchorTracker()
    event_history = PublicEventHistoryEncoder()
    states = [player_state_type(player) for player in range(4)]
    samples: dict[str, list[Any]] = {}
    kyoku_index = -1
    history_start = -1
    history_length = 0
    frame_counts = {name: {"seen": 0, "kept": 0} for name in SAMPLING_STRATA}
    analysis_kept_frames: set[int] = set()

    def append_sample(
        event_index: int,
        event: dict[str, Any],
        perspective: int,
        exact_targets: np.ndarray,
        mortal_observation: Any,
        action_mask: Any,
        state: Any,
        candidates: Any,
        *,
        policy: int,
        analysis_active: bool,
        hidden_baseline_anchor: bool,
        at_kan_select: bool = False,
    ) -> None:
        analysis = analysis_encoder.encode(event, perspective)
        observation = (
            compose_shared_model_input(
                analysis,
                encode_rule_context(
                    state,
                    candidates,
                    action_mask,
                    at_kan_select=at_kan_select,
                    seat=perspective,
                    rule_state=rule_state,
                ),
            )
            if model_format == 13
            else compose_model_input(
                analysis, extract_policy_context(mortal_observation)
            )
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
        if model_format == 13:
            values["hidden_baseline_anchor"] = np.bool_(hidden_baseline_anchor)
        for name, value in values.items():
            samples.setdefault(name, []).append(value)

    for index, event in enumerate(events):
        if event["type"] == "start_kyoku":
            kyoku_index += 1
        event_json = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        candidates = [state.update(event_json) for state in states]
        full_state.process(event)
        public_state.process(event)
        rule_state.process(event)
        exact_targets = exact_tracker.process(event, states)
        hidden_baseline_anchors = hidden_anchor_tracker.process(event)
        if event["type"] not in FRAME_EVENTS:
            continue
        history_start, history_length = event_history.advance(event)
        analysis_encoder.advance(event, public_state)
        if exact_targets is None:
            raise RuntimeError(f"missing exact targets at {source_id}:{index}")
        if hidden_baseline_anchors is None:
            raise RuntimeError(f"missing hidden baseline state at {source_id}:{index}")

        sampled: set[int] = set()
        encoded: dict[int, tuple[Any, Any]] = {}

        def encode_primary(
            perspective: int,
            cache: dict[int, tuple[Any, Any]] = encoded,
            player_states: list[Any] = states,
        ) -> tuple[Any, Any]:
            cached = cache.get(perspective)
            if cached is None:
                cached = player_states[perspective].encode_obs(OBS_VERSION, False)
                cache[perspective] = cached
            return cached

        analysis_perspective = _analysis_perspective(
            source_id, index, hidden_baseline_anchors
        )
        analysis_stratum = FrameSamplingPlan.analysis_stratum(
            event,
            baseline_anchor=bool(hidden_baseline_anchors[analysis_perspective]),
        )
        frame_counts[analysis_stratum]["seen"] += 1
        keep_analysis = sampling_plan is None or sampling_plan.keep(
            source_id, index, analysis_perspective, analysis_stratum
        )
        if keep_analysis:
            frame_counts[analysis_stratum]["kept"] += 1
            analysis_kept_frames.add(index)

        for perspective, cans in enumerate(candidates):
            if not _candidate_has_legal_action(cans):
                continue
            policy, kan_tile = action_label(
                perspective, states[perspective], cans, events, index
            )
            if policy is None:
                continue
            policy_stratum = FrameSamplingPlan.policy_stratum(event, cans)
            frame_counts[policy_stratum]["seen"] += 1
            keep_policy = sampling_plan is None or sampling_plan.keep(
                source_id, index, perspective, policy_stratum
            )
            if not keep_policy:
                continue
            mortal_observation, mask = encode_primary(perspective)
            if not bool(mask.any()):
                continue
            frame_counts[policy_stratum]["kept"] += 1
            append_sample(
                index,
                event,
                perspective,
                exact_targets,
                mortal_observation,
                mask,
                states[perspective],
                cans,
                policy=policy,
                analysis_active=(keep_analysis and perspective == analysis_perspective),
                hidden_baseline_anchor=bool(hidden_baseline_anchors[perspective]),
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
                    states[perspective],
                    cans,
                    policy=kan_tile,
                    analysis_active=False,
                    hidden_baseline_anchor=bool(hidden_baseline_anchors[perspective]),
                    at_kan_select=True,
                )

        if keep_analysis and analysis_perspective not in sampled:
            mortal_observation, mask = encode_primary(analysis_perspective)
            append_sample(
                index,
                event,
                analysis_perspective,
                exact_targets,
                mortal_observation,
                mask,
                states[analysis_perspective],
                candidates[analysis_perspective],
                policy=-1,
                analysis_active=True,
                hidden_baseline_anchor=bool(
                    hidden_baseline_anchors[analysis_perspective]
                ),
            )

    if not samples:
        raise ValueError(f"game produced no samples: {source_id}")
    arrays = {name: np.asarray(values) for name, values in samples.items()}
    frame_indices = arrays["event_index"]
    active_indices = frame_indices[arrays["analysis_active"]]
    active_frames, active_counts = np.unique(active_indices, return_counts=True)
    expected_active_frames = np.asarray(sorted(analysis_kept_frames), dtype=np.int32)
    if not np.array_equal(active_frames, expected_active_frames) or np.any(
        active_counts != 1
    ):
        raise RuntimeError(
            f"analysis supervision does not match selected frames in {source_id}"
        )
    return ConvertedGame(
        arrays=arrays,
        event_catalog=event_history.array(),
        frame_counts=frame_counts,
    )


def convert_record_to_archive(
    record_index: int,
    record: dict[str, str],
    output: str,
    mortal_python_root: str,
    overwrite: bool,
    chunk_samples: int,
    compression_level: int,
    model_format: int,
    sampling_plan: FrameSamplingPlan | None,
    use_archive_cache: bool,
) -> tuple[int, str, int, dict[str, dict[str, int]], str | None]:
    """Stage one game as independently compressed sample chunks."""

    destination = Path(output) / f"game-{record_index:06d}.zip"
    if destination.exists() and not overwrite:
        try:
            meta = read_chunk_archive_meta(destination)
            if (
                meta.get("format") == STAGED_GAME_FORMAT
                and meta.get("modelInputSchema")
                == (
                    SHARED_MODEL_INPUT_SCHEMA_ID
                    if model_format == 13
                    else MODEL_INPUT_SCHEMA_ID
                )
                and meta.get("eventMemorySchema") == EVENT_MEMORY_SCHEMA_ID
                and (
                    model_format != 13
                    or meta.get("trainingTargetSchema") == TRAINING_TARGET_SCHEMA_ID
                )
                and meta.get("frameSampling")
                == (sampling_plan.metadata() if sampling_plan else None)
            ):
                counts = meta.get("frameCounts", {})
                return (
                    record_index,
                    record["sourceId"],
                    int(meta["samples"]),
                    counts if isinstance(counts, dict) else {},
                    None,
                )
            destination.unlink()
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            destination.unlink()
    if mortal_python_root not in sys.path:
        sys.path.insert(0, mortal_python_root)
    from libriichi.state import PlayerState

    try:
        events = (
            WORKER_EVENT_ARCHIVES.read_events(record["path"])
            if use_archive_cache
            else read_events(record["path"])
        )
        converted = convert_game(
            events,
            record["sourceId"],
            player_state_type=PlayerState,
            model_format=model_format,
            sampling_plan=sampling_plan,
        )
        meta = save_chunk_archive(
            destination,
            converted.arrays,
            chunk_samples,
            event_catalog=converted.event_catalog,
            compression_level=compression_level,
            archive_metadata={
                "frameSampling": sampling_plan.metadata() if sampling_plan else None,
                "frameCounts": converted.frame_counts,
            },
            training_target_schema=(
                TRAINING_TARGET_SCHEMA_ID if model_format == 13 else None
            ),
        )
        return (
            record_index,
            record["sourceId"],
            int(meta["samples"]),
            converted.frame_counts,
            None,
        )
    except Exception as error:  # noqa: BLE001 -- one malformed game must not stop a batch
        return record_index, record["sourceId"], 0, {}, repr(error)


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
    parser.add_argument("--model-format", type=int, choices=(12, 13), default=13)
    parser.add_argument("--frame-sampling-seed", type=int)
    parser.add_argument("--rare-action-frame-rate", type=float, default=1.0)
    parser.add_argument("--state-change-frame-rate", type=float, default=0.5)
    parser.add_argument("--ordinary-frame-rate", type=float, default=0.05)
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
    sampling_plan = (
        FrameSamplingPlan(
            seed=args.frame_sampling_seed,
            rare_action_rate=args.rare_action_frame_rate,
            state_change_rate=args.state_change_frame_rate,
            ordinary_rate=args.ordinary_frame_rate,
        )
        if args.frame_sampling_seed is not None
        else None
    )
    frame_counts = {name: {"seen": 0, "kept": 0} for name in SAMPLING_STRATA}
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
            args.model_format,
            sampling_plan,
            True,
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
            "frameSampling": sampling_plan.metadata() if sampling_plan else None,
            "frameCounts": frame_counts,
            "failures": failures,
            "elapsedSeconds": time.perf_counter() - start,
        }
        temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, summary_path)
        finally:
            temporary.unlink(missing_ok=True)
        return summary

    def collect(
        result: tuple[int, str, int, dict[str, dict[str, int]], str | None],
    ) -> None:
        nonlocal converted_games, converted_samples
        _index, source_id, samples, game_counts, error = result
        if error is None:
            converted_games += 1
            converted_samples += samples
            for name, counts in game_counts.items():
                frame_counts[name]["seen"] += counts["seen"]
                frame_counts[name]["kept"] += counts["kept"]
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
                    concurrent.futures.Future[
                        tuple[int, str, int, dict[str, dict[str, int]], str | None]
                    ]
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
