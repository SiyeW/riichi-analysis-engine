import importlib.util
import io
import json
import shutil
import subprocess
import sys
import uuid
import zipfile
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from test_dataset import pack_directory
from test_storage import sample_arrays

from riichi_analysis_engine.model_input import SHARED_MODEL_INPUT_CHANNELS
from riichi_analysis_engine.packing import (
    MANIFEST_FORMAT,
    PackSlot,
    audit_packs,
    build_pack,
    load_slot,
    nonadjacent_order,
    pack_slots,
    plan_corpus,
    read_plan,
    seam_game,
    staged_games,
    validate_catalog_ownership,
    write_manifest,
    write_plan,
)
from riichi_analysis_engine.semantic_input import encode_public_event
from riichi_analysis_engine.storage import (
    LEGACY_STAGED_GAME_FORMATS,
    PACKED_METADATA_FIELDS,
    TRAINING_TARGET_SCHEMA_ID,
    read_packed_shard,
    save_chunk_archive,
    write_packed_shard,
)

MERGE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_packs.py"
MERGE_SPEC = importlib.util.spec_from_file_location("merge_packs", MERGE_SCRIPT)
assert MERGE_SPEC is not None and MERGE_SPEC.loader is not None
merge_packs = importlib.util.module_from_spec(MERGE_SPEC)
sys.modules[MERGE_SPEC.name] = merge_packs
MERGE_SPEC.loader.exec_module(merge_packs)

CHECKER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_packs.py"
CHECKER_SPEC = importlib.util.spec_from_file_location("check_packs", CHECKER_SCRIPT)
assert CHECKER_SPEC is not None and CHECKER_SPEC.loader is not None
checker = importlib.util.module_from_spec(CHECKER_SPEC)
sys.modules[CHECKER_SPEC.name] = checker
CHECKER_SPEC.loader.exec_module(checker)


def test_stage_audit_conserves_samples_and_rejects_duplicate_analysis(
    scratch: Path,
) -> None:
    path = scratch / "stage" / "game-000000.zip"
    arrays = sample_arrays(4)
    arrays["event_index"] = np.asarray([0, 0, 1, 1], dtype=np.int32)
    arrays["analysis_active"] = np.asarray([True, False, True, False])
    save_chunk_archive(path, arrays, 2)
    updates = []
    report = checker.check_stage(
        [path],
        require_analysis_active=True,
        progress=lambda completed, total: updates.append((completed, total)),
    )
    assert report["samples"] == 4
    assert report["chunks"] == 2
    assert report["analysisRows"]["analysisRows"] == 2
    assert report["analysisRows"]["frames"] == 2
    assert updates == [(1, 1)]
    arrays["analysis_active"][1] = True
    save_chunk_archive(path, arrays, 2)
    with pytest.raises(SystemExit, match="duplicate analysis rows"):
        checker.check_stage([path], require_analysis_active=True)


def test_checker_cli_progress_is_separate_from_verified_report(scratch: Path) -> None:
    output = pack_directory(
        scratch, games=4, chunks=2, samples=4, training_targets=True, model_format=10
    )
    stage = scratch / "stage"
    plan = plan_corpus(staged_games(stage), 314159)
    write_plan(output / "plan.npz", plan)
    progress = scratch / "audit-progress.json"
    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER_SCRIPT),
            "--packs",
            str(output),
            "--stage",
            str(stage),
            "--verify-storage",
            "--require-analysis-active",
            "--loader-samples",
            "16",
            "--batch-size",
            "4",
            "--workers",
            "2",
            "--progress",
            str(progress),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["verified"] is True
    assert report["samples"] == 32
    assert report["stage"]["samples"] == 32
    assert report["storage"]["samples"] == 32
    state = json.loads(progress.read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["completed"] == state["total"] == report["packs"]
    assert not list(scratch.glob("audit-progress.json.*.tmp"))
    records = [
        json.loads(line) for line in result.stderr.splitlines() if line.startswith("{")
    ]
    assert records[0]["phase"] == "pack-contract-and-order"
    assert records[-1]["status"] == "complete"
    # A later failing attempt must replace the previous successful progress state.
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["packs"][0]["samples"] += 1
    write_manifest(output / "manifest.json", manifest)
    failed = subprocess.run(
        [
            sys.executable,
            str(CHECKER_SCRIPT),
            "--packs",
            str(output),
            "--progress",
            str(progress),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert failed.returncode != 0
    assert json.loads(progress.read_text(encoding="utf-8"))["status"] == "failed"


def test_catalog_ownership_handles_shuffled_sparse_game_ids() -> None:
    catalog_games = np.asarray([90, 5, 31], dtype=np.uint32)
    offsets = np.asarray([0, 8, 14, 25], dtype=np.uint32)
    games = np.asarray([31, 90, 5, 31, 5], dtype=np.uint32)
    starts = np.asarray([14, 0, 8, 20, 13], dtype=np.uint32)
    lengths = np.asarray([11, 8, 6, 5, 1], dtype=np.uint16)
    validate_catalog_ownership("pack", games, catalog_games, offsets, starts, lengths)
    lengths[-1] = 2
    with pytest.raises(ValueError, match="outside source game 5"):
        validate_catalog_ownership(
            "pack", games, catalog_games, offsets, starts, lengths
        )
    games[-1] = 99
    with pytest.raises(ValueError, match="omits source game 99"):
        validate_catalog_ownership(
            "pack", games, catalog_games, offsets, starts, lengths
        )


def test_catalog_ownership_matches_per_game_reference() -> None:
    rng = np.random.default_rng(42)
    catalog_games = rng.permutation(np.arange(100, dtype=np.uint32) * 7)
    offsets = np.arange(101, dtype=np.uint32) * 10
    owners = rng.integers(0, 100, size=2000)
    games = catalog_games[owners]
    starts = offsets[owners] + rng.integers(0, 10, size=len(games)).astype(np.uint32)
    lengths = rng.integers(0, 10, size=len(games)).astype(np.uint16)
    for trial in range(10):
        trial_lengths = lengths.copy() if trial else np.zeros_like(lengths)
        if trial:
            trial_lengths[trial * 20] = 20
        valid = all(
            (starts[games == game] >= offsets[index]).all()
            and (
                starts[games == game].astype(np.int64) + trial_lengths[games == game]
                <= offsets[index + 1]
            ).all()
            for index, game in enumerate(catalog_games)
        )
        if valid:
            validate_catalog_ownership(
                "pack", games, catalog_games, offsets, starts, trial_lengths
            )
        else:
            with pytest.raises(ValueError, match="outside source game"):
                validate_catalog_ownership(
                    "pack", games, catalog_games, offsets, starts, trial_lengths
                )


@pytest.fixture()
def scratch():
    """A scratch directory inside the workspace.

    The tests cannot use the system temporary directory: the sandbox denies the
    write, and a packing test needs real files on disk.
    """

    root = (
        Path(__file__).resolve().parents[1] / "runs" / "test-packing" / uuid.uuid4().hex
    )
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def stage_corpus(
    root: Path, *, games: int = 8, chunks: int = 6, samples: int = 16
) -> Path:
    """Write a staged corpus whose every sample is traceable to its game."""

    stage = root / "stage"
    for game in range(games):
        count = chunks * samples
        arrays = sample_arrays(count, kyoku=game)
        arrays["event_index"] = np.arange(count, dtype=np.int32) + game * 1000
        save_chunk_archive(stage / f"game-{game:06d}.zip", arrays, samples)
    return stage


def pack_corpus(
    stage: Path, output: Path, plan: dict[str, np.ndarray]
) -> dict[str, object]:
    """Build the packs and the manifest the way the packing command does."""

    games = staged_games(stage)
    entries: list[dict[str, object]] = []
    for slot in pack_slots(plan["length"], 256):
        meta, _ = build_pack(games, output, plan, slot, 314159, seam_game(plan, slot))
        entries.append(meta)
    return {
        "format": MANIFEST_FORMAT,
        "seed": 314159,
        "samples": int(plan["length"].sum()),
        "chunks": len(plan["length"]),
        "sourceGames": len(np.unique(plan["source_game"])),
        "packs": entries,
    }


def test_plan_visits_every_chunk_exactly_once(scratch: Path) -> None:
    games = staged_games(stage_corpus(scratch))
    plan = plan_corpus(games, 314159)

    assert len(plan["length"]) == 48
    assert int(plan["length"].sum()) == 768
    assert sorted(
        zip(plan["source_game"].tolist(), plan["source_member"].tolist())
    ) == [(game, member) for game in range(8) for member in range(6)]
    assert set(plan["length"].tolist()) == {16}
    # The plan is a permutation, not a reshuffle of a prefix.
    assert sorted(plan["source_game"].tolist()) == sorted(
        game for game in range(8) for _ in range(6)
    )


def test_plan_keeps_archive_identity_when_an_earlier_game_is_missing(
    scratch: Path,
) -> None:
    stage = stage_corpus(scratch, games=4, chunks=2, samples=8)
    (stage / "game-000001.zip").unlink()
    games = staged_games(stage)

    plan = plan_corpus(games, 314159)

    assert sorted(np.unique(plan["source_game"]).tolist()) == [0, 2, 3]
    # File positions stay compact even though source identities do not.  In
    # particular, source game 2 is read from the second remaining archive.
    assert set(plan["source_path"][plan["source_game"] == 2].tolist()) == {1}
    packed = load_slot(games, plan, PackSlot(0, 0, len(plan["length"])))
    source_two_events = packed["event_index"][packed["source_game"] == 2]
    assert int(source_two_events.min()) >= 2000
    assert int(source_two_events.max()) < 3000


def test_load_slot_attaches_input_contract_to_legacy_staged_chunks(
    scratch: Path,
) -> None:
    stage = stage_corpus(scratch, games=1, chunks=2, samples=4)
    path = next(stage.glob("game-*.zip"))
    rewritten = path.with_suffix(".legacy.zip")
    with (
        zipfile.ZipFile(path, "r") as source,
        zipfile.ZipFile(rewritten, "w", compression=zipfile.ZIP_STORED) as destination,
    ):
        for name in source.namelist():
            if name == "meta.json":
                meta = json.loads(source.read(name))
                meta["format"] = next(iter(LEGACY_STAGED_GAME_FORMATS))
                meta.pop("modelInputSchema", None)
                meta.pop("observationChannels", None)
                destination.writestr(name, json.dumps(meta))
                continue
            with np.load(io.BytesIO(source.read(name)), allow_pickle=False) as chunk:
                legacy = {
                    field: chunk[field]
                    for field in chunk.files
                    if field not in PACKED_METADATA_FIELDS - {"storage_format"}
                }
            legacy["storage_format"] = np.asarray("dual-bitpack-sparse-float16-v4")
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **legacy)
            destination.writestr(name, buffer.getvalue())
    path.unlink()
    rewritten.rename(path)

    games = staged_games(stage)
    plan = plan_corpus(games, 314159)
    packed = load_slot(games, plan, PackSlot(0, 0, len(plan["length"])))

    assert int(packed["obs_channels"].item()) == 1028
    assert packed["model_input_schema"].item().endswith("legacy-v8")


def test_pack_deduplicates_and_rebases_per_game_event_catalogs(scratch: Path) -> None:
    stage = scratch / "stage"
    expected_catalogs = []
    for game, tile in enumerate(("1m", "2p")):
        arrays = sample_arrays(3, kyoku=game)
        arrays["history_start"] = np.zeros(3, dtype=np.uint32)
        arrays["history_length"] = np.asarray([1, 2, 3], dtype=np.uint16)
        catalog = np.stack(
            [
                encode_public_event({"type": "start_kyoku", "dora_marker": tile}),
                encode_public_event({"type": "tsumo", "actor": game, "pai": tile}),
                encode_public_event({"type": "dahai", "actor": game, "pai": tile}),
            ]
        )
        expected_catalogs.append(catalog)
        save_chunk_archive(
            stage / f"game-{game:06d}.zip",
            arrays,
            2,
            event_catalog=catalog,
        )
    paths = staged_games(stage)
    plan = plan_corpus(paths, 17)
    output = scratch / "packs"

    meta, _ = build_pack(
        paths,
        output,
        plan,
        PackSlot(0, 0, len(plan["length"])),
        17,
        None,
    )

    packed = read_packed_shard(output / str(meta["pack"]))
    np.testing.assert_array_equal(
        packed["event_catalog"], np.concatenate(expected_catalogs, axis=0)
    )
    np.testing.assert_array_equal(packed["event_catalog_games"], [0, 1])
    np.testing.assert_array_equal(packed["event_catalog_offsets"], [0, 3, 6])
    for game, lower in ((0, 0), (1, 3)):
        selected = packed["source_game"] == game
        assert (packed["history_start"][selected] >= lower).all()
        assert (packed["history_start"][selected] < lower + 3).all()
    audit = audit_packs(
        output,
        {"format": MANIFEST_FORMAT, "packs": [meta]},
        plan,
    )
    assert audit["verified"] is True


def test_pack_preserves_and_audits_v13_training_target_contract(scratch: Path) -> None:
    stage = scratch / "stage"
    for game in range(2):
        arrays = sample_arrays(4, observation_channels=SHARED_MODEL_INPUT_CHANNELS)
        save_chunk_archive(
            stage / f"game-{game:06d}.zip",
            arrays,
            2,
            training_target_schema=TRAINING_TARGET_SCHEMA_ID,
        )
    paths = staged_games(stage)
    plan = plan_corpus(paths, 17)
    output = scratch / "packs"

    meta, _ = build_pack(
        paths, output, plan, PackSlot(0, 0, len(plan["length"])), 17, None
    )
    packed = read_packed_shard(output / str(meta["pack"]))
    audit = audit_packs(output, {"format": MANIFEST_FORMAT, "packs": [meta]}, plan)

    assert meta["trainingTargetSchema"] == TRAINING_TARGET_SCHEMA_ID
    assert packed["training_target_schema"].item() == TRAINING_TARGET_SCHEMA_ID
    assert audit["trainingTargetSchema"] == TRAINING_TARGET_SCHEMA_ID


def test_external_storage_check_accepts_deduplicated_event_catalogs(
    scratch: Path,
) -> None:
    """The independent pack checker must understand non-sample catalog arrays."""

    stage = scratch / "stage"
    for game, tile in enumerate(("1m", "2p")):
        arrays = sample_arrays(3, kyoku=game)
        arrays["history_start"] = np.zeros(3, dtype=np.uint32)
        arrays["history_length"] = np.asarray([1, 2, 3], dtype=np.uint16)
        catalog = np.stack(
            [
                encode_public_event({"type": "start_kyoku", "dora_marker": tile}),
                encode_public_event({"type": "tsumo", "actor": game, "pai": tile}),
                encode_public_event({"type": "dahai", "actor": game, "pai": tile}),
            ]
        )
        save_chunk_archive(
            stage / f"game-{game:06d}.zip",
            arrays,
            2,
            event_catalog=catalog,
        )
    paths = staged_games(stage)
    plan = plan_corpus(paths, 17)
    output = scratch / "packs"
    meta, _ = build_pack(
        paths,
        output,
        plan,
        PackSlot(0, 0, len(plan["length"])),
        17,
        None,
    )

    import importlib.util

    checker_path = Path(__file__).parents[1] / "scripts" / "check_packs.py"
    spec = importlib.util.spec_from_file_location("check_packs", checker_path)
    assert spec is not None and spec.loader is not None
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)

    report = checker.check_one_pack(output / str(meta["pack"]))
    assert report["samples"] == 6


def test_plan_round_trips_through_disk(scratch: Path) -> None:
    plan = plan_corpus(staged_games(stage_corpus(scratch)), 7)
    path = scratch / "nested" / "plan.npz"
    write_plan(path, plan)

    restored = read_plan(path)
    assert set(restored) == set(plan)
    for name, value in plan.items():
        np.testing.assert_array_equal(restored[name], value)


def test_read_plan_rejects_a_foreign_format(scratch: Path) -> None:
    path = scratch / "plan.npz"
    np.savez_compressed(path, format=np.asarray("something-else"))

    with pytest.raises(ValueError, match="unsupported plan format"):
        read_plan(path)


@pytest.mark.parametrize("pack_samples", [16, 100, 256, 10_000])
def test_pack_slots_cover_the_chunk_order_exactly(
    scratch: Path, pack_samples: int
) -> None:
    plan = plan_corpus(staged_games(stage_corpus(scratch)), 314159)
    slots = pack_slots(plan["length"], pack_samples)

    assert slots[0].start == 0
    assert slots[-1].stop == len(plan["length"])
    for previous, current in pairwise(slots):
        assert previous.stop == current.start
    assert [slot.index for slot in slots] == list(range(len(slots)))
    assert all(slot.stop > slot.start for slot in slots)


def test_pack_slots_refuses_a_useless_size(scratch: Path) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        pack_slots(np.asarray([16, 16]), 0)


def widest_gap(games: np.ndarray, game: int) -> int:
    """The longest run of samples without this game, counting both ends."""

    positions = np.flatnonzero(games == game)
    return int(np.diff(np.concatenate(([-1], positions, [len(games)]))).max())


def test_nonadjacent_order_spreads_every_game() -> None:
    source = np.repeat(np.arange(4, dtype=np.uint32), 25)
    order = nonadjacent_order(source, 314159)

    assert sorted(order.tolist()) == list(range(100))
    games = source[order]
    assert not np.any(np.diff(games) == 0)
    # Evenly spread means no game may sit further than one turn from its turn.
    counts = [int(np.count_nonzero(games == game)) for game in range(4)]
    assert max(counts) - min(counts) <= 1
    np.testing.assert_array_equal(order, nonadjacent_order(source, 314159))


def test_nonadjacent_order_spreads_uneven_groups_evenly() -> None:
    # Ordering by remaining count instead of by due position would serve the
    # two large games first and push the small ones to the end of the pack.
    counts = [40, 40, 4, 4, 4, 4, 4]
    source = np.repeat(np.arange(len(counts), dtype=np.uint32), counts)
    games = source[nonadjacent_order(source, 314159)]

    assert not np.any(np.diff(games) == 0)
    for game, count in enumerate(counts):
        assert widest_gap(games, game) <= 2 * len(games) / count + 1


def test_nonadjacent_order_handles_a_game_holding_half_the_pack() -> None:
    # 20 of these 40 samples belong to one game, so every second sample has to
    # be one of its own; any other choice leaves it with no room left.
    counts = [20, 4, 4, 4, 4, 4]
    source = np.repeat(np.arange(len(counts), dtype=np.uint32), counts)
    games = source[nonadjacent_order(source, 314159)]

    assert not np.any(np.diff(games) == 0)
    positions = np.flatnonzero(games == 0)
    assert positions[0] in (0, 1)
    # Twenty samples in forty slots leave room for one gap of three at most,
    # and never for two of the game's samples next to each other.
    gaps = np.diff(positions).tolist()
    assert set(gaps) <= {2, 3}
    assert gaps.count(3) <= 1


def test_nonadjacent_order_avoids_the_previous_pack() -> None:
    source = np.repeat(np.arange(4, dtype=np.uint32), 25)
    order = nonadjacent_order(source, 314159, forbidden_first=2)

    assert int(source[order[0]]) != 2
    assert not np.any(np.diff(source[order]) == 0)


def test_nonadjacent_order_reserves_the_last_sample() -> None:
    counts = [20, 4, 4, 4, 4, 4]
    source = np.repeat(np.arange(len(counts), dtype=np.uint32), counts)

    order = nonadjacent_order(source, 314159, last_game=3)
    assert int(source[order[-1]]) == 3
    assert not np.any(np.diff(source[order]) == 0)
    assert sorted(order.tolist()) == list(range(len(source)))


def test_every_pack_ends_on_the_game_the_plan_ends_on(scratch: Path) -> None:
    # The seam between packs is derived from the plan, so a pack has to end on
    # the game its own last chunk belongs to.
    stage = stage_corpus(scratch, games=6, chunks=4, samples=8)
    output = scratch / "packs"
    games = staged_games(stage)
    plan = plan_corpus(games, 314159)
    slots = pack_slots(plan["length"], 128)

    for slot in slots:
        meta, _ = build_pack(games, output, plan, slot, 314159, seam_game(plan, slot))
        assert meta["lastGame"] == int(plan["source_game"][slot.stop - 1])


def test_nonadjacent_order_refuses_an_impossible_pack() -> None:
    # Six samples cannot keep four of one game apart, whatever the order is.
    source = np.asarray([0] * 4 + [1] * 2, dtype=np.uint32)

    with pytest.raises(RuntimeError, match="cannot avoid adjacent source games"):
        nonadjacent_order(source, 314159)


def test_packs_are_globally_mixed_and_audited(scratch: Path) -> None:
    stage = stage_corpus(scratch)
    output = scratch / "packs"
    plan = plan_corpus(staged_games(stage), 314159)
    manifest = pack_corpus(stage, output, plan)
    write_manifest(output / "manifest.json", manifest)

    report = audit_packs(output, manifest, plan)
    assert report["verified"] is True
    assert report["samples"] == 768
    assert report["packs"] == 3
    assert report["sourceGames"] == 8
    assert report["adjacentSameGamePairs"] == 0

    # Non-adjacency alone would also hold for one game per pack, which is not
    # mixing. Every pack has to draw on several games.
    for entry in manifest["packs"]:
        games = read_packed_shard(output / str(entry["pack"]))["source_game"]
        assert len(np.unique(games)) >= 2
        assert len(games) == entry["samples"]


def test_audit_rejects_packs_whose_neighbours_share_a_game(scratch: Path) -> None:
    stage = stage_corpus(scratch)
    output = scratch / "packs"
    plan = plan_corpus(staged_games(stage), 314159)
    manifest = pack_corpus(stage, output, plan)

    # Sorting each pack by source game keeps every sample but guarantees that
    # neighbouring samples come from the same game.
    for entry in manifest["packs"]:
        path = output / str(entry["pack"])
        packed = read_packed_shard(path)
        packed["source_game"] = np.sort(packed["source_game"])
        write_packed_shard(path, packed)

    with pytest.raises(ValueError, match="adjacent sample pairs share a source game"):
        audit_packs(output, manifest, plan)


def test_audit_rejects_packs_that_lose_samples(scratch: Path) -> None:
    stage = stage_corpus(scratch)
    output = scratch / "packs"
    plan = plan_corpus(staged_games(stage), 314159)
    manifest = pack_corpus(stage, output, plan)

    manifest["packs"][1]["samples"] = int(manifest["packs"][1]["samples"]) - 16
    with pytest.raises(ValueError, match="manifest says"):
        audit_packs(output, manifest, plan)


def pack_segment(
    stage: Path,
    output: Path,
    *,
    seed: int = 314159,
    pack_samples: int = 256,
    game_offset: int = 0,
    first_forbidden: int | None = None,
) -> dict[str, object]:
    """Pack one segment the way the packing command does for a split build."""

    games = staged_games(stage)
    plan = plan_corpus(games, seed, game_offset)
    write_plan(output / "plan.npz", plan)
    entries: list[dict[str, object]] = []
    for slot in pack_slots(plan["length"], pack_samples):
        forbidden = seam_game(plan, slot)
        if slot.index == 0 and first_forbidden is not None:
            forbidden = first_forbidden
        meta, _ = build_pack(games, output, plan, slot, seed, forbidden)
        entries.append(meta)
    manifest: dict[str, object] = {
        "format": MANIFEST_FORMAT,
        "stage": str(stage),
        "seed": seed,
        "packSamples": pack_samples,
        "gameOffset": game_offset,
        "samples": int(plan["length"].sum()),
        "chunks": len(plan["length"]),
        "sourceGames": len(np.unique(plan["source_game"])),
        "firstSourceGame": int(plan["source_game"].min()),
        "lastSourceGame": int(plan["source_game"].max()),
        "packs": entries,
    }
    write_manifest(output / "manifest.json", manifest)
    return manifest


def test_a_corpus_built_in_segments_merges_into_one_order(scratch: Path) -> None:
    root = scratch / "packs"
    first = pack_segment(stage_corpus(scratch, games=8), root / "a", game_offset=0)
    # The second segment continues the game numbering and keeps its first sample
    # away from the game the first segment ended on.
    pack_segment(
        stage_corpus(scratch / "second", games=8),
        root / "b",
        game_offset=8,
        first_forbidden=int(first["packs"][-1]["lastGame"]),
    )

    report = merge_packs.merge(root, ["a", "b"])
    assert report["verified"] is True
    assert report["sourceGames"] == 16
    assert report["samples"] == 768 * 2
    assert report["adjacentSameGamePairs"] == 0

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert [entry["pack"] for entry in manifest["packs"]] == [
        *[f"a/pack-{index:05d}.npz" for index in range(len(first["packs"]))],
        *[
            f"b/pack-{index:05d}.npz"
            for index in range(len(manifest["packs"]) - len(first["packs"]))
        ],
    ]
    stream = np.concatenate(
        [
            np.load(root / str(entry["pack"]), allow_pickle=False)["source_game"]
            for entry in manifest["packs"]
        ]
    )
    assert sorted(np.unique(stream).tolist()) == list(range(16))
    assert not np.any(np.diff(stream) == 0)


def make_legacy_segment(output: Path) -> None:
    """Rewrite one fixture's metadata in the format used by the lab packs."""

    plan_path = output / "plan.npz"
    plan = read_plan(plan_path)
    plan.pop("source_path")
    with plan_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            format=np.asarray("riichi-analysis-global-plan-v1"),
            **plan,
        )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format"] = "riichi-analysis-global-manifest-v1"
    manifest["sourceGames"] = int(manifest["gameOffset"]) + int(manifest["sourceGames"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_merge_accepts_existing_v1_segments(scratch: Path) -> None:
    root = scratch / "packs"
    pack_segment(stage_corpus(scratch, games=4), root / "a")
    pack_segment(stage_corpus(scratch / "second", games=4), root / "b", game_offset=4)
    make_legacy_segment(root / "a")
    make_legacy_segment(root / "b")

    report = merge_packs.merge(root, ["a", "b"])

    assert report["verified"] is True
    assert report["sourceGames"] == 8


def test_merge_rejects_a_segment_that_repeats_source_games(scratch: Path) -> None:
    root = scratch / "packs"
    pack_segment(stage_corpus(scratch, games=8), root / "a", game_offset=0)
    pack_segment(stage_corpus(scratch / "second", games=8), root / "b", game_offset=0)

    with pytest.raises(ValueError, match="repeats source game 0"):
        merge_packs.merge(root, ["a", "b"])


def test_merge_refuses_a_gap_without_moving_segment_packs(scratch: Path) -> None:
    root = scratch / "packs"
    stage = stage_corpus(scratch, games=4)
    (stage / "game-000001.zip").unlink()
    segment = root / "a"
    pack_segment(stage, segment)
    original_packs = sorted(path.name for path in segment.glob("pack-*.npz"))

    with pytest.raises(ValueError, match="missing source game 1"):
        merge_packs.merge(root, ["a"])

    assert sorted(path.name for path in segment.glob("pack-*.npz")) == original_packs
    assert not (root / "manifest.json").exists()


def test_merge_checks_the_expected_corpus_size_before_writing(scratch: Path) -> None:
    root = scratch / "packs"
    segment = root / "a"
    pack_segment(stage_corpus(scratch, games=4), segment)

    with pytest.raises(ValueError, match="holds 4 source games, expected 5"):
        merge_packs.merge(root, ["a"], expected_source_games=5)

    assert list(segment.glob("pack-*.npz"))
    assert not (root / "manifest.json").exists()


def test_audit_rejects_a_dropped_pack(scratch: Path) -> None:
    stage = stage_corpus(scratch)
    output = scratch / "packs"
    plan = plan_corpus(staged_games(stage), 314159)
    manifest = pack_corpus(stage, output, plan)

    manifest["packs"] = manifest["packs"][:-1]
    with pytest.raises(ValueError, match="the plan holds"):
        audit_packs(output, manifest, plan)
