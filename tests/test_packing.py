import importlib.util
import json
import shutil
import sys
import uuid
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from test_storage import sample_arrays

from riichi_analysis_engine.packing import (
    MANIFEST_FORMAT,
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
from riichi_analysis_engine.storage import (
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


@pytest.fixture()
def scratch():
    """A scratch directory inside the workspace.

    The tests cannot use the system temporary directory: the sandbox denies the
    write, and a packing test needs real files on disk.
    """

    root = Path(__file__).resolve().parents[1] / "runs" / "test-packing" / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def stage_corpus(root: Path, *, games: int = 8, chunks: int = 6, samples: int = 16) -> Path:
    """Write a staged corpus whose every sample is traceable to its game."""

    stage = root / "stage"
    for game in range(games):
        count = chunks * samples
        arrays = sample_arrays(count, kyoku=game)
        arrays["event_index"] = np.arange(count, dtype=np.int32) + game * 1000
        save_chunk_archive(stage / f"game-{game:06d}.zip", arrays, samples)
    return stage


def pack_corpus(stage: Path, output: Path, plan: dict[str, np.ndarray]) -> dict[str, object]:
    """Build the packs and the manifest the way the packing command does."""

    games = staged_games(stage)
    entries: list[dict[str, object]] = []
    for slot in pack_slots(plan["length"], 256):
        meta, _ = build_pack(games, output, plan, slot, 20252026, seam_game(plan, slot))
        entries.append(meta)
    return {
        "format": MANIFEST_FORMAT,
        "seed": 20252026,
        "samples": int(plan["length"].sum()),
        "chunks": len(plan["length"]),
        "sourceGames": int(plan["source_game"].max()) + 1,
        "packs": entries,
    }


def test_plan_visits_every_chunk_exactly_once(scratch: Path) -> None:
    games = staged_games(stage_corpus(scratch))
    plan = plan_corpus(games, 20252026)

    assert len(plan["length"]) == 48
    assert int(plan["length"].sum()) == 768
    assert sorted(zip(plan["source_game"].tolist(), plan["source_member"].tolist())) == [
        (game, member) for game in range(8) for member in range(6)
    ]
    assert set(plan["length"].tolist()) == {16}
    # The plan is a permutation, not a reshuffle of a prefix.
    assert sorted(plan["source_game"].tolist()) == sorted(
        game for game in range(8) for _ in range(6)
    )


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
def test_pack_slots_cover_the_chunk_order_exactly(scratch: Path, pack_samples: int) -> None:
    plan = plan_corpus(staged_games(stage_corpus(scratch)), 20252026)
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
    order = nonadjacent_order(source, 20252026)

    assert sorted(order.tolist()) == list(range(100))
    games = source[order]
    assert not np.any(np.diff(games) == 0)
    # Evenly spread means no game may sit further than one turn from its turn.
    counts = [int(np.count_nonzero(games == game)) for game in range(4)]
    assert max(counts) - min(counts) <= 1
    np.testing.assert_array_equal(order, nonadjacent_order(source, 20252026))


def test_nonadjacent_order_spreads_uneven_groups_evenly() -> None:
    # Ordering by remaining count instead of by due position would serve the
    # two large games first and push the small ones to the end of the pack.
    counts = [40, 40, 4, 4, 4, 4, 4]
    source = np.repeat(np.arange(len(counts), dtype=np.uint32), counts)
    games = source[nonadjacent_order(source, 20252026)]

    assert not np.any(np.diff(games) == 0)
    for game, count in enumerate(counts):
        assert widest_gap(games, game) <= 2 * len(games) / count + 1


def test_nonadjacent_order_handles_a_game_holding_half_the_pack() -> None:
    # 20 of these 40 samples belong to one game, so every second sample has to
    # be one of its own; any other choice leaves it with no room left.
    counts = [20, 4, 4, 4, 4, 4]
    source = np.repeat(np.arange(len(counts), dtype=np.uint32), counts)
    games = source[nonadjacent_order(source, 20252026)]

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
    order = nonadjacent_order(source, 20252026, forbidden_first=2)

    assert int(source[order[0]]) != 2
    assert not np.any(np.diff(source[order]) == 0)


def test_nonadjacent_order_reserves_the_last_sample() -> None:
    counts = [20, 4, 4, 4, 4, 4]
    source = np.repeat(np.arange(len(counts), dtype=np.uint32), counts)

    order = nonadjacent_order(source, 20252026, last_game=3)
    assert int(source[order[-1]]) == 3
    assert not np.any(np.diff(source[order]) == 0)
    assert sorted(order.tolist()) == list(range(len(source)))


def test_every_pack_ends_on_the_game_the_plan_ends_on(scratch: Path) -> None:
    # The seam between packs is derived from the plan, so a pack has to end on
    # the game its own last chunk belongs to.
    stage = stage_corpus(scratch, games=6, chunks=4, samples=8)
    output = scratch / "packs"
    games = staged_games(stage)
    plan = plan_corpus(games, 20252026)
    slots = pack_slots(plan["length"], 128)

    for slot in slots:
        meta, _ = build_pack(games, output, plan, slot, 20252026, seam_game(plan, slot))
        assert meta["lastGame"] == int(plan["source_game"][slot.stop - 1])


def test_nonadjacent_order_refuses_an_impossible_pack() -> None:
    # Six samples cannot keep four of one game apart, whatever the order is.
    source = np.asarray([0] * 4 + [1] * 2, dtype=np.uint32)

    with pytest.raises(RuntimeError, match="cannot avoid adjacent source games"):
        nonadjacent_order(source, 20252026)


def test_packs_are_globally_mixed_and_audited(scratch: Path) -> None:
    stage = stage_corpus(scratch)
    output = scratch / "packs"
    plan = plan_corpus(staged_games(stage), 20252026)
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
    plan = plan_corpus(staged_games(stage), 20252026)
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
    plan = plan_corpus(staged_games(stage), 20252026)
    manifest = pack_corpus(stage, output, plan)

    manifest["packs"][1]["samples"] = int(manifest["packs"][1]["samples"]) - 16
    with pytest.raises(ValueError, match="manifest says"):
        audit_packs(output, manifest, plan)


def pack_segment(
    stage: Path,
    output: Path,
    *,
    seed: int = 20252026,
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
        "sourceGames": int(plan["source_game"].max()) + 1,
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
        f"pack-{index:05d}.npz" for index in range(len(manifest["packs"]))
    ]
    stream = np.concatenate(
        [
            np.load(root / str(entry["pack"]), allow_pickle=False)["source_game"]
            for entry in manifest["packs"]
        ]
    )
    assert sorted(np.unique(stream).tolist()) == list(range(16))
    assert not np.any(np.diff(stream) == 0)


def test_merge_rejects_a_segment_that_restarts_the_numbering(scratch: Path) -> None:
    root = scratch / "packs"
    pack_segment(stage_corpus(scratch, games=8), root / "a", game_offset=0)
    pack_segment(stage_corpus(scratch / "second", games=8), root / "b", game_offset=0)

    with pytest.raises(ValueError, match="starts at game 0"):
        merge_packs.merge(root, ["a", "b"])


def test_audit_rejects_a_dropped_pack(scratch: Path) -> None:
    stage = stage_corpus(scratch)
    output = scratch / "packs"
    plan = plan_corpus(staged_games(stage), 20252026)
    manifest = pack_corpus(stage, output, plan)

    manifest["packs"] = manifest["packs"][:-1]
    with pytest.raises(ValueError, match="the plan holds"):
        audit_packs(output, manifest, plan)
