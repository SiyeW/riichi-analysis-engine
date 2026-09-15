import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest
from test_storage import sample_arrays

from riichi_analysis_engine import dataset
from riichi_analysis_engine.dataset import (
    METADATA_FIELDS,
    PackDataset,
    _settle_legacy_terminal_scores,
    read_manifest,
)
from riichi_analysis_engine.packing import (
    LEGACY_MANIFEST_FORMATS,
    MANIFEST_FORMAT,
    build_pack,
    pack_slots,
    plan_corpus,
    seam_game,
    staged_games,
    write_manifest,
)
from riichi_analysis_engine.storage import save_chunk_archive


@pytest.fixture()
def scratch() -> Path:
    """A scratch directory inside the workspace.

    The tests cannot use the system temporary directory: the sandbox denies the
    write, and a packing test needs real files on disk.
    """

    root = (
        Path(__file__).resolve().parents[1] / "runs" / "test-dataset" / uuid.uuid4().hex
    )
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def pack_directory(
    root: Path,
    *,
    games: int = 4,
    chunks: int = 4,
    samples: int = 8,
    pack_samples: int = 32,
    training_targets: bool = False,
) -> Path:
    """Stage a corpus, pack it, and return the pack directory."""

    stage = root / "stage"
    for game in range(games):
        count = chunks * samples
        arrays = sample_arrays(count, kyoku=game)
        arrays["event_index"] = np.arange(count, dtype=np.int32) + game * 1000
        # A target-shaped array whose values identify one sample out of the
        # whole corpus, the way the real targets are shaped.
        arrays["sample_tag"] = np.arange(count, dtype=np.int32) + game * 1000
        if training_targets:
            arrays.update(
                {
                    "policy": np.zeros(count, dtype=np.int8),
                    "shanten": np.zeros((count, 3), dtype=np.int8),
                    "furiten_no_yaku": np.zeros((count, 3), dtype=np.float32),
                    "deal_in_tile": np.zeros((count, 3, 34), dtype=np.float32),
                    "concealed_count": np.zeros((count, 3, 34), dtype=np.int8),
                    "concealed_red_count": np.zeros((count, 3, 3), dtype=np.int8),
                    "wall_count": np.zeros((count, 34), dtype=np.int8),
                    "wall_red_count": np.zeros((count, 3), dtype=np.int8),
                    "dora": np.zeros((count, 3), dtype=np.int8),
                    "score": np.zeros((count, 3), dtype=np.int32),
                    "winner_mask": np.zeros((count, 3), dtype=bool),
                    "outcome": np.zeros(count, dtype=np.int8),
                    "draw": np.ones(count, dtype=np.float32),
                    "win": np.zeros((count, 4), dtype=np.float32),
                    "deal_in_player": np.zeros((count, 4), dtype=np.float32),
                    "kyoku_delta": np.zeros((count, 4), dtype=np.float32),
                    "placement": np.zeros(count, dtype=np.int8),
                    "match_score": np.zeros((count, 4), dtype=np.float32),
                }
            )
        save_chunk_archive(stage / f"game-{game:06d}.zip", arrays, samples)

    output = root / "packs"
    paths = staged_games(stage)
    plan = plan_corpus(paths, 20252026)
    entries = []
    for slot in pack_slots(plan["length"], pack_samples):
        meta, _ = build_pack(paths, output, plan, slot, 20252026, seam_game(plan, slot))
        entries.append(meta)
    write_manifest(
        output / "manifest.json",
        {
            "format": MANIFEST_FORMAT,
            "stage": str(stage),
            "seed": 20252026,
            "samples": int(plan["length"].sum()),
            "chunks": len(plan["length"]),
            "sourceGames": games,
            "packs": entries,
        },
    )
    return output


def stored_tags(root: Path) -> np.ndarray:
    """Every sample tag the packs hold, read straight from the files."""

    return np.concatenate(
        [
            np.load(root / str(entry["pack"]), allow_pickle=False)["sample_tag"]
            for entry in read_manifest(root)["packs"]
        ]
    )


def collect(dataset: PackDataset) -> list[dict[str, np.ndarray]]:
    return [{name: value.numpy() for name, value in batch.items()} for batch in dataset]


def test_legacy_terminal_score_migration_awards_pool_to_absolute_first_place() -> None:
    arrays = {
        "match_score": np.asarray(
            [
                [17_600, 10_100, 44_300, 27_000],
                [10_100, 44_300, 27_000, 17_600],
            ],
            dtype=np.int32,
        ),
        "perspective": np.asarray([0, 1], dtype=np.uint8),
    }

    _settle_legacy_terminal_scores(arrays)

    assert arrays["match_score"].tolist() == [
        [17_600, 10_100, 45_300, 27_000],
        [10_100, 45_300, 27_000, 17_600],
    ]


def test_a_full_pass_yields_every_sample_once(scratch: Path) -> None:
    root = pack_directory(scratch)
    batches = collect(PackDataset(root, batch_size=8))

    seen = np.concatenate([batch["sample_tag"] for batch in batches])
    assert sorted(seen.tolist()) == sorted(stored_tags(root).tolist())
    assert len(seen) == read_manifest(root)["samples"] == 128


def test_legacy_pack_manifest_remains_readable(scratch: Path) -> None:
    root = pack_directory(scratch)
    manifest_path = root / "manifest.json"
    manifest = read_manifest(root)
    manifest["format"] = next(iter(LEGACY_MANIFEST_FORMATS))
    write_manifest(manifest_path, manifest)

    batches = collect(PackDataset(root, batch_size=8))

    assert sum(len(batch["policy"]) for batch in batches) == 128


def test_batches_are_whole_and_only_the_last_one_may_be_short(scratch: Path) -> None:
    root = pack_directory(scratch)
    batches = collect(PackDataset(root, batch_size=8))

    assert [len(batch["policy"]) for batch in batches] == [8] * 16
    for batch in batches:
        assert len({len(value) for value in batch.values()}) == 1


def test_a_short_pack_carries_its_remainder_into_the_next_pack(scratch: Path) -> None:
    # 32 samples per pack do not divide into batches of five, so the two
    # samples left over at the end of a pack have to start the next batch.
    root = pack_directory(scratch)
    batches = collect(PackDataset(root, batch_size=5))

    assert [len(batch["policy"]) for batch in batches] == [5] * 25 + [3]
    seen = np.concatenate([batch["sample_tag"] for batch in batches])
    assert sorted(seen.tolist()) == sorted(stored_tags(root).tolist())


def test_max_samples_stops_the_pass_exactly(scratch: Path) -> None:
    root = pack_directory(scratch)
    batches = collect(PackDataset(root, batch_size=8, max_samples=20))

    assert [len(batch["policy"]) for batch in batches] == [8, 8, 4]


def test_max_samples_does_not_emit_an_empty_batch(scratch: Path) -> None:
    # A cap that lands exactly on a batch boundary used to leave the loop with
    # nothing left to take and yield a batch of no samples at all.
    root = pack_directory(scratch)
    batches = collect(PackDataset(root, batch_size=8, max_samples=32))

    assert [len(batch["policy"]) for batch in batches] == [8, 8, 8, 8]
    assert all(len(batch["policy"]) for batch in batches)


def test_starting_from_a_sample_matches_the_tail_of_the_pass(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = pack_directory(scratch)
    full = collect(PackDataset(root, batch_size=8))
    reference = np.concatenate([batch["sample_tag"] for batch in full])

    # A resumed pass must not open the packs it skips, not merely discard their
    # samples, so the packs it reads are recorded.
    opened: list[str] = []
    original = dataset.read_packed_shard

    def recording(path: Path) -> dict[str, np.ndarray]:
        opened.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(dataset, "read_packed_shard", recording)
    skipped = collect(PackDataset(root, batch_size=8, start_sample=64))

    assert opened == ["pack-00002.npz", "pack-00003.npz"]
    np.testing.assert_array_equal(
        np.concatenate([batch["sample_tag"] for batch in skipped]), reference[64:]
    )


def test_sample_cursor_is_exact_after_a_short_budget_batch(scratch: Path) -> None:
    root = pack_directory(scratch)
    full = np.concatenate(
        [batch["sample_tag"] for batch in collect(PackDataset(root, batch_size=8))]
    )

    prefix = collect(PackDataset(root, batch_size=8, max_samples=13))
    suffix = collect(PackDataset(root, batch_size=8, start_sample=13))

    assert [len(batch["policy"]) for batch in prefix] == [8, 5]
    np.testing.assert_array_equal(
        np.concatenate([batch["sample_tag"] for batch in prefix + suffix]), full
    )


def test_sample_budget_stops_inside_a_batch_carried_between_packs(
    scratch: Path,
) -> None:
    root = pack_directory(scratch, pack_samples=10)
    full = np.concatenate(
        [batch["sample_tag"] for batch in collect(PackDataset(root, batch_size=8))]
    )
    prefix = collect(PackDataset(root, batch_size=8, max_samples=13))

    assert [len(batch["policy"]) for batch in prefix] == [8, 5]
    np.testing.assert_array_equal(
        np.concatenate([batch["sample_tag"] for batch in prefix]), full[:13]
    )


def test_metadata_stays_out_of_the_batches(scratch: Path) -> None:
    root = pack_directory(scratch)
    batch = next(iter(PackDataset(root, batch_size=4)))

    assert not METADATA_FIELDS.intersection(batch)
    assert {"obs", "action_mask", "policy", "wall_count"}.issubset(batch)
    assert batch["obs"].shape[1:] == (1028, 34)


def test_workers_split_the_packs_without_gaps_or_repeats(scratch: Path) -> None:
    root = pack_directory(scratch)
    dataset = PackDataset(root, batch_size=8)

    selected = [dataset.selected_packs(worker, 3) for worker in range(3)]
    assert sorted(path for group in selected for path in group) == dataset.packs
    assert len({path for group in selected for path in group}) == len(dataset.packs)
    for group in selected:
        assert [path for path in dataset.packs if path in set(group)] == group
    with pytest.raises(ValueError, match="worker id"):
        dataset.selected_packs(3, 3)


def test_a_directory_without_a_manifest_is_rejected(scratch: Path) -> None:
    empty = scratch / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match="manifest"):
        PackDataset(empty, batch_size=4)


def test_a_manifest_pointing_at_a_missing_pack_is_rejected(scratch: Path) -> None:
    root = pack_directory(scratch)
    (root / "pack-00001.npz").unlink()

    with pytest.raises(FileNotFoundError, match="missing"):
        PackDataset(root, batch_size=8)


def test_a_foreign_manifest_is_rejected(scratch: Path) -> None:
    root = scratch / "foreign"
    root.mkdir()
    (root / "manifest.json").write_text(
        '{"format": "something-else", "packs": []}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unsupported manifest format"):
        PackDataset(root, batch_size=4)
