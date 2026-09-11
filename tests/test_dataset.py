import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest
from test_storage import sample_arrays

from riichi_analysis_engine.dataset import METADATA_FIELDS, PackDataset, read_manifest
from riichi_analysis_engine.packing import (
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

    root = Path(__file__).resolve().parents[1] / "runs" / "test-dataset" / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def pack_directory(
    root: Path, *, games: int = 4, chunks: int = 4, samples: int = 8, pack_samples: int = 32
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


def test_a_full_pass_yields_every_sample_once(scratch: Path) -> None:
    root = pack_directory(scratch)
    batches = collect(PackDataset(root, batch_size=8))

    seen = np.concatenate([batch["sample_tag"] for batch in batches])
    assert sorted(seen.tolist()) == sorted(stored_tags(root).tolist())
    assert len(seen) == read_manifest(root)["samples"] == 128


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
