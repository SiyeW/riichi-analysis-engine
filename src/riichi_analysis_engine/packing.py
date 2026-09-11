"""Global mixing of staged games into training packs.

Conversion stages each game as independently compressed chunks. Training reads
the packs once, in order, so the corpus has to be mixed when it is written
rather than every time it is read.

The order is produced in two steps. A seeded permutation visits the staged
chunks of the whole corpus in one random order, which by itself already makes
two neighbours from the same game unlikely. Then each pack is rearranged so
that every source game inside it is spread as evenly as possible, which turns
that likelihood into a guarantee and matters most when the corpus is small
enough that one pack holds many chunks of the same game.

Both steps are deterministic, so the same staged corpus always yields the same
order, and :func:`audit_packs` re-derives the invariants from the written packs
so training can refuse to start on an order that violates them.
"""

from __future__ import annotations

import heapq
import io
import json
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .storage import (
    STORAGE_FORMAT,
    concatenate_packed,
    permute_packed,
    read_chunk_archive_meta,
    write_packed_shard,
)

PLAN_FORMAT = "riichi-analysis-global-plan-v1"
MANIFEST_FORMAT = "riichi-analysis-global-manifest-v1"


@dataclass(frozen=True)
class PackSlot:
    """One contiguous run of the globally permuted chunk order."""

    index: int
    start: int
    stop: int


def staged_games(stage: Path) -> list[Path]:
    """Return the staged game archives in a stable order."""

    paths = sorted(stage.glob("game-*.zip"))
    if not paths:
        raise FileNotFoundError(f"no staged games under {stage}")
    return paths


def plan_corpus(game_paths: list[Path], seed: int) -> dict[str, np.ndarray]:
    """Enumerate every staged chunk and apply one seeded corpus-wide permutation."""

    games: list[int] = []
    members: list[int] = []
    lengths: list[int] = []
    for game_index, path in enumerate(game_paths):
        chunk_lengths = [int(value) for value in read_chunk_archive_meta(path)["chunkLengths"]]
        games.extend([game_index] * len(chunk_lengths))
        members.extend(range(len(chunk_lengths)))
        lengths.extend(chunk_lengths)
    if not lengths:
        raise ValueError("staged games contain no chunks")
    order = np.random.default_rng(seed).permutation(len(lengths))
    return {
        "source_game": np.asarray(games, dtype=np.uint32)[order],
        "source_member": np.asarray(members, dtype=np.uint16)[order],
        "length": np.asarray(lengths, dtype=np.uint16)[order],
    }


def write_plan(path: Path, plan: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, format=np.asarray(PLAN_FORMAT), **plan)
    temporary.replace(path)


def read_plan(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    if arrays.pop("format").item() != PLAN_FORMAT:
        raise ValueError(f"unsupported plan format in {path}")
    return arrays


def pack_slots(lengths: np.ndarray, pack_samples: int) -> list[PackSlot]:
    """Split the permuted chunk order into packs of roughly equal sample count."""

    if pack_samples <= 0:
        raise ValueError("pack size must be positive")
    if len(lengths) == 0:
        raise ValueError("the plan holds no chunks")
    slots: list[PackSlot] = []
    start = 0
    running = 0
    for index, length in enumerate(lengths):
        if running >= pack_samples and index > start:
            slots.append(PackSlot(len(slots), start, index))
            start = index
            running = 0
        running += int(length)
    slots.append(PackSlot(len(slots), start, len(lengths)))
    return slots


def seam_game(plan: dict[str, np.ndarray], slot: PackSlot) -> int | None:
    """The source game the pack before this one ended on, if there is one."""

    if slot.start == 0:
        return None
    return int(plan["source_game"][slot.start - 1])


def nonadjacent_order(
    source_game: np.ndarray, seed: int, forbidden_first: int | None = None
) -> np.ndarray:
    """Order the samples so that every source game is spread evenly.

    Each game is given a spacing of total/count samples and is served at its
    earliest due position. Serving by due position rather than by remaining
    count is what keeps a game that contributes more samples than the others
    from clustering at the front of the pack. A game that is due but is also
    the previous sample is deferred by one position, which makes two neighbours
    from the same game impossible rather than merely unlikely.

    One case overrides the due position: as soon as a single game holds half of
    the samples that are left, every second sample has to be one of its own, so
    that game is served whenever it comes due at all. Without that rule a pack
    can run out of partners for its largest game and end with two of its
    samples next to each other.
    """

    rng = np.random.default_rng(seed)
    groups: dict[int, list[int]] = {}
    for game in np.unique(source_game):
        indices = np.flatnonzero(source_game == game)
        rng.shuffle(indices)
        groups[int(game)] = indices.astype(int).tolist()

    total = len(source_game)
    largest = max(len(indices) for indices in groups.values())
    if largest > (total + 1) // 2:
        raise RuntimeError(f"cannot avoid adjacent source games: largest group {largest}/{total}")

    # Each game's first sample is staggered inside its own spacing, so a pack
    # does not always open with the same games in the same order.
    spacing = {game: total / len(indices) for game, indices in groups.items()}
    due_heap: list[tuple[float, float, int]] = [
        (float(rng.random()) * spacing[game], float(rng.random()), game) for game in groups
    ]
    heapq.heapify(due_heap)
    # A second heap tracks which game has the most samples left. Serving a game
    # makes its entry there stale, so the loop below re-pushes the current count
    # whenever it finds one.
    count_heap: list[tuple[int, int]] = [(-len(indices), game) for game, indices in groups.items()]
    heapq.heapify(count_heap)

    remaining = {game: len(indices) for game, indices in groups.items()}
    order: list[int] = []
    previous: int | None = forbidden_first
    left = total
    while left:
        while -count_heap[0][0] != remaining[count_heap[0][1]]:
            _, stale = heapq.heappop(count_heap)
            if remaining[stale]:
                heapq.heappush(count_heap, (-remaining[stale], stale))
        most = count_heap[0][1]
        # Serving any other game would leave the largest one with more samples
        # than the remaining slots can separate, so this position is its own.
        # With an even number of slots left there is always room to serve
        # somebody else first, which is what keeps the order from alternating
        # needlessly.
        if left % 2 == 1 and remaining[most] == (left + 1) // 2:
            held, entry = _take_until(due_heap, most)
        else:
            held, entry = _take_next(due_heap, previous)
        due, _, game = entry
        order.append(groups[game].pop())
        remaining[game] -= 1
        left -= 1
        previous = game
        for item in held:
            heapq.heappush(due_heap, item)
        if remaining[game]:
            heapq.heappush(due_heap, (due + spacing[game], float(rng.random()), game))
    return np.asarray(order, dtype=np.int64)


def _take_next(
    heap: list[tuple[float, float, int]], previous: int | None
) -> tuple[list[tuple[float, float, int]], tuple[float, float, int]]:
    """Pop the earliest due game that is not the previous sample."""

    held: list[tuple[float, float, int]] = []
    while heap:
        entry = heapq.heappop(heap)
        if entry[2] != previous:
            return held, entry
        held.append(entry)
    raise RuntimeError(f"cannot avoid source game {previous} at the end of the order")


def _take_until(
    heap: list[tuple[float, float, int]], game: int
) -> tuple[list[tuple[float, float, int]], tuple[float, float, int]]:
    """Pop one game's own entry, holding the entries that come before it."""

    held: list[tuple[float, float, int]] = []
    while heap:
        entry = heapq.heappop(heap)
        if entry[2] == game:
            return held, entry
        held.append(entry)
    raise RuntimeError(f"source game {game} has no entry left to serve")


def load_slot(
    game_paths: list[Path], plan: dict[str, np.ndarray], slot: PackSlot
) -> dict[str, np.ndarray]:
    """Read every chunk a slot needs, opening each staged game at most once."""

    games = plan["source_game"][slot.start : slot.stop]
    members = plan["source_member"][slot.start : slot.stop]
    lengths = plan["length"][slot.start : slot.stop]

    positions_by_game: dict[int, list[int]] = defaultdict(list)
    for position, game in enumerate(games.tolist()):
        positions_by_game[game].append(position)

    payloads: list[bytes | None] = [None] * len(games)
    for game, positions in sorted(positions_by_game.items()):
        with zipfile.ZipFile(game_paths[game], "r") as archive:
            for position in positions:
                member = int(members[position])
                payloads[position] = archive.read(f"chunk_{member:05d}.npz")

    parts = []
    source_game: list[int] = []
    for position, payload in enumerate(payloads):
        assert payload is not None
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            parts.append({name: data[name] for name in data.files})
        source_game.extend([int(games[position])] * int(lengths[position]))
    combined = concatenate_packed(parts)
    combined["source_game"] = np.asarray(source_game, dtype=np.uint32)
    return combined


def build_pack(
    game_paths: list[Path],
    output: Path,
    plan: dict[str, np.ndarray],
    slot: PackSlot,
    seed: int,
    forbidden_first: int | None,
) -> tuple[dict[str, object], int]:
    """Materialize one globally mixed pack and report the game it ends on."""

    combined = load_slot(game_paths, plan, slot)
    order = nonadjacent_order(
        combined["source_game"],
        seed=int(np.random.SeedSequence([seed, slot.index]).generate_state(1)[0]),
        forbidden_first=forbidden_first,
    )
    permuted = permute_packed(combined, order)
    permuted["pack_index"] = np.full(len(order), slot.index, dtype=np.uint32)
    name = f"pack-{slot.index:05d}.npz"
    write_packed_shard(output / name, permuted)
    meta: dict[str, object] = {
        "pack": name,
        "samples": len(order),
        "chunks": slot.stop - slot.start,
        "sourceGames": len(np.unique(combined["source_game"])),
        "firstGame": int(permuted["source_game"][0]),
        "lastGame": int(permuted["source_game"][-1]),
    }
    return meta, int(permuted["source_game"][-1])


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def audit_packs(
    output: Path, manifest: dict[str, object], plan: dict[str, np.ndarray]
) -> dict[str, object]:
    """Re-derive the training-order invariants from the written packs."""

    entries = manifest.get("packs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest lists no packs")

    expected_samples = int(plan["length"].sum())
    game_count = int(plan["source_game"].max()) + 1
    plan_samples = np.bincount(
        np.repeat(plan["source_game"], plan["length"]), minlength=game_count
    )
    seen_samples = np.zeros(game_count, dtype=np.int64)

    total = 0
    adjacent_pairs = 0
    previous_game: int | None = None
    for entry in entries:
        assert isinstance(entry, dict)
        name = str(entry["pack"])
        with np.load(output / name, allow_pickle=False) as source:
            if source["storage_format"].item() != STORAGE_FORMAT:
                raise ValueError(f"{name} declares an unsupported storage format")
            games = source["source_game"]
        if len(games) != int(entry["samples"]):
            raise ValueError(f"{name} holds {len(games)} samples, manifest says {entry['samples']}")
        if len(games) == 0:
            raise ValueError(f"{name} is empty")
        seen_samples += np.bincount(games, minlength=game_count)
        total += len(games)
        if previous_game is not None and int(games[0]) == previous_game:
            adjacent_pairs += 1
        if len(games) > 1:
            adjacent_pairs += int(np.count_nonzero(np.diff(games) == 0))
        previous_game = int(games[-1])

    if total != expected_samples:
        raise ValueError(f"packs hold {total} samples, the plan holds {expected_samples}")
    if not np.array_equal(seen_samples, plan_samples):
        raise ValueError("packs do not reproduce the planned source-game sample counts")
    if adjacent_pairs:
        raise ValueError(f"{adjacent_pairs} adjacent sample pairs share a source game")
    return {
        "format": MANIFEST_FORMAT,
        "verified": True,
        "samples": total,
        "packs": len(entries),
        "sourceGames": game_count,
        "plannedChunks": len(plan["length"]),
        "adjacentSameGamePairs": 0,
    }
