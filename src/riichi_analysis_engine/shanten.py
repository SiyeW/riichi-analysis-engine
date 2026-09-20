"""Exact shanten calculation backed by the proven libriichi lookup tables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

_TABLE_DIR = Path(__file__).with_name("data")
_SUHAI = np.load(_TABLE_DIR / "shanten_suhai.npy", mmap_mode="r")
_JIHAI = np.load(_TABLE_DIR / "shanten_jihai.npy", mmap_mode="r")
_KOKUSHI = np.asarray(
    [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33], dtype=np.int8
)


def _base_five_index(counts: np.ndarray) -> int:
    result = 0
    for count in counts:
        result = result * 5 + int(count)
    return result


def _add_suit(accumulator: np.ndarray, index: int, groups: int) -> None:
    table = _SUHAI[index]
    for target in range(5 + groups, 4, -1):
        value = min(
            accumulator[target] + table[0],
            accumulator[0] + table[target],
        )
        for split in range(5, target):
            value = min(
                value,
                accumulator[split] + table[target - split],
                accumulator[target - split] + table[split],
            )
        accumulator[target] = value
    for target in range(groups, -1, -1):
        value = accumulator[target] + table[0]
        for split in range(target):
            value = min(value, accumulator[split] + table[target - split])
        accumulator[target] = value


def _add_honors(accumulator: np.ndarray, index: int, groups: int) -> None:
    table = _JIHAI[index]
    target = groups + 5
    value = min(accumulator[target] + table[0], accumulator[0] + table[target])
    for split in range(5, target):
        value = min(
            value,
            accumulator[split] + table[target - split],
            accumulator[target - split] + table[split],
        )
    accumulator[target] = value


def _normal_shanten(counts: np.ndarray, concealed_groups: int) -> int:
    result = np.asarray(_SUHAI[_base_five_index(counts[:9])]).copy()
    _add_suit(result, _base_five_index(counts[9:18]), concealed_groups)
    _add_suit(result, _base_five_index(counts[18:27]), concealed_groups)
    _add_honors(result, _base_five_index(counts[27:]), concealed_groups)
    return int(result[5 + concealed_groups]) - 1


def _chiitoi_shanten(counts: np.ndarray) -> int:
    pairs = int(np.sum(counts >= 2))
    kinds = int(np.sum(counts > 0))
    return 7 - pairs + max(0, 7 - kinds) - 1


def _kokushi_shanten(counts: np.ndarray) -> int:
    terminals = counts[_KOKUSHI]
    return 13 - int(np.sum(terminals > 0)) - int(np.any(terminals >= 2))


@lru_cache(maxsize=131_072)
def _cached_shanten(encoded: bytes, concealed_groups: int, closed: bool) -> int:
    counts = np.frombuffer(encoded, dtype=np.uint8)
    result = _normal_shanten(counts, concealed_groups)
    if closed:
        result = min(result, _chiitoi_shanten(counts), _kokushi_shanten(counts))
    return result


def calculate_shanten(counts: np.ndarray, open_melds: int = 0) -> int:
    """Return -1 for complete, 0 for tenpai, then ordinary shanten values."""

    hand = np.asarray(counts, dtype=np.uint8)
    if hand.shape != (34,):
        raise ValueError(f"hand must contain 34 counts, got {hand.shape}")
    if np.any(hand > 4):
        raise ValueError("a tile count cannot exceed four")
    concealed_groups = int(hand.sum()) // 3
    return _cached_shanten(hand.tobytes(), concealed_groups, open_melds == 0)


def calculate_shanten_batch(counts: np.ndarray, open_melds: int = 0) -> np.ndarray:
    """Vectorized equivalent used for the bounded draw/discard query grid."""

    hands = np.asarray(counts, dtype=np.uint8)
    if hands.ndim != 2 or hands.shape[1] != 34:
        raise ValueError(f"hands must have shape (n, 34), got {hands.shape}")
    if np.any(hands > 4):
        raise ValueError("a tile count cannot exceed four")
    if len(hands) == 0:
        return np.empty(0, dtype=np.int8)
    group_counts = hands.sum(axis=1) // 3
    if np.any(group_counts != group_counts[0]):
        return np.asarray(
            [calculate_shanten(hand, open_melds) for hand in hands], dtype=np.int8
        )
    groups = int(group_counts[0])
    indices = np.asarray(
        [[_base_five_index(hand[start : start + width]) for hand in hands]
         for start, width in ((0, 9), (9, 9), (18, 9), (27, 7))]
    )
    accumulator = np.asarray(_SUHAI[indices[0]]).copy()

    def add_suit(table: np.ndarray) -> None:
        for target in range(5 + groups, 4, -1):
            value = np.minimum(
                accumulator[:, target] + table[:, 0],
                accumulator[:, 0] + table[:, target],
            )
            for split in range(5, target):
                value = np.minimum(
                    value,
                    np.minimum(
                        accumulator[:, split] + table[:, target - split],
                        accumulator[:, target - split] + table[:, split],
                    ),
                )
            accumulator[:, target] = value
        for target in range(groups, -1, -1):
            value = accumulator[:, target] + table[:, 0]
            for split in range(target):
                value = np.minimum(
                    value, accumulator[:, split] + table[:, target - split]
                )
            accumulator[:, target] = value

    add_suit(np.asarray(_SUHAI[indices[1]]))
    add_suit(np.asarray(_SUHAI[indices[2]]))
    honors = np.asarray(_JIHAI[indices[3]])
    target = groups + 5
    value = np.minimum(
        accumulator[:, target] + honors[:, 0],
        accumulator[:, 0] + honors[:, target],
    )
    for split in range(5, target):
        value = np.minimum(
            value,
            np.minimum(
                accumulator[:, split] + honors[:, target - split],
                accumulator[:, target - split] + honors[:, split],
            ),
        )
    accumulator[:, target] = value
    result = accumulator[:, 5 + groups].astype(np.int8) - 1
    if open_melds == 0:
        pairs = np.sum(hands >= 2, axis=1)
        kinds = np.sum(hands > 0, axis=1)
        chiitoi = 7 - pairs + np.maximum(0, 7 - kinds) - 1
        terminals = hands[:, _KOKUSHI]
        kokushi = 13 - np.sum(terminals > 0, axis=1) - np.any(
            terminals >= 2, axis=1
        )
        result = np.minimum(result, np.minimum(chiitoi, kokushi)).astype(np.int8)
    return result
