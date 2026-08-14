from __future__ import annotations

from functools import lru_cache

import numpy as np

TERMINALS = frozenset({0, 8, 9, 17, 18, 26})
HONORS = frozenset(range(27, 34))
YAOCHUU = TERMINALS | HONORS
DRAGONS = frozenset({31, 32, 33})
WINDS = frozenset({27, 28, 29, 30})
GREEN = frozenset({19, 20, 21, 23, 25, 32})


def _is_kokushi(hand: tuple[int, ...]) -> bool:
    return all(hand[tile] for tile in YAOCHUU) and sum(hand[tile] for tile in YAOCHUU) == 14


def _is_chiitoi(hand: tuple[int, ...]) -> bool:
    return sum(count == 2 for count in hand) == 7 and sum(hand) == 14


@lru_cache(maxsize=131_072)
def _mentsu_divisions(
    hand: tuple[int, ...],
    needed: int,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    if needed == 0:
        return (((), ()),) if not any(hand) else ()
    try:
        tile = next(index for index, count in enumerate(hand) if count)
    except StopIteration:
        return ()

    counts = list(hand)
    divisions: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    if counts[tile] >= 3:
        counts[tile] -= 3
        for triplets, sequences in _mentsu_divisions(tuple(counts), needed - 1):
            divisions.append(((tile, *triplets), sequences))
        counts[tile] += 3
    if tile < 27 and tile % 9 <= 6 and counts[tile + 1] and counts[tile + 2]:
        counts[tile] -= 1
        counts[tile + 1] -= 1
        counts[tile + 2] -= 1
        for triplets, sequences in _mentsu_divisions(tuple(counts), needed - 1):
            divisions.append((triplets, (tile, *sequences)))
    return tuple(divisions)


def _standard_divisions(
    hand: tuple[int, ...],
    needed: int,
) -> tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]:
    divisions: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
    for pair, count in enumerate(hand):
        if count < 2:
            continue
        remainder = list(hand)
        remainder[pair] -= 2
        for triplets, sequences in _mentsu_divisions(tuple(remainder), needed):
            divisions.append((pair, triplets, sequences))
    return tuple(divisions)


def _ron_opens_triplet(
    winning_tile: int,
    triplets: tuple[int, ...],
    sequences: tuple[int, ...],
) -> bool:
    if winning_tile not in triplets:
        return False
    if winning_tile >= 27:
        return True
    number = winning_tile % 9
    low = winning_tile - min(2, number)
    high = winning_tile + min(6 - number, 2)
    return not any(low <= start <= high and start <= winning_tile <= start + 2 for start in sequences)


def _is_pinfu(
    pair: int,
    sequences: tuple[int, ...],
    bakaze: int,
    jikaze: int,
    winning_tile: int,
) -> bool:
    if len(sequences) != 4 or pair in DRAGONS or pair in {bakaze, jikaze}:
        return False
    for start in sequences:
        number = start % 9
        if number <= 5 and winning_tile == start:
            return True
        if number >= 1 and winning_tile == start + 2:
            return True
    return False


def _division_has_yaku(
    pair: int,
    concealed_triplets: tuple[int, ...],
    concealed_sequences: tuple[int, ...],
    *,
    is_menzen: bool,
    chis: tuple[int, ...],
    pons: tuple[int, ...],
    minkans: tuple[int, ...],
    ankans: tuple[int, ...],
    bakaze: int,
    jikaze: int,
    winning_tile: int,
) -> bool:
    triplets = concealed_triplets + pons + minkans + ankans
    sequences = concealed_sequences + chis
    grouped = triplets + sequences + (pair,)

    if is_menzen and _is_pinfu(pair, concealed_sequences, bakaze, jikaze, winning_tile):
        return True
    if all(0 < start % 9 < 6 for start in sequences) and all(
        tile < 27 and tile % 9 not in {0, 8} for tile in triplets + (pair,)
    ):
        return True
    if not sequences:
        return True

    suits = {tile // 9 for tile in grouped if tile < 27}
    if len(suits) <= 1:
        return True

    if is_menzen:
        sequence_counts = {start: concealed_sequences.count(start) for start in concealed_sequences}
        if any(count >= 2 for count in sequence_counts.values()):
            return True

    sequence_starts = set(sequences)
    for suit in range(3):
        base = suit * 9
        if {base, base + 3, base + 6}.issubset(sequence_starts):
            return True
    for number in range(7):
        if all(suit * 9 + number in sequence_starts for suit in range(3)):
            return True
    triplet_tiles = set(triplets)
    for number in range(9):
        if all(suit * 9 + number in triplet_tiles for suit in range(3)):
            return True

    closed_triplet_count = len(ankans) + len(concealed_triplets)
    if _ron_opens_triplet(winning_tile, concealed_triplets, concealed_sequences):
        closed_triplet_count -= 1
    if closed_triplet_count >= 3 or len(ankans) + len(minkans) >= 3:
        return True
    if all(tile in GREEN for tile in triplets + (pair,)) and all(start == 19 for start in sequences):
        return True

    honor_triplets = set(triplets) & HONORS
    if honor_triplets & ({bakaze, jikaze} | DRAGONS):
        return True
    dragon_triplets = len(honor_triplets & DRAGONS)
    if dragon_triplets >= 2 and (dragon_triplets == 3 or pair in DRAGONS):
        return True
    wind_triplets = len(honor_triplets & WINDS)
    if wind_triplets == 4 or wind_triplets == 3 and pair in WINDS:
        return True

    triplet_side = triplets + (pair,)
    return all(tile in YAOCHUU for tile in triplet_side) and (
        not sequences or all(start % 9 in {0, 6} for start in sequences)
    )


def has_ron_yaku(
    hand: np.ndarray,
    *,
    chis: list[int] | tuple[int, ...] = (),
    pons: list[int] | tuple[int, ...] = (),
    minkans: list[int] | tuple[int, ...] = (),
    ankans: list[int] | tuple[int, ...] = (),
    bakaze: int,
    jikaze: int,
    winning_tile: int,
) -> bool:
    counts = tuple(int(value) for value in hand)
    chis = tuple(int(value) for value in chis)
    pons = tuple(int(value) for value in pons)
    minkans = tuple(int(value) for value in minkans)
    ankans = tuple(int(value) for value in ankans)
    is_menzen = not (chis or pons or minkans)
    if is_menzen and (_is_kokushi(counts) or _is_chiitoi(counts)):
        return True
    needed = 4 - len(chis) - len(pons) - len(minkans) - len(ankans)
    return any(
        _division_has_yaku(
            pair,
            triplets,
            sequences,
            is_menzen=is_menzen,
            chis=chis,
            pons=pons,
            minkans=minkans,
            ankans=ankans,
            bakaze=bakaze,
            jikaze=jikaze,
            winning_tile=winning_tile,
        )
        for pair, triplets, sequences in _standard_divisions(counts, needed)
    )


def is_complete_hand(hand: np.ndarray, open_melds: int) -> bool:
    counts = tuple(int(value) for value in hand)
    if open_melds == 0 and (_is_kokushi(counts) or _is_chiitoi(counts)):
        return True
    return bool(_standard_divisions(counts, 4 - open_melds))
