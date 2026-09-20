import numpy as np

from riichi_analysis_engine.yaku import has_ron_yaku, has_tsumo_yaku, is_complete_hand


def hand(*groups: tuple[int, int]) -> np.ndarray:
    counts = np.zeros(34, dtype=np.int16)
    for tile, count in groups:
        counts[tile] = count
    return counts


def test_complete_special_hands() -> None:
    chiitoi = hand((0, 2), (1, 2), (9, 2), (10, 2), (18, 2), (27, 2), (31, 2))
    assert is_complete_hand(chiitoi, 0)
    assert has_ron_yaku(chiitoi, bakaze=27, jikaze=28, winning_tile=31)


def test_open_hand_requires_yaku() -> None:
    no_yaku = hand((1, 1), (2, 1), (3, 1), (12, 1), (13, 1), (14, 1), (22, 2))
    assert not has_ron_yaku(
        no_yaku,
        chis=[6, 15],
        bakaze=27,
        jikaze=28,
        winning_tile=3,
    )
    assert has_ron_yaku(
        no_yaku,
        chis=[6],
        pons=[31],
        bakaze=27,
        jikaze=28,
        winning_tile=3,
    )


def test_closed_tsumo_is_yaku_only_for_a_complete_hand() -> None:
    complete = hand((0, 3), (1, 3), (2, 3), (3, 3), (27, 2))
    incomplete = complete.copy()
    incomplete[27] -= 1

    assert has_tsumo_yaku(complete, bakaze=27, jikaze=28, winning_tile=27)
    assert not has_tsumo_yaku(incomplete, bakaze=27, jikaze=28, winning_tile=27)
