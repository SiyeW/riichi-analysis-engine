from __future__ import annotations


def _ceil_100(value: int) -> int:
    return ((value + 99) // 100) * 100


def _settlement_values(basic_points: int) -> tuple[int, int, int, int]:
    """Return non-dealer/dealer ron and total tsumo settlement values."""
    return (
        _ceil_100(basic_points * 4),
        _ceil_100(basic_points * 6),
        _ceil_100(basic_points) * 2 + _ceil_100(basic_points * 2),
        _ceil_100(basic_points * 2) * 3,
    )


def _score_value_sets() -> tuple[tuple[int, ...], tuple[int, ...]]:
    non_dealer: set[int] = set()
    dealer: set[int] = set()

    def add(basic_points: int) -> None:
        non_dealer_ron, dealer_ron, non_dealer_tsumo, dealer_tsumo = _settlement_values(
            basic_points
        )
        non_dealer.update((non_dealer_ron, non_dealer_tsumo))
        dealer.update((dealer_ron, dealer_tsumo))

    fu_values = (20, 25, *range(30, 111, 10))
    for han in range(1, 5):
        for fu in fu_values:
            if han == 1 and fu in {20, 25}:
                continue
            basic_points = min(fu * (2 ** (han + 2)), 2_000)
            add(basic_points)

    for basic_points in (2_000, 3_000, 4_000, 6_000):
        add(basic_points)
    for multiplier in range(1, 7):
        add(8_000 * multiplier)
    return tuple(sorted(non_dealer)), tuple(sorted(dealer))


DORA_TAIL_START = 7
DORA_VALUES: tuple[int | str, ...] = (*range(DORA_TAIL_START), f"{DORA_TAIL_START}+")
NON_DEALER_SCORE_VALUES, DEALER_SCORE_VALUES = _score_value_sets()
SCORE_VALUES = tuple(sorted(set(NON_DEALER_SCORE_VALUES) | set(DEALER_SCORE_VALUES)))
SCORE_VALUE_SET = frozenset(SCORE_VALUES)
NON_DEALER_SCORE_VALUE_SET = frozenset(NON_DEALER_SCORE_VALUES)
DEALER_SCORE_VALUE_SET = frozenset(DEALER_SCORE_VALUES)


def score_class_mask(*, dealer: bool) -> tuple[bool, ...]:
    allowed = DEALER_SCORE_VALUE_SET if dealer else NON_DEALER_SCORE_VALUE_SET
    return tuple(value in allowed for value in SCORE_VALUES)
