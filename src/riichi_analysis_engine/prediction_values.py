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


def _score_values() -> tuple[int, ...]:
    values: set[int] = set()
    fu_values = (20, 25, *range(30, 111, 10))
    for han in range(1, 5):
        for fu in fu_values:
            if han == 1 and fu in {20, 25}:
                continue
            basic_points = min(fu * (2 ** (han + 2)), 2_000)
            values.update(_settlement_values(basic_points))

    for basic_points in (2_000, 3_000, 4_000, 6_000):
        values.update(_settlement_values(basic_points))
    for multiplier in range(1, 7):
        values.update(_settlement_values(8_000 * multiplier))
    return tuple(sorted(values))


DORA_TAIL_START = 7
DORA_VALUES: tuple[int | str, ...] = (*range(DORA_TAIL_START), f"{DORA_TAIL_START}+")
SCORE_VALUES = _score_values()
SCORE_VALUE_SET = frozenset(SCORE_VALUES)
