import numpy as np

from riichi_analysis_engine.rule_certainties import (
    PublicRuleState,
    apply_opponent_rule_certainties,
    constrain_distribution,
)


def start_kyoku(hands: list[list[str]], marker: str = "9s") -> dict:
    return {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "dora_marker": marker,
        "scores": [25_000] * 4,
        "tehais": hands,
    }


def test_distribution_mask_uses_exact_endpoints() -> None:
    raw = np.asarray([0.10, 0.20, 0.30, 0.35, 0.05])
    exact = constrain_distribution(raw, 2, 2)
    assert exact.tolist() == [0, 0, 1, 0, 0]

    bounded = constrain_distribution(raw, 1, 3)
    assert bounded[0] == 0
    assert bounded[4] == 0
    assert np.isclose(bounded.sum(), 1)


def test_visible_four_copies_make_other_locations_impossible() -> None:
    hands = [
        ["1m"] * 4 + ["?"] * 9,
        ["?"] * 13,
        ["?"] * 13,
        ["?"] * 13,
    ]
    state = PublicRuleState.from_events([start_kyoku(hands)])

    assert state.concealed_range(1, "1m") == (0, 0)
    assert state.wall_range("1m") == (0, 0)


def test_known_red_five_makes_every_other_red_location_impossible() -> None:
    hands = [
        ["5mr"] + ["?"] * 12,
        ["?"] * 13,
        ["?"] * 13,
        ["?"] * 13,
    ]
    state = PublicRuleState.from_events([start_kyoku(hands)])

    assert state.concealed_red_range(1, "5mr") == (0, 0)
    assert state.wall_red_range("5mr") == (0, 0)


def test_public_discards_reduce_count_ranges() -> None:
    hands = [["?"] * 13 for _ in range(4)]
    events = [
        start_kyoku(hands, marker="1p"),
        {"type": "tsumo", "actor": 0, "pai": "?"},
        {"type": "dahai", "actor": 0, "pai": "1m"},
        {"type": "tsumo", "actor": 1, "pai": "?"},
        {"type": "dahai", "actor": 1, "pai": "1m"},
        {"type": "tsumo", "actor": 2, "pai": "?"},
        {"type": "dahai", "actor": 2, "pai": "1m"},
        {"type": "tsumo", "actor": 3, "pai": "?"},
        {"type": "dahai", "actor": 3, "pai": "1m"},
    ]
    state = PublicRuleState.from_events(events)

    assert state.wall_range("1m") == (0, 0)
    assert all(state.concealed_range(seat, "1m") == (0, 0) for seat in range(4))


def test_riichi_and_furiten_rules_use_exact_probabilities() -> None:
    events = [
        start_kyoku([["?"] * 13 for _ in range(4)]),
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "3m"},
    ]
    state = PublicRuleState.from_events(events)
    shanten, waits = apply_opponent_rule_certainties(
        np.full(7, 1 / 7),
        np.full(34, 0.25),
        is_riichi=state.riichi[1],
        forbidden_tiles=state.forbidden_tiles[1],
    )

    assert shanten.tolist() == [1, 0, 0, 0, 0, 0, 0]
    assert waits[2] == 0
    assert waits[3] == 0.25
