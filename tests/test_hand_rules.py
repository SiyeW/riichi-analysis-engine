from types import SimpleNamespace

import numpy as np

from riichi_analysis_engine.hand_rules import analyze_hand_rules
from riichi_analysis_engine.shanten import calculate_shanten


class _RuleState:
    bakaze = 0
    oya = 0

    def __init__(self, visible: dict[str, int] | None = None) -> None:
        self.visible = visible or {}

    def visible_family_count(self, _seat: int, family: str) -> int:
        return self.visible.get(family, 0)

    def furiten_causes(
        self, _seat: int, waits: np.ndarray
    ) -> tuple[bool, bool, bool]:
        return bool(waits[27]), False, False


def _state(hand: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        player_id=0,
        tehai=hand,
        chis=[],
        pons=[],
        minkans=[],
        ankans=[],
        self_riichi_declared=False,
        self_riichi_accepted=False,
    )


def _tanki_hand() -> np.ndarray:
    hand = np.zeros(34, dtype=np.uint8)
    hand[[0, 1, 2, 3]] = 3
    hand[27] = 1
    return hand


def test_shanten_tables_cover_complete_tenpai_and_iishanten() -> None:
    tenpai = _tanki_hand()
    complete = tenpai.copy()
    complete[27] += 1
    iishanten = tenpai.copy()
    iishanten[3] -= 1

    assert calculate_shanten(complete) == -1
    assert calculate_shanten(tenpai) == 0
    assert calculate_shanten(iishanten) == 1


def test_vectorized_shanten_matches_scalar_queries() -> None:
    from riichi_analysis_engine.shanten import calculate_shanten_batch

    random = np.random.default_rng(20260920)
    hands = []
    while len(hands) < 64:
        hand = np.zeros(34, dtype=np.uint8)
        for tile in random.choice(34, size=13, replace=True):
            if hand[tile] < 4:
                hand[tile] += 1
        if hand.sum() == 13:
            hands.append(hand)
    batch = np.asarray(hands)

    expected = np.asarray([calculate_shanten(hand) for hand in batch])
    np.testing.assert_array_equal(calculate_shanten_batch(batch), expected)


def test_rule_facts_separate_shape_yaku_copies_and_furiten() -> None:
    facts = analyze_hand_rules(
        _state(_tanki_hand()),
        seat=0,
        rule_state=_RuleState({"E": 2}),
    )

    assert facts.shanten == 0
    assert facts.structural_waits[27]
    assert facts.ron_yaku[27]
    assert facts.tsumo_yaku[27]
    assert facts.effective_draws[27]
    assert facts.effective_remaining[27] == 2
    assert facts.discard_furiten


def test_discard_summary_reports_result_shanten_and_ukeire() -> None:
    hand = _tanki_hand()
    hand[9] += 1
    facts = analyze_hand_rules(
        _state(hand),
        seat=0,
        rule_state=_RuleState({"E": 1, "1p": 1}),
    )

    assert facts.discard_result_shanten[9] == 0
    assert facts.discard_ukeire[9] == 3
