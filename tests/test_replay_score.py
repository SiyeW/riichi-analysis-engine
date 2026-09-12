import pytest

from riichi_analysis_engine.prediction_values import SCORE_VALUE_SET
from riichi_analysis_engine.replay import hand_score


class StubState:
    """The two fields a settlement is read against."""

    def __init__(self, honba: int = 0, kyotaku: int = 0) -> None:
        self.honba = honba
        self.kyotaku = kyotaku


def test_a_plain_ron_yields_its_hand_value() -> None:
    event = {"actor": 2, "target": 3, "deltas": [0, 0, 12000, -12000]}
    assert hand_score(event, StubState(), first_winner=True) == 12000


def test_honba_is_stripped_for_the_first_winner_only() -> None:
    first = {"actor": 2, "target": 3, "deltas": [0, 0, 12300, -12300]}
    assert hand_score(first, StubState(honba=1), first_winner=True) == 12000

    # The second winner of a double ron is paid without the honba.
    second = {"actor": 2, "target": 3, "deltas": [0, 0, 12000, -12000]}
    assert hand_score(second, StubState(honba=1), first_winner=False) == 12000


def test_riichi_sticks_are_stripped_from_the_first_winner() -> None:
    # The winner also collects the sticks already lying on the table.
    event = {"actor": 2, "target": 3, "deltas": [0, 0, 13300, -12300]}
    assert hand_score(event, StubState(honba=1, kyotaku=1), first_winner=True) == 12000


def test_a_tsumo_yields_the_sum_of_every_payment() -> None:
    event = {"actor": 1, "target": 1, "deltas": [-2000, 8000, -2000, -4000]}
    assert hand_score(event, StubState(), first_winner=True) == 8000


def test_a_ron_paid_by_two_players_still_yields_its_hand_value() -> None:
    # 2025061121gm-00a9-0000-5f8f6cdd: player 2 completes daisangen by ponning the
    # third dragon from player 0, then wins on player 3's discard. Liability
    # splits the payment between them, and the hand is a non-dealer yakuman.
    event = {"actor": 2, "target": 3, "deltas": [-16300, 0, 34300, -16000]}
    assert hand_score(event, StubState(honba=1, kyotaku=2), first_winner=True) == 32000


def test_a_value_outside_the_published_table_is_still_rejected() -> None:
    event = {"actor": 2, "target": 3, "deltas": [0, 0, 15700, -15700]}
    with pytest.raises(ValueError, match="unsupported hand score: 15700"):
        hand_score(event, StubState(), first_winner=True)


def test_the_score_vocabulary_covers_the_standard_settlements() -> None:
    for value in (1000, 1300, 2000, 3900, 7700, 8000, 12000, 18000, 24000, 32000, 48000):
        assert value in SCORE_VALUE_SET
