import numpy as np
import pytest

from riichi_analysis_engine.replay import hand_score


class StubState:
    def __init__(self, honba: int = 0) -> None:
        self.honba = honba


def test_a_plain_ron_yields_its_hand_value() -> None:
    event = {"actor": 2, "target": 3, "deltas": [0, 0, 12000, -12000]}
    assert hand_score(event, StubState(honba=0), first_winner=True) == 12000


def test_honba_is_stripped_for_the_first_winner_only() -> None:
    # The discarder pays the hand plus 300 per honba; only the first winner's
    # payment carries it, so only that one is stripped back to the hand value.
    event = {"actor": 2, "target": 3, "deltas": [0, 0, 12300, -12300]}
    assert hand_score(event, StubState(honba=1), first_winner=True) == 12000

    second = {"actor": 2, "target": 3, "deltas": [0, 0, 12000, -12000]}
    assert hand_score(second, StubState(honba=1), first_winner=False) == 12000


def test_a_tsumo_yields_the_sum_of_every_payment() -> None:
    event = {"actor": 1, "target": 1, "deltas": [-2000, 8000, -2000, -4000]}
    assert hand_score(event, StubState(honba=0), first_winner=True) == 8000


def test_a_ron_with_two_losers_is_rejected_as_a_replay_problem() -> None:
    # Seen in 2025061121gm-00a9-0000-5f8f6cdd.mjson: two players lose points on a
    # single hora, so no legal hand value can be read out of it.
    event = {"actor": 2, "target": 3, "deltas": [-16300, 0, 34300, -16000]}
    with pytest.raises(ValueError, match="not a legal single-winner settlement"):
        hand_score(event, StubState(honba=1), first_winner=True)


def test_a_settlement_that_is_not_a_legal_tsumo_is_rejected() -> None:
    event = {"actor": 1, "target": 1, "deltas": [-2000, 8000, 0, -4000]}
    with pytest.raises(ValueError, match="not a legal tsumo settlement"):
        hand_score(event, StubState(honba=0), first_winner=True)


def test_a_published_table_gap_still_reports_the_hand_score() -> None:
    # One payer, so the settlement is legal, but the value is outside the table.
    event = {"actor": 2, "target": 3, "deltas": [0, 0, 15700, -15700]}
    with pytest.raises(ValueError, match="unsupported hand score: 15700"):
        hand_score(event, StubState(honba=0), first_winner=True)


def test_the_score_vocabulary_covers_the_standard_settlements() -> None:
    from riichi_analysis_engine.prediction_values import SCORE_VALUE_SET

    for value in (1000, 1300, 2000, 3900, 7700, 8000, 12000, 18000, 24000, 32000, 48000):
        assert value in SCORE_VALUE_SET
    assert not np.any(np.asarray(sorted(SCORE_VALUE_SET)) % 100)
