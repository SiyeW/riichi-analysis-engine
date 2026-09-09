import numpy as np
from types import SimpleNamespace

from riichi_analysis_engine.prediction_values import SCORE_VALUE_SET, SCORE_VALUES
from riichi_analysis_engine.prepare import mortal_validation_members, stable_fraction
from riichi_analysis_engine.replay import FullState, action_label, hand_score, placement_label


def test_wall_and_red_dora_tracking() -> None:
    event = {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "dora_marker": "4m",
        "scores": [25000] * 4,
        "tehais": [
            ["5mr"] + ["1p"] * 4 + ["2p"] * 4 + ["3p"] * 4,
            ["1m"] * 4 + ["2m"] * 4 + ["3m"] * 4 + ["4m"],
            ["1s"] * 4 + ["2s"] * 4 + ["3s"] * 4 + ["4s"],
            ["E"] * 4 + ["S"] * 4 + ["W"] * 4 + ["N"],
        ],
    }
    state = FullState()
    state.process(event)
    assert int(state.wall.sum()) == 83
    assert state.wall_red.tolist() == [0, 1, 1]
    assert state.concealed_red_counts(0).tolist() == [1, 0, 0]
    state.process({"type": "tsumo", "actor": 0, "pai": "5m"})
    assert int(state.wall.sum()) == 82
    assert state.dora_count(0, 0, []) == 3
    state.process({"type": "dora", "dora_marker": "4m"})
    assert state.dora_count(0, 0, []) == 5


def test_score_values_follow_the_non_kiriage_table() -> None:
    assert 7_700 in SCORE_VALUE_SET
    assert 7_900 in SCORE_VALUE_SET
    assert 11_600 in SCORE_VALUE_SET
    assert 11_700 in SCORE_VALUE_SET
    assert 36_000 in SCORE_VALUE_SET
    assert 64_000 in SCORE_VALUE_SET
    assert 96_000 in SCORE_VALUE_SET
    assert SCORE_VALUES[-1] == 288_000
    assert 700 not in SCORE_VALUE_SET


def test_only_first_winner_receives_honba_in_multiple_ron() -> None:
    state = FullState(honba=1)
    first = {"actor": 2, "target": 0, "deltas": [-4200, 0, 6200, 0]}
    second = {"actor": 3, "target": 0, "deltas": [-1000, 0, 0, 1000]}
    assert hand_score(first, state, first_winner=True) == 3_900
    assert hand_score(second, state, first_winner=False) == 1_000


def test_placement_ties_use_initial_seat_order() -> None:
    scores = np.asarray([30000, 30000, 20000, 20000])
    assert placement_label(scores, 0) == 0


def test_manifest_selection_is_order_independent(tmp_path) -> None:
    paths = [tmp_path / name for name in ["c.mjson", "a.mjson", "b.mjson", "d.mjson"]]
    assert stable_fraction(paths, 1, 2) == stable_fraction(list(reversed(paths)), 1, 2)
    assert mortal_validation_members(["x.mjson", "y.mjson"]) == mortal_validation_members(
        ["y.mjson", "x.mjson"]
    )


def _empty_cans() -> SimpleNamespace:
    return SimpleNamespace(
        can_ryukyoku=False,
        can_chi_low=False,
        can_chi_mid=False,
        can_chi_high=False,
        can_pon=False,
        can_daiminkan=False,
        can_ron_agari=False,
    )


def test_kan_labels_belong_only_to_the_player_who_declared_the_kan() -> None:
    state = SimpleNamespace(ankan_candidates=["1m", "2m"], kakan_candidates=[])
    events = [
        {"type": "tsumo", "actor": 1, "pai": "3m"},
        {"type": "ankan", "actor": 1, "consumed": ["2m"] * 4},
    ]

    assert action_label(0, state, _empty_cans(), events, 0) == (None, None)
    assert action_label(1, state, _empty_cans(), events, 0) == (42, 1)
