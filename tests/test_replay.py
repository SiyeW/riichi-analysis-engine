import gzip
import json
import zipfile
from types import SimpleNamespace

import numpy as np

from riichi_analysis_engine.prediction_values import SCORE_VALUE_SET, SCORE_VALUES
from riichi_analysis_engine.replay import (
    ExactTargetTracker,
    FullState,
    HiddenBaselineAnchorTracker,
    _ron_label_hand,
    action_label,
    annotate_game,
    hand_score,
    placement_label,
    read_events,
    terminal_match_scores,
)


def _start_kyoku(dealer: int) -> dict[str, object]:
    return {
        "type": "start_kyoku",
        "oya": dealer,
        "bakaze": "E",
        "dora_marker": "1m",
        "tehais": [[], [], [], []],
    }


def test_hidden_baseline_anchor_ends_on_an_opponent_dealers_first_choice() -> None:
    tracker = HiddenBaselineAnchorTracker()

    assert tracker.process(_start_kyoku(1)).tolist() == [True] * 4
    assert tracker.process({"type": "tsumo", "actor": 1, "pai": "?"}).tolist() == [
        True,
        True,
        True,
        True,
    ]
    choice = tracker.process({"type": "dahai", "actor": 1, "pai": "1m"})

    assert choice.tolist() == [False, True, False, False]


def test_hidden_baseline_anchor_includes_own_dealers_first_discard_only() -> None:
    tracker = HiddenBaselineAnchorTracker()

    tracker.process(_start_kyoku(0))
    assert tracker.process({"type": "tsumo", "actor": 0, "pai": "1m"})[0]
    assert tracker.process(
        {"type": "ankan", "actor": 0, "consumed": ["1m"] * 4}
    )[0]
    assert tracker.process({"type": "dora", "dora_marker": "2m"})[0]
    assert tracker.process({"type": "dahai", "actor": 0, "pai": "2m"})[0]

    # A normal next draw proves that every eligible opponent passed the first
    # discard, so the first frame after it is already evidence-bearing.
    assert not tracker.process({"type": "tsumo", "actor": 1, "pai": "?"})[0]
    assert not tracker.process({"type": "dahai", "actor": 1, "pai": "3m"})[0]


def test_hidden_baseline_anchor_ends_immediately_on_an_opponent_ankan() -> None:
    tracker = HiddenBaselineAnchorTracker()
    tracker.process(_start_kyoku(2))
    tracker.process({"type": "tsumo", "actor": 2, "pai": "?"})

    frame = tracker.process(
        {"type": "ankan", "actor": 2, "consumed": ["5p"] * 4}
    )

    assert frame.tolist() == [False, False, True, False]


def test_read_events_detects_gzip_payload_inside_misnamed_zip_member(tmp_path) -> None:
    events = [{"type": "start_game"}, {"type": "end_game"}]
    payload = "".join(json.dumps(event) + "\n" for event in events).encode()
    archive_path = tmp_path / "year.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("game.mjson", gzip.compress(payload))

    assert read_events(f"zip://{archive_path}!game.mjson") == events


def test_read_events_detects_gzip_payload_without_gz_suffix(tmp_path) -> None:
    events = [{"type": "start_game"}, {"type": "end_game"}]
    source = tmp_path / "game.mjson"
    source.write_bytes(
        gzip.compress("".join(json.dumps(event) + "\n" for event in events).encode())
    )

    assert read_events(str(source)) == events


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
    # A settlement has to balance: the winner's gain is what the payers lost
    # plus any riichi sticks on the table, which is what the hand value is read
    # from. The first winner also receives the honba, so it is stripped again.
    state = FullState(honba=1)
    first = {"actor": 2, "target": 0, "deltas": [-4200, 0, 4200, 0]}
    second = {"actor": 3, "target": 0, "deltas": [-1000, 0, 0, 1000]}
    assert hand_score(first, state, first_winner=True) == 3_900
    assert hand_score(second, state, first_winner=False) == 1_000


def test_placement_ties_use_initial_seat_order() -> None:
    scores = np.asarray([30000, 30000, 20000, 20000])
    assert placement_label(scores, 0) == 0


def test_terminal_riichi_sticks_are_awarded_to_first_place() -> None:
    scores = np.asarray([17600, 10100, 44300, 27000])

    assert terminal_match_scores(scores, 1).tolist() == [17600, 10100, 45300, 27000]
    assert scores.tolist() == [17600, 10100, 44300, 27000]


def test_terminal_riichi_stick_tie_uses_initial_seat_order() -> None:
    scores = np.asarray([30000, 30000, 20000, 19000])

    assert terminal_match_scores(scores, 1).tolist() == [31000, 30000, 20000, 19000]


def test_exact_targets_reuse_an_unchanged_player_state(monkeypatch) -> None:
    tracker = ExactTargetTracker()
    state = SimpleNamespace(
        tehai=np.zeros(34, dtype=np.uint8),
        chis=[],
        pons=[],
        minkans=[],
        ankans=[],
        shanten=3,
        has_next_shanten_discard=False,
        self_riichi_declared=False,
        self_riichi_accepted=False,
    )
    calls = 0
    original = tracker._compute_player_target

    def counted(player, player_state):
        nonlocal calls
        calls += 1
        return original(player, player_state)

    monkeypatch.setattr(tracker, "_compute_player_target", counted)

    first = tracker._player_target(0, state)
    second = tracker._player_target(0, state)

    assert second is first
    assert calls == 1


def test_annotate_game_settles_unclaimed_terminal_riichi_sticks() -> None:
    events = [
        {
            "type": "start_kyoku",
            "bakaze": "S",
            "kyoku": 4,
            "honba": 0,
            "kyotaku": 0,
            "oya": 3,
            "dora_marker": "1m",
            "scores": [18600, 11100, 45300, 25000],
            "tehais": [[], [], [], []],
        },
        {"type": "reach_accepted", "actor": 1},
        {"type": "ryukyoku", "deltas": [-1000, 3000, -1000, -1000]},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]

    annotations = annotate_game(events)

    assert annotations[0].final_kyoku_scores.tolist() == [17600, 13100, 44300, 24000]
    assert annotations[0].final_match_scores.tolist() == [17600, 13100, 45300, 24000]


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


def test_ron_label_hand_keeps_the_last_draw_across_a_call_frame() -> None:
    # A call can leave its actor with a 3n+2 concealed shape before their
    # required discard.  The established target convention removes that
    # actor's latest draw even though an external policy-state implementation
    # may have cleared its own transient draw marker during the call.
    hand = np.zeros(34, dtype=np.int16)
    hand[[19, 20, 21, 28, 30]] = [1, 1, 2, 1, 3]

    result = _ron_label_hand(
        hand,
        player=2,
        last_tsumo_actor=2,
        last_tsumo_tile=21,
    )

    assert result.tolist()[21] == 1
    assert int(result.sum()) == 7
    assert hand.tolist()[21] == 2
