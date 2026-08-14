import numpy as np

from riichi_analysis_engine.prepare import mortal_validation_members, stable_fraction
from riichi_analysis_engine.replay import FullState, placement_label


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
    state.process({"type": "tsumo", "actor": 0, "pai": "5m"})
    assert int(state.wall.sum()) == 82
    assert state.dora_count(0, 0, []) == 3


def test_placement_ties_use_initial_seat_order() -> None:
    scores = np.asarray([30000, 30000, 20000, 20000])
    assert placement_label(scores, 0) == 0


def test_manifest_selection_is_order_independent(tmp_path) -> None:
    paths = [tmp_path / name for name in ["c.mjson", "a.mjson", "b.mjson", "d.mjson"]]
    assert stable_fraction(paths, 1, 2) == stable_fraction(list(reversed(paths)), 1, 2)
    assert mortal_validation_members(["x.mjson", "y.mjson"]) == mortal_validation_members(
        ["y.mjson", "x.mjson"]
    )
