from collections import Counter
from types import SimpleNamespace

import numpy as np

from riichi_analysis_engine.constants import TILE37_TO_ACTION
from riichi_analysis_engine.replay import FullState
from riichi_analysis_engine.v16_candidates import candidate_signature
from riichi_analysis_engine.v16_legal_candidates import (
    chosen_candidate_index,
    legal_candidates,
)


def _mask(*actions: int) -> np.ndarray:
    mask = np.zeros(46, dtype=bool)
    mask[list(actions)] = True
    return mask


def test_same_tile_hand_cut_and_drawn_discard_are_separate_candidates():
    state = FullState()
    state.hands[0] = Counter({"5m": 2, "5mr": 1})
    state.last_draw_actor = 0
    state.last_draw = "5m"
    candidates = legal_candidates(
        state,
        0,
        _mask(TILE37_TO_ACTION["5m"], TILE37_TO_ACTION["5mr"]),
        SimpleNamespace(),
        SimpleNamespace(),
    )
    assert [(action["pai"], action["tsumogiri"]) for action in candidates] == [
        ("5m", True),
        ("5m", False),
        ("5mr", False),
    ]
    assert len({candidate_signature(action, 0) for action in candidates}) == 3
    events = [
        {"type": "tsumo", "actor": 0, "pai": "5m"},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
    ]
    assert chosen_candidate_index(candidates, events, 0, 0, 4) == 1


def test_chi_variants_keep_called_tile_and_red_consumed_identity():
    state = FullState()
    state.hands[1] = Counter({"4m": 1, "5m": 1, "5mr": 1})
    state.last_discard = "6m"
    state.last_discard_actor = 0
    candidates = legal_candidates(
        state,
        1,
        _mask(40, 45),
        SimpleNamespace(can_chi_high=True, can_pass=True),
        SimpleNamespace(),
    )
    assert [action.get("consumed") for action in candidates] == [
        ["4m", "5m"],
        ["4m", "5mr"],
        None,
    ]


def test_reach_declaration_discard_can_cut_a_matching_hand_tile():
    state = FullState()
    state.hands[2] = Counter({"3s": 2})
    state.last_draw_actor = 2
    state.last_draw = "3s"
    state.riichi[2] = True
    actions = legal_candidates(
        state,
        2,
        _mask(TILE37_TO_ACTION["3s"]),
        SimpleNamespace(),
        SimpleNamespace(),
    )
    assert [action["tsumogiri"] for action in actions] == [True, False]
    state.riichi_accepted[2] = True
    accepted = legal_candidates(
        state,
        2,
        _mask(TILE37_TO_ACTION["3s"]),
        SimpleNamespace(),
        SimpleNamespace(),
    )
    assert [action["tsumogiri"] for action in accepted] == [True]


def test_hora_settlement_without_pai_matches_the_unique_known_win():
    candidates = [
        {"type": "hora", "actor": 0, "target": 0, "pai": "P"},
    ]
    events = [
        {"type": "tsumo", "actor": 0, "pai": "P"},
        {"type": "hora", "actor": 0, "target": 0},
    ]
    assert chosen_candidate_index(candidates, events, 0, 0, 43) == 0


def test_kakan_family_choice_expands_to_physical_red_tile():
    state = FullState()
    state.hands[1] = Counter({"5mr": 1})
    state.melds[1] = [["5m", "5m", "5m"]]
    candidates = legal_candidates(
        state,
        1,
        _mask(42),
        SimpleNamespace(can_kakan=True),
        SimpleNamespace(kakan_candidates=[4]),
    )
    assert candidates == [
        {
            "type": "kakan",
            "actor": 1,
            "pai": "5mr",
            "consumed": ["5m", "5m", "5m"],
        }
    ]
