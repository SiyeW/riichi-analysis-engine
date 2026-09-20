from types import SimpleNamespace

import numpy as np

from riichi_analysis_engine.constants import ACTION_SPACE
from riichi_analysis_engine.rule_context import (
    RULE_GLOBAL_FEATURE_NAMES,
    RULE_TILE_FEATURE_NAMES,
    encode_rule_context,
)


def test_rule_context_names_and_encodes_decisive_self_facts() -> None:
    state = SimpleNamespace(
        shanten=0,
        waits=np.eye(1, 34, 12, dtype=np.float32)[0],
        at_furiten=False,
        self_riichi_declared=True,
        self_riichi_accepted=True,
        keep_shanten_discards=np.zeros(34, dtype=np.float32),
        next_shanten_discards=np.zeros(34, dtype=np.float32),
        ankan_candidates=lambda: ["5m"],
        kakan_candidates=lambda: ["5pr"],
    )
    candidates = SimpleNamespace(
        can_pass=True,
        can_discard=False,
        can_riichi=False,
        can_chi_low=False,
        can_chi_mid=False,
        can_chi_high=False,
        can_pon=False,
        can_daiminkan=False,
        can_ankan=True,
        can_kakan=True,
        can_tsumo_agari=False,
        can_ron_agari=True,
        can_ryukyoku=False,
    )
    mask = np.zeros(ACTION_SPACE, dtype=bool)
    mask[45] = True

    encoded = encode_rule_context(state, candidates, mask)
    tile = {name: index for index, name in enumerate(RULE_TILE_FEATURE_NAMES)}
    global_start = len(RULE_TILE_FEATURE_NAMES)
    global_ = {
        name: global_start + index
        for index, name in enumerate(RULE_GLOBAL_FEATURE_NAMES)
    }

    assert encoded[tile["completion_wait"], 12] == 1
    assert encoded[tile["ankan_candidate"], 4] == 1
    assert encoded[tile["kakan_candidate"], 13] == 1
    assert np.all(encoded[global_["shanten_complete"]] == 1)
    assert np.all(encoded[global_["can_ron"]] == 1)
    assert np.all(encoded[global_["phase_response"]] == 1)
