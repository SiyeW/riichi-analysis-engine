from types import SimpleNamespace

import numpy as np

from riichi_analysis_engine.constants import (
    ACTION_SPACE,
    MORTAL_OBS_CHANNELS,
    TILE_TYPES,
)
from riichi_analysis_engine.convert import convert_game
from riichi_analysis_engine.semantic_input import (
    EVENT_TILE,
    PUBLIC_EVENT_TYPE_TO_ID,
)


class _PassivePlayerState:
    def __init__(self, player: int) -> None:
        self.player = player
        self.tehai = np.zeros(TILE_TYPES, dtype=np.uint8)
        self.shanten = 1
        self.has_next_shanten_discard = False
        self.self_riichi_declared = False
        self.self_riichi_accepted = False
        self.chis: list[object] = []
        self.pons: list[object] = []
        self.minkans: list[object] = []
        self.ankans: list[object] = []
        self.ankan_candidates: list[str] = []
        self.kakan_candidates: list[str] = []

    def update(self, _event: str) -> SimpleNamespace:
        return SimpleNamespace(
            can_ryukyoku=False,
            can_chi_low=False,
            can_chi_mid=False,
            can_chi_high=False,
            can_pon=False,
            can_daiminkan=False,
            can_ron_agari=False,
        )

    def encode_obs(self, _version: int, _kan_select: bool) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros((MORTAL_OBS_CHANNELS, TILE_TYPES), dtype=np.float32),
            np.zeros(ACTION_SPACE, dtype=bool),
        )


def test_conversion_references_one_shared_full_event_catalog() -> None:
    events = [
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "dora_marker": "1m",
            "scores": [25_000] * 4,
            "tehais": [[], [], [], []],
        },
        {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]

    converted = convert_game(
        events,
        "synthetic-game",
        player_state_type=_PassivePlayerState,
    )

    assert converted.event_catalog.shape[0] == 1
    assert converted.event_catalog[0, 0] == PUBLIC_EVENT_TYPE_TO_ID["start_kyoku"]
    assert converted.event_catalog[0, EVENT_TILE] > 0
    np.testing.assert_array_equal(converted.arrays["history_start"], [0])
    np.testing.assert_array_equal(converted.arrays["history_length"], [1])
    np.testing.assert_array_equal(converted.arrays["analysis_active"], [True])
