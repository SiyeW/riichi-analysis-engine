import json
from types import SimpleNamespace

import numpy as np
import pytest

from riichi_analysis_engine.constants import (
    ACTION_SPACE,
    MORTAL_OBS_CHANNELS,
    TILE_TYPES,
)
from riichi_analysis_engine.convert import (
    _analysis_perspective,
    convert_game,
    preflight_conversion,
)
from riichi_analysis_engine.frame_sampling import FrameSamplingPlan
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

    def encode_obs(
        self, _version: int, _kan_select: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros((MORTAL_OBS_CHANNELS, TILE_TYPES), dtype=np.float32),
            np.zeros(ACTION_SPACE, dtype=bool),
        )


class _CountingPlayerState(_PassivePlayerState):
    encode_calls = 0

    def encode_obs(
        self, version: int, kan_select: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        type(self).encode_calls += 1
        return super().encode_obs(version, kan_select)


def test_analysis_perspective_is_identity_deterministic() -> None:
    first = [_analysis_perspective("game", index) for index in range(16)]

    assert first == [_analysis_perspective("game", index) for index in range(16)]
    assert all(0 <= perspective < 4 for perspective in first)


def test_analysis_perspective_prefers_an_exact_baseline_anchor() -> None:
    anchors = np.asarray([False, False, True, False])

    assert _analysis_perspective("game", 4, anchors) == 2


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


def test_conversion_encodes_only_the_retained_analysis_perspective() -> None:
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
    _CountingPlayerState.encode_calls = 0

    convert_game(events, "count-observations", player_state_type=_CountingPlayerState)

    assert _CountingPlayerState.encode_calls == 1


def test_conversion_records_perspective_relative_hidden_baseline_anchors() -> None:
    events = [
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "dora_marker": "9p",
            "scores": [25_000] * 4,
            "tehais": [["1m"], [], [], []],
        },
        {"type": "dahai", "actor": 0, "pai": "1m"},
        {"type": "tsumo", "actor": 1, "pai": "2m"},
        {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]

    converted = convert_game(
        events,
        "baseline-anchor-game",
        player_state_type=_PassivePlayerState,
        model_format=13,
    )

    anchors = converted.arrays["hidden_baseline_anchor"]
    perspectives = converted.arrays["perspective"]
    np.testing.assert_array_equal(converted.arrays["event_index"], [0, 1, 2])
    assert bool(anchors[0])
    assert bool(anchors[1]) == bool(perspectives[1] == 0)
    assert not bool(anchors[2])


def test_conversion_keeps_only_exact_analysis_anchors_at_zero_rates() -> None:
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
            "tehais": [["1m"], [], [], []],
        },
        {"type": "dahai", "actor": 0, "pai": "1m"},
        {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]
    converted = convert_game(
        events,
        "sampled-game",
        player_state_type=_PassivePlayerState,
        sampling_plan=FrameSamplingPlan(
            seed=9,
            rare_action_rate=0.0,
            state_change_rate=0.0,
            ordinary_rate=0.0,
        ),
    )

    np.testing.assert_array_equal(converted.arrays["event_index"], [0, 1])
    assert converted.frame_counts["analysis_baseline_anchor"] == {
        "seen": 2,
        "kept": 2,
    }
    assert converted.frame_counts["analysis_ordinary"] == {"seen": 0, "kept": 0}


def test_conversion_preflight_performs_no_writes(tmp_path) -> None:
    sources = []
    records = []
    for index in range(2):
        source = tmp_path / f"game-{index}.mjson"
        source.write_text("", encoding="utf-8")
        sources.append(source)
        records.append({"sourceId": f"game-{index}", "path": str(source)})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [json.dumps({"manifest": {"split": "synthetic"}})]
            + [json.dumps(record) for record in records]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "not-created" / "stage"

    metadata, selected, report = preflight_conversion(
        manifest,
        output,
        start_game=0,
        end_game=0,
        max_games=1,
    )

    assert metadata["split"] == "synthetic"
    assert selected == records
    assert report["selectedGames"] == 1
    assert report["writesPerformed"] is False
    assert not output.exists()


def test_conversion_preflight_rejects_duplicate_source_identity(tmp_path) -> None:
    source = tmp_path / "game.mjson"
    source.write_text("", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    duplicate = {"sourceId": "same", "path": str(source)}
    manifest.write_text(
        "\n".join(
            [json.dumps({"manifest": {}}), json.dumps(duplicate), json.dumps(duplicate)]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="repeats sourceId"):
        preflight_conversion(
            manifest,
            tmp_path / "stage",
            start_game=0,
            end_game=0,
            max_games=0,
        )
