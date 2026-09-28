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


class _OneDiscardPlayerState(_PassivePlayerState):
    def __init__(self, player: int) -> None:
        super().__init__(player)
        self.last_kind = ""

    def update(self, event: str) -> SimpleNamespace:
        self.last_kind = json.loads(event)["type"]
        return SimpleNamespace(
            can_discard=self.player == 0 and self.last_kind == "tsumo"
        )

    def encode_obs(self, _version: int, _kan_select: bool):
        mask = np.zeros(ACTION_SPACE, dtype=bool)
        if self.player == 0 and self.last_kind == "tsumo":
            mask[4] = True  # ordinary 5m
        return np.zeros((MORTAL_OBS_CHANNELS, TILE_TYPES), dtype=np.float32), mask


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


def test_v15_conversion_carries_exact_public_facts_without_changing_targets() -> None:
    events = [
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 12,
            "kyotaku": 11,
            "oya": 0,
            "dora_marker": "9p",
            "scores": [105_000, 22_000, -5_000, 18_000],
            "tehais": [["1m"], [], [], []],
        },
        {"type": "dahai", "actor": 0, "pai": "1m"},
        {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]
    converted = convert_game(
        events,
        "v15-facts-game",
        player_state_type=_PassivePlayerState,
        model_format=15,
    )
    facts = converted.arrays["v15_facts"]
    assert facts.shape[1] == 57
    assert 105_000 in facts[:, [0, 6, 12, 18]]
    assert np.all(facts[:, 26] == 12)
    assert np.all(facts[:, 27] == 11)
    assert "hidden_baseline_anchor" in converted.arrays


def test_v16_analysis_only_rows_keep_empty_complete_candidate_sets() -> None:
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
            "tehais": [["2m"], [], [], []],
        },
        {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]
    converted = convert_game(
        events,
        "v16-analysis-row",
        player_state_type=_PassivePlayerState,
        model_format=16,
    )
    arrays = converted.arrays
    assert arrays["candidate_codes"].shape == (1, 32, 9)
    assert arrays["candidate_mask"].shape == (1, 32)
    assert arrays["candidate_label"].tolist() == [-1]
    assert not arrays["candidate_mask"].any()
    assert arrays["public_meld_count"].shape == (1, 4, 37)
    assert arrays["current_concealed_dora"].shape == (1, 3)


def test_v16_conversion_labels_the_complete_hand_discard_not_tsumogiri() -> None:
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
            "tehais": [["5m"], [], [], []],
        },
        {"type": "tsumo", "actor": 0, "pai": "5m"},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]
    arrays = convert_game(
        events,
        "v16-discard-row",
        player_state_type=_OneDiscardPlayerState,
        model_format=16,
    ).arrays
    policy_rows = np.flatnonzero(arrays["candidate_label"] >= 0)
    assert len(policy_rows) == 1
    row = policy_rows[0]
    assert arrays["candidate_mask"][row].sum() == 2
    # The freshly drawn copy is first; the observed hand discard is second.
    assert arrays["candidate_label"][row] == 1
    assert arrays["candidate_codes"][row, 0, 3] == 1
    assert arrays["candidate_codes"][row, 1, 3] == 0


def test_v16_converted_rows_have_finite_forward_and_loss() -> None:
    import torch

    from riichi_analysis_engine.losses import multitask_loss
    from riichi_analysis_engine.model import RiichiAnalysisModel
    from riichi_analysis_engine.semantic_input import materialize_event_memory
    from riichi_analysis_engine.storage import pack_shard_arrays, unpack_shard_arrays
    from riichi_analysis_engine.train import forward_batch
    from riichi_analysis_engine.training_schema import validate_v16_training_batch

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
            "tehais": [["5m"], [], [], []],
        },
        {"type": "tsumo", "actor": 0, "pai": "5m"},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]
    converted = convert_game(
        events,
        "v16-train-step",
        player_state_type=_OneDiscardPlayerState,
        model_format=16,
    )
    arrays = converted.arrays
    stored = unpack_shard_arrays(pack_shard_arrays(arrays))
    for key in (
        "candidate_codes",
        "candidate_mask",
        "candidate_label",
        "public_meld_count",
        "current_concealed_dora",
    ):
        np.testing.assert_array_equal(stored[key], arrays[key])
    tokens, mask = materialize_event_memory(
        converted.event_catalog,
        arrays["history_start"],
        arrays["history_length"],
        arrays["perspective"],
    )
    batch = {
        name: torch.from_numpy(np.asarray(value)) for name, value in arrays.items()
    }
    batch["event_tokens"] = torch.from_numpy(tokens)
    batch["event_mask"] = torch.from_numpy(mask)
    assert validate_v16_training_batch(batch)["completePolicyRows"] == 1
    model = RiichiAnalysisModel(format_version=16)
    outputs = forward_batch(model, batch)
    assert torch.isfinite(outputs["policy"][batch["candidate_mask"]]).all()
    total, losses, _active, _weights = multitask_loss(outputs, batch)
    assert torch.isfinite(total)
    assert all(torch.isfinite(value) for value in losses.values())
    total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    analysis_only = batch["candidate_label"] < 0
    assert analysis_only.any()
    passive = {name: value[analysis_only] for name, value in batch.items()}
    passive_outputs = forward_batch(model, passive)
    passive_total, passive_losses, _active, _weights = multitask_loss(
        passive_outputs, passive
    )
    assert torch.isfinite(passive_total)
    assert torch.isfinite(passive_losses["policy"])


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
