"""Inference artifact and complete-candidate boundary for the new format."""

import sys

import numpy as np
import torch

from riichi_analysis_engine.analysis_state import PublicHistoryState
from riichi_analysis_engine.current_dora import (
    add_known_dora_to_distribution,
    known_meld_dora,
)
from riichi_analysis_engine.export_weights import main as export_weights
from riichi_analysis_engine.model import RiichiAnalysisModel
from riichi_analysis_engine.model_input import v16_model_input_metadata
from riichi_analysis_engine.prediction_values import DORA_VALUES, SCORE_VALUES
from riichi_analysis_engine.runtime import AnalysisRuntime
from riichi_analysis_engine.semantic_input import semantic_input_metadata
from riichi_analysis_engine.v16_candidates import encode_candidate_set


def test_v16_checkpoint_exports_and_runtime_loads(tmp_path, monkeypatch):
    model = RiichiAnalysisModel(format_version=16)
    checkpoint = tmp_path / "checkpoint.pt"
    weights = tmp_path / "weights.pt"
    torch.save(
        {
            "format": "riichi-analysis-model-v16",
            "model": model.state_dict(),
            "modelArchitecture": model.architecture.to_dict(),
            "modelInput": v16_model_input_metadata(),
            "semanticInput": semantic_input_metadata(),
            "predictionValues": {
                "dora": list(DORA_VALUES),
                "score": list(SCORE_VALUES),
            },
            "step": 1,
            "samplesSeen": 16,
        },
        checkpoint,
    )
    monkeypatch.setattr(sys, "argv", ["export_weights", str(checkpoint), str(weights)])
    export_weights()
    monkeypatch.setattr(
        "riichi_analysis_engine.runtime._load_player_state", lambda: object
    )
    runtime = AnalysisRuntime(weights, "cpu")
    assert runtime.format_version == 16
    assert runtime.model.architecture == model.architecture


def test_v16_candidate_scores_depend_on_complete_action_not_list_order():
    torch.manual_seed(16)
    model = RiichiAnalysisModel(format_version=16).eval()
    from riichi_analysis_engine.model_input import SHARED_MODEL_INPUT_CHANNELS
    from riichi_analysis_engine.v15_facts import V15_FACTS_WIDTH

    observation = torch.zeros(1, SHARED_MODEL_INPUT_CHANNELS, 34)
    facts = torch.zeros(1, V15_FACTS_WIDTH)
    facts[:, 24] = 1
    events = torch.zeros(1, 1, 9, dtype=torch.uint8)
    events[..., 0] = 1
    actions = [
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": True},
        {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": False},
    ]

    def scores(items):
        features = torch.from_numpy(encode_candidate_set(items, 0))[None]
        with torch.inference_mode():
            return model(
                observation,
                events,
                torch.ones(1, 1, dtype=torch.bool),
                facts,
                torch.zeros(1, 4, 37),
                features,
                torch.ones(1, len(items), dtype=torch.bool),
            )["policy"][0]

    original = scores(actions)
    reordered = scores(actions[::-1])
    torch.testing.assert_close(original, reordered.flip(0))
    assert torch.isfinite(original).all()


def test_v16_public_meld_dora_is_an_exact_distribution_floor():
    state = PublicHistoryState()
    state.process(
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000] * 4,
            "dora_marker": "4p",
            "tehais": [["?"] * 13 for _ in range(4)],
        }
    )
    state.process(
        {
            "type": "chi",
            "actor": 1,
            "target": 0,
            "pai": "5p",
            "consumed": ["4p", "6p"],
        }
    )
    counts = state.physical_meld_counts(0)
    known = known_meld_dora(counts, state.visible_dora_markers())
    assert known[1] == 1
    shifted = add_known_dora_to_distribution(
        np.array([0.7, 0.3, 0, 0, 0, 0, 0, 0], dtype=np.float32), int(known[1])
    )
    assert shifted[0] == 0
    np.testing.assert_allclose(shifted[1:3], [0.7, 0.3])
