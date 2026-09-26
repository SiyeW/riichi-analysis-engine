import sys

import numpy as np
import torch

from riichi_analysis_engine.analysis_state import PublicHistoryState
from riichi_analysis_engine.architecture import V15Architecture
from riichi_analysis_engine.export_weights import main as export_weights
from riichi_analysis_engine.model import RiichiAnalysisModel, count_parameters
from riichi_analysis_engine.model_input import (
    SHARED_MODEL_INPUT_CHANNELS,
    V15_MODEL_INPUT_SCHEMA_ID,
    v15_model_input_metadata,
)
from riichi_analysis_engine.prediction_values import DORA_VALUES, SCORE_VALUES
from riichi_analysis_engine.rule_context import RULE_CONTEXT_CHANNELS
from riichi_analysis_engine.runtime import AnalysisRuntime
from riichi_analysis_engine.semantic_input import semantic_input_metadata
from riichi_analysis_engine.storage import pack_shard_arrays, unpack_shard_arrays
from riichi_analysis_engine.v15_facts import V15_FACTS_WIDTH, encode_v15_facts


def _start():
    return {
        "type": "start_kyoku",
        "bakaze": "E",
        "kyoku": 1,
        "honba": 12,
        "kyotaku": 11,
        "oya": 0,
        "scores": [105_000, 22_000, -5_000, 18_000],
        "dora_marker": "1m",
        "tehais": [["1m"] * 13, ["?"] * 13, ["?"] * 13, ["?"] * 13],
    }


def test_public_facts_keep_unclipped_scores_and_unknown_draw_size():
    state = PublicHistoryState()
    state.process(_start())
    state.process({"type": "tsumo", "actor": 1, "pai": "?"})
    rules = np.zeros((RULE_CONTEXT_CHANNELS, 34), dtype=np.float32)
    facts = encode_v15_facts(state, 0, rules)
    assert facts.shape == (V15_FACTS_WIDTH,)
    assert facts[0] == 105_000
    assert facts[6 + 5] == 14
    assert facts[24:28].tolist() == [1, 0, 12, 11]
    assert facts[-1] == 69


def test_v15_pack_schema_and_round_trip():
    arrays = {
        "obs": np.zeros((2, SHARED_MODEL_INPUT_CHANNELS, 34), dtype=np.float16),
        "action_mask": np.zeros((2, 46), dtype=bool),
        "v15_facts": np.ones((2, V15_FACTS_WIDTH), dtype=np.float32),
    }
    packed = pack_shard_arrays(arrays)
    assert packed["model_input_schema"].item() == V15_MODEL_INPUT_SCHEMA_ID
    restored = unpack_shard_arrays(packed)
    np.testing.assert_array_equal(restored["v15_facts"], arrays["v15_facts"])


def test_v15_shared_path_has_finite_forward_backward_and_ignores_padding():
    torch.manual_seed(15)
    model = RiichiAnalysisModel(format_version=15)
    observation = torch.zeros(2, SHARED_MODEL_INPUT_CHANNELS, 34)
    facts = torch.zeros(2, V15_FACTS_WIDTH)
    facts[:, 24] = 1
    events = torch.zeros(2, 3, 9, dtype=torch.uint8)
    events[:, :2, 0] = 1
    mask = torch.tensor([[True, True, False], [True, True, False]])
    first = model(observation, events, mask, facts)
    changed = events.clone()
    changed[:, 2] = torch.tensor([11, 4, 3, 37, 36, 35, 34, 33, 1], dtype=torch.uint8)
    second = model(observation, changed, mask, facts)
    for name in first:
        assert torch.isfinite(first[name]).all()
        torch.testing.assert_close(first[name], second[name])
    first["shanten"].sum().backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert count_parameters(model)["total"] == 3_449_498


def test_v15_checkpoint_exports_and_runtime_loads(tmp_path, monkeypatch):
    architecture = V15Architecture(
        width=16, blocks=1, attention_heads=4, feed_forward_width=32, decoder_width=32
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    weights_path = tmp_path / "weights.pt"
    torch.save(
        {
            "format": "riichi-analysis-model-v15",
            "model": RiichiAnalysisModel(
                format_version=15, architecture=architecture
            ).state_dict(),
            "modelArchitecture": architecture.to_dict(),
            "modelInput": v15_model_input_metadata(),
            "semanticInput": semantic_input_metadata(),
            "predictionValues": {
                "dora": list(DORA_VALUES),
                "score": list(SCORE_VALUES),
            },
            "step": 1,
            "samplesSeen": 16,
        },
        checkpoint_path,
    )
    monkeypatch.setattr(
        sys, "argv", ["export_weights", str(checkpoint_path), str(weights_path)]
    )
    export_weights()
    monkeypatch.setattr(
        "riichi_analysis_engine.runtime._load_player_state", lambda: object
    )
    runtime = AnalysisRuntime(weights_path, "cpu")
    assert runtime.format_version == 15
    assert runtime.model.architecture == architecture
