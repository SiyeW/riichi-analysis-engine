import numpy as np
import pytest
import torch

from riichi_analysis_engine.architecture import V15Architecture
from riichi_analysis_engine.model import RiichiAnalysisModel
from riichi_analysis_engine.model_input import SHARED_MODEL_INPUT_CHANNELS
from riichi_analysis_engine.v15_facts import V15_FACTS_WIDTH
from riichi_analysis_engine.v16_candidates import (
    candidate_features_from_codes,
    candidate_signature,
    encode_candidate,
    encode_candidate_set,
    pack_candidate_set,
)
from riichi_analysis_engine.v16_raw_facts import MIN_RAW_FACT_WIDTH


def test_discard_identity_distinguishes_red_and_tsumogiri():
    hand = {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False}
    drawn = {**hand, "tsumogiri": True}
    red = {**hand, "pai": "5mr"}
    assert not np.array_equal(encode_candidate(hand, 0), encode_candidate(drawn, 0))
    assert not np.array_equal(encode_candidate(hand, 0), encode_candidate(red, 0))
    assert encode_candidate_set([hand, drawn, red], 0).shape[0] == 3
    codes, mask = pack_candidate_set([hand, drawn, red], 0)
    assert mask.sum() == 3
    features = candidate_features_from_codes(torch.from_numpy(codes))
    np.testing.assert_array_equal(
        features[:3].numpy(), encode_candidate_set([hand, drawn, red], 0)
    )
    assert torch.count_nonzero(features[3:]) == 0


def test_consumed_is_a_physical_tile_multiset_not_ordered_slots():
    chi = {
        "type": "chi",
        "actor": 0,
        "target": 3,
        "pai": "6m",
        "consumed": ["4m", "5mr"],
    }
    reversed_chi = {**chi, "consumed": ["5mr", "4m"]}
    ordinary_chi = {**chi, "consumed": ["4m", "5m"]}
    assert candidate_signature(chi, 0) == candidate_signature(reversed_chi, 0)
    np.testing.assert_array_equal(
        encode_candidate(chi, 0), encode_candidate(reversed_chi, 0)
    )
    assert not np.array_equal(
        encode_candidate(chi, 0), encode_candidate(ordinary_chi, 0)
    )
    with pytest.raises(ValueError, match="duplicate"):
        encode_candidate_set([chi, reversed_chi], 0)


def test_unknown_variant_is_rejected_instead_of_merged():
    with pytest.raises(ValueError, match="variant"):
        encode_candidate({"type": "none", "actor": 0, "variant": "future-option"}, 0)


def test_v16_one_shared_reader_scores_complete_candidates_in_any_order():
    torch.manual_seed(16)
    architecture = V15Architecture(
        width=MIN_RAW_FACT_WIDTH,
        blocks=1,
        attention_heads=4,
        feed_forward_width=128,
        decoder_width=128,
    )
    model = RiichiAnalysisModel(format_version=16, architecture=architecture)
    hand = {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False}
    drawn = {**hand, "tsumogiri": True}
    red = {**hand, "pai": "5mr"}
    candidates = [hand, drawn, red]
    features = torch.from_numpy(encode_candidate_set(candidates, 0))[None]
    observation = torch.zeros(1, SHARED_MODEL_INPUT_CHANNELS, 34)
    facts = torch.zeros(1, V15_FACTS_WIDTH)
    events = torch.zeros(1, 1, 9, dtype=torch.uint8)
    events[..., 0] = 1
    mask = torch.ones(1, 1, dtype=torch.bool)
    melds = torch.zeros(1, 4, 37)
    candidate_mask = torch.ones(1, 3, dtype=torch.bool)
    first = model(observation, events, mask, facts, melds, features, candidate_mask)
    reversed_values = model(
        observation,
        events,
        mask,
        facts,
        melds,
        features.flip(1),
        candidate_mask,
    )
    assert first["policy"].shape == (1, 3)
    assert torch.isfinite(first["policy"]).all()
    torch.testing.assert_close(first["policy"], reversed_values["policy"].flip(1))
    first["policy"].sum().backward()
    assert model.semantic_model.decoder.candidate_projection.weight.grad is not None
