import torch

from riichi_analysis_engine.analysis_observation import PLANE_CHANNEL_INDEX
from riichi_analysis_engine.model_input import (
    RULE_CONTEXT_START,
    SHARED_MODEL_INPUT_CHANNELS,
)
from riichi_analysis_engine.rule_context import RULE_TILE_FEATURE_NAMES
from riichi_analysis_engine.v15_facts import V15_FACTS_WIDTH
from riichi_analysis_engine.v16_raw_facts import (
    EVENT_CONSUMED_START_RAW,
    EVENT_POSITION_SLOT,
    EVENT_TSUMOGIRI_SLOT,
    GLOBAL_FACT_START,
    PLAYER_FACT_START,
    TILE_ID_START,
    TILE_MELD_START,
    TILE_RULE_START,
    WALL_FACT_START,
    encode_v16_raw_facts,
)


def test_all_public_fact_owners_reach_the_same_reader_without_learned_projection():
    observation = torch.zeros(1, SHARED_MODEL_INPUT_CHANNELS, 34)
    rule = RULE_TILE_FEATURE_NAMES.index("structural_wait")
    observation[0, RULE_CONTEXT_START + rule, 4] = 1
    observation[0, PLANE_CHANNEL_INDEX["dora_indicator_count_1"], 4] = 1
    facts = torch.zeros(1, V15_FACTS_WIDTH)
    facts[0, 0] = 32_000
    facts[0, 24] = 2
    facts[0, -1] = 55
    events = torch.zeros(1, 2, 9, dtype=torch.uint8)
    events[0, 0] = torch.tensor([3, 2, 0, 35, 35, 5, 0, 0, 1])
    events[0, 1] = torch.tensor([3, 2, 0, 35, 5, 35, 0, 0, 1])
    mask = torch.tensor([[True, False]])
    melds = torch.zeros(1, 4, 37)
    melds[0, 2, 34] = 2

    raw = encode_v16_raw_facts(observation, facts, events, mask, melds)

    assert raw.shape == (1, 45, 256)
    assert raw[0, 4, TILE_ID_START + 4] == 1
    assert raw[0, 4, TILE_RULE_START + rule] == 1
    assert raw[0, 34, TILE_MELD_START + 2] == 0.5
    assert raw[0, 37, PLAYER_FACT_START] == 0.32
    assert raw[0, 41, GLOBAL_FACT_START] == 2
    assert raw[0, 42, WALL_FACT_START] == 55 / 70
    assert raw[0, 43, EVENT_CONSUMED_START_RAW + 34] == 1
    assert raw[0, 43, EVENT_CONSUMED_START_RAW + 4] == 1
    assert raw[0, 43, EVENT_TSUMOGIRI_SLOT] == 1
    assert raw[0, 43, EVENT_POSITION_SLOT] == 0
    assert torch.count_nonzero(raw[0, 44]) == 0
    events[0, 1] = events[0, 0]
    mask[0, 1] = True
    ordered = encode_v16_raw_facts(observation, facts, events, mask, melds)
    assert ordered[0, 44, EVENT_POSITION_SLOT] == 1 / 1024
    events[0, 1, 4:6] = torch.tensor([5, 35])
    reversed_order = encode_v16_raw_facts(observation, facts, events, mask, melds)
    torch.testing.assert_close(ordered, reversed_order)
