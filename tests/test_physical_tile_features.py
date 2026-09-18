from __future__ import annotations

import pytest
import torch

from riichi_analysis_engine.analysis_observation import (
    PLANE_CHANNEL_INDEX,
    PLANE_CHANNELS,
)
from riichi_analysis_engine.constants import TILE_TYPES
from riichi_analysis_engine.physical_tile_features import (
    PHYSICAL_TILE_TYPES,
    dora_tile_index,
    physical_dora_features,
    physical_red_features,
)


def _set_indicator(
    observation: torch.Tensor,
    batch: int,
    tile: int,
    count: int,
    *,
    red_suit: str | None = None,
) -> None:
    for copy in range(1, count + 1):
        observation[
            batch, PLANE_CHANNEL_INDEX[f"dora_indicator_count_{copy}"], tile
        ] = 1
    if red_suit is not None:
        observation[
            batch, PLANE_CHANNEL_INDEX[f"dora_indicator_red_{red_suit}"], :
        ] = 1


def test_dora_indicator_cycles_match_riichi_rules() -> None:
    assert dora_tile_index(8) == 0
    assert dora_tile_index(17) == 9
    assert dora_tile_index(26) == 18
    assert dora_tile_index(30) == 27
    assert dora_tile_index(33) == 31
    with pytest.raises(ValueError, match="34-tile"):
        dora_tile_index(34)


def test_physical_dora_features_split_red_identity_and_stack_dora_value() -> None:
    observation = torch.zeros(2, PLANE_CHANNELS, TILE_TYPES)

    # Two 4m indicators make both physical 5m identities worth two visible
    # dora; the red entity receives its intrinsic aka bonus as a third dora.
    _set_indicator(observation, 0, 3, 2)
    # A red 5p indicator is physically separate but indicates 6p exactly like
    # a normal 5p marker.
    _set_indicator(observation, 0, 13, 1, red_suit="p")
    # Honor cycles remain rule-derived rather than learned.
    _set_indicator(observation, 1, 30, 1)
    _set_indicator(observation, 1, 33, 1)

    features = physical_dora_features(observation)

    assert features.indicator_count.shape == (2, PHYSICAL_TILE_TYPES)
    assert features.dora_multiplier.shape == (2, PHYSICAL_TILE_TYPES)
    assert features.stacked().shape == (2, PHYSICAL_TILE_TYPES, 4)

    assert features.indicator_count[0, 3].item() == 2
    assert features.indicator_count[0, 13].item() == 0
    assert features.indicator_count[0, 35].item() == 1
    assert features.dora_multiplier[0, 4].item() == 2
    assert features.dora_multiplier[0, 34].item() == 2
    assert features.visible_dora_weight[0, 4].item() == 2
    assert features.visible_dora_weight[0, 34].item() == 3
    assert features.dora_multiplier[0, 14].item() == 1

    assert features.dora_multiplier[1, 27].item() == 1
    assert features.dora_multiplier[1, 31].item() == 1
    assert torch.equal(features.aka_bonus[0, :TILE_TYPES], torch.zeros(TILE_TYPES))
    assert torch.equal(features.aka_bonus[0, TILE_TYPES:], torch.ones(3))


def test_physical_dora_features_reject_the_combined_model_input() -> None:
    with pytest.raises(ValueError, match="public-history"):
        physical_dora_features(torch.zeros(1, PLANE_CHANNELS + 1, TILE_TYPES))


def test_red_five_public_facts_are_localized_to_red_entities() -> None:
    observation = torch.zeros(1, PLANE_CHANNELS, TILE_TYPES)
    observation[:, PLANE_CHANNEL_INDEX["hand_red_m"], :] = 1
    observation[:, PLANE_CHANNEL_INDEX["river_p2_red_p"], :] = 1
    observation[:, PLANE_CHANNEL_INDEX["current_tile_red_s"], :] = 1

    features = physical_red_features(observation)

    assert features.shape == (1, PHYSICAL_TILE_TYPES, 7)
    assert torch.count_nonzero(features[:, :TILE_TYPES]) == 0
    assert features[0, 34, 0] == 1
    assert features[0, 35, 4] == 1
    assert features[0, 36, 6] == 1
    assert torch.count_nonzero(features) == 3
