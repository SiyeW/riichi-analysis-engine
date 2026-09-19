from __future__ import annotations

import numpy as np
import pytest

from riichi_analysis_engine.constants import TILE37_TO_ACTION
from riichi_analysis_engine.semantic_input import (
    EVENT_ACTOR,
    EVENT_CONSUMED_START,
    EVENT_FIELDS,
    EVENT_TARGET,
    EVENT_TILE,
    EVENT_TYPE,
    PUBLIC_EVENT_TYPE_TO_ID,
    PublicEventHistoryEncoder,
    encode_public_event,
    materialize_event_memory,
    semantic_input_metadata,
)


def test_public_event_token_preserves_physical_tiles_and_action_structure() -> None:
    event = {
        "type": "chi",
        "actor": 1,
        "target": 0,
        "pai": "5mr",
        "consumed": ["4m", "6m"],
    }
    token = encode_public_event(event)

    assert token.shape == (EVENT_FIELDS,)
    assert token[EVENT_TYPE] == PUBLIC_EVENT_TYPE_TO_ID["chi"]
    assert token[EVENT_ACTOR] == 2
    assert token[EVENT_TARGET] == 1
    assert token[EVENT_TILE] == TILE37_TO_ACTION["5mr"] + 1
    assert token[EVENT_CONSUMED_START] == TILE37_TO_ACTION["4m"] + 1
    assert token[EVENT_CONSUMED_START + 1] == TILE37_TO_ACTION["6m"] + 1


def test_event_history_resets_prefix_at_each_kyoku() -> None:
    encoder = PublicEventHistoryEncoder()
    first = encoder.advance(
        {"type": "start_kyoku", "dora_marker": "1m"}
    )
    second = encoder.advance({"type": "tsumo", "actor": 0, "pai": "5mr"})
    third = encoder.advance({"type": "dahai", "actor": 0, "pai": "1m"})
    next_kyoku = encoder.advance(
        {"type": "start_kyoku", "dora_marker": "P"}
    )

    assert first == (0, 1)
    assert second == (0, 2)
    assert third == (0, 3)
    assert next_kyoku == (3, 1)
    assert encoder.array().shape == (4, EVENT_FIELDS)


def test_event_memory_rotates_players_and_hides_opponent_draws() -> None:
    catalog = np.stack(
        [
            encode_public_event({"type": "start_kyoku", "dora_marker": "4m"}),
            encode_public_event({"type": "tsumo", "actor": 1, "pai": "5mr"}),
            encode_public_event(
                {"type": "dahai", "actor": 1, "pai": "5mr", "tsumogiri": True}
            ),
        ]
    )
    memory, mask = materialize_event_memory(
        catalog,
        starts=np.asarray([0, 0]),
        lengths=np.asarray([3, 2]),
        perspectives=np.asarray([0, 1]),
    )

    assert memory.shape == (2, 3, EVENT_FIELDS)
    assert mask.tolist() == [[True, True, True], [True, True, False]]
    assert memory[0, 1, EVENT_ACTOR] == 2
    assert memory[0, 1, EVENT_TILE] == 0
    assert memory[1, 1, EVENT_ACTOR] == 1
    assert memory[1, 1, EVENT_TILE] == TILE37_TO_ACTION["5mr"] + 1
    assert memory[0, 2, EVENT_TILE] == TILE37_TO_ACTION["5mr"] + 1


def test_semantic_input_rejects_invalid_events_and_references() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        encode_public_event({"type": "hora"})
    with pytest.raises(ValueError, match="outside"):
        materialize_event_memory(
            np.zeros((1, EVENT_FIELDS), dtype=np.uint8),
            np.asarray([1]),
            np.asarray([1]),
            np.asarray([0]),
        )
    assert semantic_input_metadata()["physicalTileTypes"] == 37

