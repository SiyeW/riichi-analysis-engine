"""Versioned semantic input shared by future CNN and Transformer models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constants import PLAYERS, TILE37_TO_ACTION

SEMANTIC_INPUT_SCHEMA_ID = "riichi-analysis-semantic-input-v1"
EVENT_MEMORY_SCHEMA_ID = "riichi-analysis-public-events-v1"

# Keep these ids stable. A new public frame event requires a new semantic input
# schema rather than silently changing the meaning of existing stored values.
PUBLIC_EVENT_TYPES = (
    "start_kyoku",
    "tsumo",
    "dahai",
    "chi",
    "pon",
    "daiminkan",
    "ankan",
    "kakan",
    "reach",
    "reach_accepted",
    "dora",
)
PUBLIC_EVENT_TYPE_TO_ID = {
    event_type: index + 1 for index, event_type in enumerate(PUBLIC_EVENT_TYPES)
}

EVENT_TYPE = 0
EVENT_ACTOR = 1
EVENT_TARGET = 2
EVENT_TILE = 3
EVENT_CONSUMED_START = 4
MAX_EVENT_CONSUMED = 4
EVENT_FLAGS = EVENT_CONSUMED_START + MAX_EVENT_CONSUMED
EVENT_FIELDS = EVENT_FLAGS + 1

EVENT_FLAG_TSUMOGIRI = 1 << 0


def _player_code(value: Any) -> int:
    return int(value) + 1 if isinstance(value, int) and 0 <= value < PLAYERS else 0


def _tile_code(value: Any) -> int:
    if not isinstance(value, str) or value == "?":
        return 0
    try:
        return TILE37_TO_ACTION[value] + 1
    except KeyError as error:
        raise ValueError(f"unknown physical tile {value!r}") from error


def encode_public_event(event: dict[str, Any]) -> np.ndarray:
    """Encode one frame event without flattening its semantic fields."""

    kind = event.get("type")
    if kind not in PUBLIC_EVENT_TYPE_TO_ID:
        raise ValueError(f"unsupported public frame event {kind!r}")
    token = np.zeros(EVENT_FIELDS, dtype=np.uint8)
    token[EVENT_TYPE] = PUBLIC_EVENT_TYPE_TO_ID[kind]
    token[EVENT_ACTOR] = _player_code(event.get("actor"))
    token[EVENT_TARGET] = _player_code(event.get("target"))

    if kind in {"start_kyoku", "dora"}:
        token[EVENT_TILE] = _tile_code(event.get("dora_marker"))
    else:
        token[EVENT_TILE] = _tile_code(event.get("pai"))

    consumed = event.get("consumed", [])
    if not isinstance(consumed, list):
        raise TypeError("event consumed tiles must be a list")
    if len(consumed) > MAX_EVENT_CONSUMED:
        raise ValueError("event contains too many consumed tiles")
    for offset, tile in enumerate(consumed):
        token[EVENT_CONSUMED_START + offset] = _tile_code(tile)
    if bool(event.get("tsumogiri")):
        token[EVENT_FLAGS] |= EVENT_FLAG_TSUMOGIRI
    return token


@dataclass
class PublicEventHistoryEncoder:
    """Accumulate one shared event catalog and report each frame's kyoku prefix."""

    _tokens: list[np.ndarray] = field(default_factory=list)
    _kyoku_start: int | None = None

    def advance(self, event: dict[str, Any]) -> tuple[int, int]:
        token = encode_public_event(event)
        if event["type"] == "start_kyoku":
            self._kyoku_start = len(self._tokens)
        if self._kyoku_start is None:
            raise ValueError("public event history starts before start_kyoku")
        self._tokens.append(token)
        return self._kyoku_start, len(self._tokens) - self._kyoku_start

    def array(self) -> np.ndarray:
        if not self._tokens:
            return np.zeros((0, EVENT_FIELDS), dtype=np.uint8)
        return np.stack(self._tokens, axis=0)


def materialize_event_memory(
    catalog: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    perspectives: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one padded, perspective-relative event batch.

    The catalog may retain a player's own draw tile once per game. During
    materialization the tile is hidden from every other perspective, matching
    the information available to the engine at inference time.
    """

    catalog = np.asarray(catalog, dtype=np.uint8)
    starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
    perspectives = np.asarray(perspectives, dtype=np.int64).reshape(-1)
    if catalog.ndim != 2 or catalog.shape[1] != EVENT_FIELDS:
        raise ValueError("event catalog has the wrong shape")
    if not (len(starts) == len(lengths) == len(perspectives)):
        raise ValueError("event references do not share one sample count")
    if (lengths <= 0).any():
        raise ValueError("event history lengths must be positive")
    if (starts < 0).any() or (starts + lengths > len(catalog)).any():
        raise ValueError("event history reference lies outside the catalog")
    if ((perspectives < 0) | (perspectives >= PLAYERS)).any():
        raise ValueError("event perspective must be in 0..3")

    width = int(lengths.max(initial=0))
    memory = np.zeros((len(starts), width, EVENT_FIELDS), dtype=np.uint8)
    mask = np.zeros((len(starts), width), dtype=bool)
    tsumo_id = PUBLIC_EVENT_TYPE_TO_ID["tsumo"]
    for row, (start, length, perspective) in enumerate(
        zip(starts, lengths, perspectives, strict=True)
    ):
        stop = int(start + length)
        view = catalog[int(start) : stop].copy()
        for field_index in (EVENT_ACTOR, EVENT_TARGET):
            absolute = view[:, field_index]
            present = absolute > 0
            view[present, field_index] = (
                (absolute[present].astype(np.int16) - 1 - perspective) % PLAYERS + 1
            ).astype(np.uint8)
        opponent_draw = (view[:, EVENT_TYPE] == tsumo_id) & (
            view[:, EVENT_ACTOR] != 1
        )
        view[opponent_draw, EVENT_TILE] = 0
        memory[row, : int(length)] = view
        mask[row, : int(length)] = True
    return memory, mask


def semantic_input_metadata() -> dict[str, object]:
    return {
        "schema": SEMANTIC_INPUT_SCHEMA_ID,
        "eventMemorySchema": EVENT_MEMORY_SCHEMA_ID,
        "eventTypes": list(PUBLIC_EVENT_TYPES),
        "eventFields": EVENT_FIELDS,
        "physicalTileTypes": len(TILE37_TO_ACTION),
        "players": PLAYERS,
    }
