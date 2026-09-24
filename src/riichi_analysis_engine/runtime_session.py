from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .analysis_observation import IncrementalTilePlaneEncoder
from .analysis_state import FRAME_EVENTS, PublicHistoryState
from .rule_certainties import PublicRuleState
from .score_state import PublicScoreState
from .semantic_input import PublicEventHistoryEncoder, materialize_event_memory


def canonical_event(event: dict[str, Any]) -> str:
    return json.dumps(
        event,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass
class RuntimeSession:
    """Incrementally maintain the public state machines for one host session."""

    controlled_seat: int
    player_state: Any
    event_keys: list[str] = field(default_factory=list)
    score_state: PublicScoreState = field(default_factory=PublicScoreState)
    public_state: PublicHistoryState = field(default_factory=PublicHistoryState)
    tile_encoder: IncrementalTilePlaneEncoder = field(
        default_factory=IncrementalTilePlaneEncoder
    )
    event_encoder: PublicEventHistoryEncoder = field(
        default_factory=PublicEventHistoryEncoder
    )
    frame: dict[str, Any] | None = None
    event_reference: tuple[int, int] | None = None
    start_kyoku: dict[str, Any] | None = None
    terminal: bool = False
    rule_state: PublicRuleState = field(default_factory=PublicRuleState)
    result_cache: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    @classmethod
    def create(cls, player_state_type: Any, controlled_seat: int) -> RuntimeSession:
        return cls(
            controlled_seat=controlled_seat,
            player_state=player_state_type(controlled_seat),
        )

    def append(self, event: dict[str, Any], event_key: str) -> None:
        event_type = event.get("type")
        if event_type == "start_kyoku":
            self.terminal = False
        elif event_type in {"hora", "ryukyoku", "end_kyoku"}:
            self.terminal = True
        self.player_state.update(event_key)
        self.score_state.process(event)
        self.public_state.process(event)
        self.rule_state.process(event)
        if event_type in FRAME_EVENTS:
            self.tile_encoder.advance(event, self.public_state)
            self.event_reference = self.event_encoder.advance(event)
            self.frame = event
        if event_type == "start_kyoku":
            self.start_kyoku = event
        self.event_keys.append(event_key)

    def analysis_observation(self) -> np.ndarray:
        if self.frame is None:
            raise ValueError("history has no analysis frame event")
        return self.tile_encoder.encode(self.frame, self.controlled_seat)

    def semantic_event_memory(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.event_reference is None:
            raise ValueError("history has no semantic frame event")
        start, length = self.event_reference
        memory, mask = materialize_event_memory(
            self.event_encoder.array(),
            np.asarray([start]),
            np.asarray([length]),
            np.asarray([self.controlled_seat]),
        )
        return torch.from_numpy(memory), torch.from_numpy(mask)
