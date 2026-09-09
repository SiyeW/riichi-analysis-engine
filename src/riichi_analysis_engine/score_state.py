"""Public score state needed by the rank-augmented observation.

``libriichi.PlayerState`` intentionally exposes game-state features rather
than its internal score array.  The protocol already supplies the public
score-changing events, so this small state machine keeps the model input
independent of private binding details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constants import PLAYERS


def relative_scores(scores: np.ndarray | list[int] | tuple[int, ...], perspective: int) -> np.ndarray:
    """Rotate absolute seat scores into Mortal's controlled-player order."""

    if not 0 <= perspective < PLAYERS:
        raise ValueError(f"perspective is outside 0..{PLAYERS - 1}: {perspective}")
    values = np.asarray(scores, dtype=np.int32)
    if values.shape != (PLAYERS,):
        raise ValueError(f"scores must contain exactly {PLAYERS} values: {values.shape}")
    return np.roll(values, -perspective)


def _event_scores(event: dict[str, Any]) -> np.ndarray | None:
    scores = event.get("scores")
    if not isinstance(scores, list):
        return None
    values = np.asarray(scores, dtype=np.int32)
    if values.shape != (PLAYERS,):
        raise ValueError(f"event scores must contain exactly {PLAYERS} values")
    return values


@dataclass
class PublicScoreState:
    """Reconstruct scores from the public mjai event sequence."""

    scores: np.ndarray = field(
        default_factory=lambda: np.full(PLAYERS, 25_000, dtype=np.int32)
    )
    started: bool = False

    def process(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "start_kyoku":
            scores = _event_scores(event)
            if scores is None:
                raise ValueError("start_kyoku requires four scores")
            self.scores = scores
            self.started = True
            return
        if not self.started:
            return
        if kind == "reach_accepted":
            scores = _event_scores(event)
            if scores is not None:
                self.scores = scores
                return
            actor = event.get("actor")
            if not isinstance(actor, int) or not 0 <= actor < PLAYERS:
                raise ValueError("reach_accepted requires an actor in 0..3")
            self.scores[actor] -= 1_000
            return
        if kind in {"hora", "ryukyoku"}:
            deltas = event.get("deltas")
            if isinstance(deltas, list):
                values = np.asarray(deltas, dtype=np.int32)
                if values.shape != (PLAYERS,):
                    raise ValueError(f"event deltas must contain exactly {PLAYERS} values")
                self.scores = self.scores + values

    def relative(self, perspective: int) -> np.ndarray:
        if not self.started:
            raise ValueError("cannot encode ranks before start_kyoku")
        return relative_scores(self.scores, perspective)
