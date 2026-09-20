from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

SAMPLING_SCHEMA = "riichi-analysis-frame-sampling-v1"

STATE_CHANGE_EVENTS = frozenset(
    {
        "start_kyoku",
        "chi",
        "pon",
        "daiminkan",
        "ankan",
        "kakan",
        "reach",
        "reach_accepted",
        "dora",
    }
)
RARE_ACTION_FIELDS = (
    "can_riichi",
    "can_chi_low",
    "can_chi_mid",
    "can_chi_high",
    "can_pon",
    "can_daiminkan",
    "can_ankan",
    "can_kakan",
    "can_tsumo_agari",
    "can_ron_agari",
    "can_ryukyoku",
)


def _value(candidate: Any, name: str) -> bool:
    value = getattr(candidate, name, False)
    return bool(value() if callable(value) else value)


def is_terminal_preceding(events: list[dict[str, Any]], event_index: int) -> bool:
    """Return whether this frame is immediately followed by a terminal result."""

    for event in events[event_index + 1 :]:
        kind = event.get("type")
        if kind in {"hora", "ryukyoku", "end_kyoku"}:
            return True
        if kind in STATE_CHANGE_EVENTS or kind in {"tsumo", "dahai"}:
            return False
        if kind == "end_game":
            return False
    return False


def has_rare_legal_action(candidates: list[Any]) -> bool:
    return any(
        _value(candidate, field)
        for candidate in candidates
        for field in RARE_ACTION_FIELDS
    )


@dataclass(frozen=True)
class FrameSamplingPlan:
    seed: int
    rare_action_rate: float = 1.0
    state_change_rate: float = 0.5
    ordinary_rate: float = 0.05

    def __post_init__(self) -> None:
        for name, rate in (
            ("rare action", self.rare_action_rate),
            ("state change", self.state_change_rate),
            ("ordinary", self.ordinary_rate),
        ):
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"{name} sampling rate must be in [0, 1]")

    def metadata(self) -> dict[str, object]:
        return {
            "schema": SAMPLING_SCHEMA,
            "seed": self.seed,
            "rates": {
                "decisive": 1.0,
                "rare_action": self.rare_action_rate,
                "state_change": self.state_change_rate,
                "ordinary": self.ordinary_rate,
            },
        }

    def classify(
        self,
        events: list[dict[str, Any]],
        event_index: int,
        candidates: list[Any],
    ) -> str:
        if is_terminal_preceding(events, event_index):
            return "decisive"
        if has_rare_legal_action(candidates):
            return "rare_action"
        if events[event_index]["type"] in STATE_CHANGE_EVENTS:
            return "state_change"
        return "ordinary"

    def keep(self, source_id: str, event_index: int, stratum: str) -> bool:
        rates = {
            "decisive": 1.0,
            "rare_action": self.rare_action_rate,
            "state_change": self.state_change_rate,
            "ordinary": self.ordinary_rate,
        }
        try:
            rate = rates[stratum]
        except KeyError as error:
            raise ValueError(f"unknown sampling stratum: {stratum}") from error
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False
        digest = hashlib.blake2b(
            f"{self.seed}\0{source_id}\0{event_index}\0{stratum}".encode(),
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, "little") < int(float(rate) * 2**64)
