from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

SAMPLING_SCHEMA = "riichi-analysis-frame-sampling-v2"

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
SAMPLING_STRATA = (
    "analysis_baseline_anchor",
    "analysis_state_change",
    "analysis_ordinary",
    "policy_rare_action",
    "policy_state_change",
    "policy_ordinary",
)


def _value(candidate: Any, name: str) -> bool:
    value = getattr(candidate, name, False)
    return bool(value() if callable(value) else value)


def has_rare_legal_action(candidate: Any) -> bool:
    """Return whether one controlled player's current choice is uncommon."""

    return any(_value(candidate, field) for field in RARE_ACTION_FIELDS)


@dataclass(frozen=True)
class FrameSamplingPlan:
    """Deterministically sample policy decisions and analysis states separately.

    Policy retention may use the controlled player's legal actions. Analysis
    retention only uses current public state plus the exact no-information anchor;
    neither stream is selected from a future result or another player's private
    legal-action candidates.
    """

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
            "policyRates": {
                "rare_action": self.rare_action_rate,
                "state_change": self.state_change_rate,
                "ordinary": self.ordinary_rate,
            },
            "analysisRates": {
                "baseline_anchor": 1.0,
                "state_change": self.state_change_rate,
                "ordinary": self.ordinary_rate,
            },
        }

    @staticmethod
    def analysis_stratum(event: dict[str, Any], *, baseline_anchor: bool) -> str:
        if baseline_anchor:
            return "analysis_baseline_anchor"
        if event["type"] in STATE_CHANGE_EVENTS:
            return "analysis_state_change"
        return "analysis_ordinary"

    @staticmethod
    def policy_stratum(event: dict[str, Any], candidate: Any) -> str:
        if has_rare_legal_action(candidate):
            return "policy_rare_action"
        if event["type"] in STATE_CHANGE_EVENTS:
            return "policy_state_change"
        return "policy_ordinary"

    def keep(
        self,
        source_id: str,
        event_index: int,
        perspective: int,
        stratum: str,
    ) -> bool:
        rates = {
            "analysis_baseline_anchor": 1.0,
            "analysis_state_change": self.state_change_rate,
            "analysis_ordinary": self.ordinary_rate,
            "policy_rare_action": self.rare_action_rate,
            "policy_state_change": self.state_change_rate,
            "policy_ordinary": self.ordinary_rate,
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
            (
                f"{self.seed}\0{source_id}\0{event_index}\0{perspective}\0{stratum}"
            ).encode(),
            digest_size=8,
        ).digest()
        return int.from_bytes(digest, "little") < int(float(rate) * 2**64)
