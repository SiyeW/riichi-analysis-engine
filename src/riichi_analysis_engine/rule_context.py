"""Named, model-owned rule facts shared by every semantic prediction head.

The v12 model inherited slices of Mortal's observation as a policy-only
context.  That made exact facts such as self shanten, furiten and a currently
legal win invisible to the kyoku heads.  This module replaces those opaque
channel ranges with a small contract built by the engine's shared rule kernel.

Candidate identity remains a policy concern: the shared context describes the
position and which classes of action are legal, while the policy decoder still
scores the concrete candidates and applies the authoritative action mask.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .constants import ACTION_SPACE, TILE34_TO_INDEX, TILE_TYPES, deaka
from .hand_rules import analyze_hand_rules

RULE_CONTEXT_SCHEMA_ID = "riichi-analysis-rule-context-v2"

RULE_GLOBAL_FEATURE_NAMES = (
    "shanten_complete",
    *(f"shanten_{value}" for value in range(7)),
    "furiten",
    "discard_furiten",
    "temporary_furiten",
    "riichi_furiten",
    "riichi_declared",
    "riichi_accepted",
    "phase_discard",
    "phase_response",
    "phase_kan_select",
    "can_pass",
    "can_discard",
    "can_riichi",
    "can_chi_low",
    "can_chi_mid",
    "can_chi_high",
    "can_pon",
    "can_daiminkan",
    "can_ankan",
    "can_kakan",
    "can_tsumo",
    "can_ron",
    "can_ryukyoku",
)
RULE_TILE_FEATURE_NAMES = (
    "structural_wait",
    "ron_yaku",
    "tsumo_yaku",
    "effective_draw",
    "effective_remaining",
    "legal_discard",
    "discard_result_complete",
    *(f"discard_result_shanten_{value}" for value in range(7)),
    "discard_ukeire",
    "discard_creates_furiten",
    "ankan_candidate",
    "kakan_candidate",
)

RULE_GLOBAL_CHANNELS = len(RULE_GLOBAL_FEATURE_NAMES)
RULE_TILE_CHANNELS = len(RULE_TILE_FEATURE_NAMES)
RULE_CONTEXT_CHANNELS = RULE_GLOBAL_CHANNELS + RULE_TILE_CHANNELS

_GLOBAL = {name: index for index, name in enumerate(RULE_GLOBAL_FEATURE_NAMES)}
_TILE = {name: index for index, name in enumerate(RULE_TILE_FEATURE_NAMES)}


def _bool(value: Any) -> float:
    return float(bool(value))


def _tile_flags(values: Iterable[Any]) -> np.ndarray:
    result = np.zeros(TILE_TYPES, dtype=np.float32)
    for value in values:
        if isinstance(value, str):
            tile = TILE34_TO_INDEX.get(deaka(value))
            if tile is not None:
                result[tile] = 1.0
    return result


def _value(owner: Any, name: str, default: Any) -> Any:
    value = getattr(owner, name, default)
    return value() if callable(value) else value


def encode_rule_context(
    state: Any,
    candidates: Any,
    action_mask: Any,
    *,
    at_kan_select: bool = False,
    seat: int | None = None,
    rule_state: Any | None = None,
) -> np.ndarray:
    """Encode exact self-state facts without depending on Mortal channel ids."""

    global_values = np.zeros(RULE_GLOBAL_CHANNELS, dtype=np.float32)
    tile_values = np.zeros((RULE_TILE_CHANNELS, TILE_TYPES), dtype=np.float32)
    mask = np.asarray(action_mask, dtype=bool)
    if mask.shape != (ACTION_SPACE,):
        raise ValueError(f"wrong action-mask shape for rule context: {mask.shape}")

    resolved_seat = int(
        seat if seat is not None else _value(state, "player_id", 0)
    )
    facts = analyze_hand_rules(state, seat=resolved_seat, rule_state=rule_state)
    can_tsumo = bool(_value(candidates, "can_tsumo_agari", False))
    can_ron = bool(_value(candidates, "can_ron_agari", False))
    shanten_name = (
        "shanten_complete"
        if facts.complete
        else f"shanten_{facts.shanten}"
    )
    global_values[_GLOBAL[shanten_name]] = 1.0
    furiten = (
        facts.discard_furiten or facts.temporary_furiten or facts.riichi_furiten
    )
    global_values[_GLOBAL["furiten"]] = _bool(furiten)
    global_values[_GLOBAL["discard_furiten"]] = _bool(facts.discard_furiten)
    global_values[_GLOBAL["temporary_furiten"]] = _bool(facts.temporary_furiten)
    global_values[_GLOBAL["riichi_furiten"]] = _bool(facts.riichi_furiten)
    global_values[_GLOBAL["riichi_declared"]] = _bool(
        getattr(state, "self_riichi_declared", False)
    )
    global_values[_GLOBAL["riichi_accepted"]] = _bool(
        getattr(state, "self_riichi_accepted", False)
    )

    can_discard = bool(_value(candidates, "can_discard", False))
    can_pass = bool(_value(candidates, "can_pass", False))
    global_values[_GLOBAL["phase_discard"]] = _bool(can_discard)
    global_values[_GLOBAL["phase_response"]] = _bool(can_pass and not can_discard)
    global_values[_GLOBAL["phase_kan_select"]] = _bool(at_kan_select)
    candidate_features = {
        "can_pass": can_pass,
        "can_discard": can_discard,
        "can_riichi": _value(candidates, "can_riichi", False),
        "can_chi_low": _value(candidates, "can_chi_low", False),
        "can_chi_mid": _value(candidates, "can_chi_mid", False),
        "can_chi_high": _value(candidates, "can_chi_high", False),
        "can_pon": _value(candidates, "can_pon", False),
        "can_daiminkan": _value(candidates, "can_daiminkan", False),
        "can_ankan": _value(candidates, "can_ankan", False),
        "can_kakan": _value(candidates, "can_kakan", False),
        "can_tsumo": can_tsumo,
        "can_ron": can_ron,
        "can_ryukyoku": _value(candidates, "can_ryukyoku", False),
    }
    for name, value in candidate_features.items():
        global_values[_GLOBAL[name]] = _bool(value)

    for name, values in (
        ("structural_wait", facts.structural_waits),
        ("ron_yaku", facts.ron_yaku),
        ("tsumo_yaku", facts.tsumo_yaku),
        ("effective_draw", facts.effective_draws),
        ("effective_remaining", facts.effective_remaining / 4.0),
        ("discard_ukeire", facts.discard_ukeire / 136.0),
        ("discard_creates_furiten", facts.discard_creates_furiten),
    ):
        if values.shape != (TILE_TYPES,):
            raise ValueError(f"wrong {name} shape: {values.shape}")
        tile_values[_TILE[name]] = values
    for tile, result_shanten in enumerate(facts.discard_result_shanten):
        if result_shanten == -1:
            tile_values[_TILE["discard_result_complete"], tile] = 1.0
        elif 0 <= result_shanten <= 6:
            tile_values[_TILE[f"discard_result_shanten_{result_shanten}"], tile] = 1.0

    # The action space keeps red fives distinct; shared tile semantics use the
    # corresponding base tile, while candidate embeddings preserve red identity.
    legal_discard = mask[:37]
    tile_values[_TILE["legal_discard"]] = legal_discard[:TILE_TYPES]
    for red_action, base_tile in ((34, 4), (35, 13), (36, 22)):
        tile_values[_TILE["legal_discard"], base_tile] = max(
            tile_values[_TILE["legal_discard"], base_tile],
            float(legal_discard[red_action]),
        )
    tile_values[_TILE["ankan_candidate"]] = _tile_flags(
        _value(state, "ankan_candidates", ())
    )
    tile_values[_TILE["kakan_candidate"]] = _tile_flags(
        _value(state, "kakan_candidates", ())
    )

    return np.concatenate(
        (
            tile_values,
            np.broadcast_to(global_values[:, None], (RULE_GLOBAL_CHANNELS, TILE_TYPES)),
        ),
        axis=0,
    ).copy()


def rule_context_metadata() -> dict[str, object]:
    return {
        "schema": RULE_CONTEXT_SCHEMA_ID,
        "tileFeatures": list(RULE_TILE_FEATURE_NAMES),
        "globalFeatures": list(RULE_GLOBAL_FEATURE_NAMES),
        "channels": RULE_CONTEXT_CHANNELS,
        "tileTypes": TILE_TYPES,
    }
