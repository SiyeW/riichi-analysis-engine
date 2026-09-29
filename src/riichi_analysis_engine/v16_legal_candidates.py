"""Rebuild complete legal actions for supervised policy rows.

Mortal's 46-way mask is the legality authority for action *families*.  The
current known hand and exact last public action expand each allowed family
into physical, executable actions; unknown or inconsistent states fail rather
than silently merging candidates.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations, product
from typing import Any

import numpy as np

from .constants import TILES_34, TILES_37, deaka
from .replay import FullState
from .v16_candidates import candidate_signature


def _flag(owner: Any, name: str) -> bool:
    value = getattr(owner, name, False)
    return bool(value() if callable(value) else value)


def _choices(owner: Any, name: str) -> list[str]:
    value = getattr(owner, name, ())
    value = value() if callable(value) else value
    return [TILES_34[item] if isinstance(item, int) else str(item) for item in value]


def _physical_choices(hand: Counter[str], family: str) -> list[str]:
    return [tile for tile in TILES_37 if hand[tile] > 0 and deaka(tile) == family]


def _consumed_combinations(
    hand: Counter[str], family: str, size: int
) -> list[list[str]]:
    physical = [
        tile for tile in TILES_37 if deaka(tile) == family for _ in range(hand[tile])
    ]
    return [list(items) for items in sorted(set(combinations(physical, size)))]


def _chi_consumed(
    hand: Counter[str], called: str, offsets: tuple[int, int]
) -> list[list[str]]:
    family = deaka(called)
    if len(family) != 2 or family[1] not in "mps":
        return []
    rank = int(family[0])
    required = tuple(rank + offset for offset in offsets)
    if not all(1 <= value <= 9 for value in required):
        return []
    options = [_physical_choices(hand, f"{number}{family[1]}") for number in required]
    if not all(options):
        return []
    return [list(items) for items in product(*options)]


def legal_candidates(
    state: FullState,
    actor: int,
    action_mask: Any,
    cans: Any,
    player_state: Any,
) -> list[dict[str, Any]]:
    """Enumerate one row without using the future action as an input."""

    mask = np.asarray(action_mask, dtype=bool)
    if mask.shape != (46,) or not 0 <= actor < 4:
        raise ValueError("action mask or actor is invalid")
    hand = state.hands[actor]
    actions: list[dict[str, Any]] = []
    if mask[:37].any():
        for index, tile in enumerate(TILES_37):
            if not mask[index]:
                continue
            if hand[tile] <= 0:
                raise ValueError(f"legal discard {tile} is absent from the hand")
            just_drawn = state.last_draw_actor == actor and state.last_draw == tile
            if just_drawn:
                actions.append(
                    {"type": "dahai", "actor": actor, "pai": tile, "tsumogiri": True}
                )
            if not just_drawn or (hand[tile] > 1 and not state.riichi_accepted[actor]):
                actions.append(
                    {"type": "dahai", "actor": actor, "pai": tile, "tsumogiri": False}
                )
    if mask[37]:
        actions.append({"type": "reach", "actor": actor})

    called = state.last_discard
    target = state.last_discard_actor
    if mask[38:42].any() or (mask[42] and _flag(cans, "can_daiminkan")):
        if called is None or target is None:
            raise ValueError("call candidate has no preceding discard")
        for action_index, offsets in ((38, (1, 2)), (39, (-1, 1)), (40, (-2, -1))):
            if mask[action_index]:
                for consumed in _chi_consumed(hand, called, offsets):
                    actions.append(
                        {
                            "type": "chi",
                            "actor": actor,
                            "target": target,
                            "pai": called,
                            "consumed": consumed,
                        }
                    )
        if mask[41]:
            for consumed in _consumed_combinations(hand, deaka(called), 2):
                actions.append(
                    {
                        "type": "pon",
                        "actor": actor,
                        "target": target,
                        "pai": called,
                        "consumed": consumed,
                    }
                )
        if mask[42] and _flag(cans, "can_daiminkan"):
            for consumed in _consumed_combinations(hand, deaka(called), 3):
                actions.append(
                    {
                        "type": "daiminkan",
                        "actor": actor,
                        "target": target,
                        "pai": called,
                        "consumed": consumed,
                    }
                )

    if mask[42] and _flag(cans, "can_ankan"):
        for family in set(map(deaka, _choices(player_state, "ankan_candidates"))):
            consumed = [
                tile
                for tile in TILES_37
                for _ in range(hand[tile])
                if deaka(tile) == family
            ]
            if len(consumed) != 4:
                raise ValueError(f"ankan candidate {family} does not own four tiles")
            actions.append({"type": "ankan", "actor": actor, "consumed": consumed})
    if mask[42] and _flag(cans, "can_kakan"):
        for family in set(map(deaka, _choices(player_state, "kakan_candidates"))):
            matches = [
                meld
                for meld in state.melds[actor]
                if len(meld) == 3 and all(deaka(item) == family for item in meld)
            ]
            if len(matches) != 1:
                raise ValueError(f"kakan candidate {family} has no unique pon")
            physical = _physical_choices(hand, family)
            if not physical:
                raise ValueError(f"kakan candidate {family} is absent from the hand")
            for tile in physical:
                actions.append(
                    {
                        "type": "kakan",
                        "actor": actor,
                        "pai": tile,
                        "consumed": sorted(matches[0]),
                    }
                )

    if mask[43]:
        if _flag(cans, "can_tsumo_agari"):
            target, tile = actor, state.last_draw
        elif _flag(cans, "can_ron_agari"):
            target, tile = (
                (state.last_kan_actor, state.last_kan_tile)
                if state.last_kan_tile is not None
                else (state.last_discard_actor, state.last_discard)
            )
        else:
            raise ValueError("hora mask has no matching legal state")
        if target is None or tile is None:
            raise ValueError("hora candidate has no winning tile or target")
        actions.append({"type": "hora", "actor": actor, "target": target, "pai": tile})
    if mask[44]:
        actions.append({"type": "ryukyoku", "actor": actor})
    if mask[45]:
        if (
            state.riichi[actor]
            and _flag(cans, "can_ankan")
            and state.last_draw is not None
        ):
            actions.append(
                {
                    "type": "none",
                    "actor": actor,
                    "variant": "skip-ankan",
                    "pai": state.last_draw,
                    "tsumogiri": True,
                }
            )
        else:
            actions.append({"type": "none", "actor": actor})
    if not actions:
        raise ValueError("legal action mask produced no complete candidates")
    signatures = [candidate_signature(action, actor) for action in actions]
    if len(signatures) != len(set(signatures)):
        raise ValueError("legal action enumeration produced duplicate candidates")
    return actions


def chosen_candidate_index(
    actions: list[dict[str, Any]],
    events: list[dict[str, Any]],
    event_index: int,
    actor: int,
    coarse_label: int,
) -> int:
    """Match the observed full action, never a coarse tile/class surrogate."""

    if coarse_label == 45:
        matches = [
            index for index, action in enumerate(actions) if action["type"] == "none"
        ]
        if len(matches) == 1:
            return matches[0]
        raise ValueError("observed pass has no unique complete candidate")
    window = events[event_index + 1 : event_index + 4]
    for observed in window:
        if coarse_label == 44 and observed.get("type") == "ryukyoku":
            matches = [
                index
                for index, action in enumerate(actions)
                if action["type"] == "ryukyoku"
            ]
            if len(matches) == 1:
                return matches[0]
        if observed.get("actor") != actor:
            continue
        kind = observed.get("type")
        if kind not in {
            "dahai",
            "reach",
            "chi",
            "pon",
            "daiminkan",
            "ankan",
            "kakan",
            "hora",
            "ryukyoku",
        }:
            continue
        if kind == "hora" and observed.get("pai") is None:
            # MJAI settlement omits the winning tile.  The preceding public
            # frame fixes it, so match the one legal win by actor and target.
            matches = [
                candidate_index
                for candidate_index, action in enumerate(actions)
                if action["type"] == "hora"
                and action.get("target") == observed.get("target")
            ]
            if len(matches) == 1:
                return matches[0]
            continue
        try:
            identity = candidate_signature(observed, actor)
        except (ValueError, TypeError):
            continue
        matches = [
            index
            for index, action in enumerate(actions)
            if candidate_signature(action, actor) == identity
        ]
        if len(matches) == 1:
            return matches[0]
    raise ValueError("observed action does not match any complete legal candidate")
