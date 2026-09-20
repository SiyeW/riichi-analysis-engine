"""Deterministic self-hand facts shared by training and runtime inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .constants import TILE_TYPES, TILES_34
from .shanten import calculate_shanten, calculate_shanten_batch
from .yaku import has_ron_yaku, has_tsumo_yaku, is_complete_hand


@dataclass(frozen=True)
class HandRuleFacts:
    complete: bool
    shanten: int
    structural_waits: np.ndarray
    ron_yaku: np.ndarray
    tsumo_yaku: np.ndarray
    effective_draws: np.ndarray
    effective_remaining: np.ndarray
    discard_result_shanten: np.ndarray
    discard_ukeire: np.ndarray
    discard_creates_furiten: np.ndarray
    discard_furiten: bool
    temporary_furiten: bool
    riichi_furiten: bool


def _array(owner: Any, name: str) -> np.ndarray:
    raw = getattr(owner, name)
    raw = raw() if callable(raw) else raw
    if isinstance(raw, bytes):
        raw = np.frombuffer(raw, dtype=np.uint8)
    result = np.asarray(raw, dtype=np.uint8).copy()
    if result.shape != (TILE_TYPES,):
        raise ValueError(f"{name} must contain 34 tile counts")
    return result


def _values(owner: Any, name: str) -> tuple[int, ...]:
    raw = getattr(owner, name, ())
    raw = raw() if callable(raw) else raw
    return tuple(int(value) for value in raw)


def _waits(hand: np.ndarray, open_melds: int) -> np.ndarray:
    result = np.zeros(TILE_TYPES, dtype=bool)
    if int(hand.sum()) % 3 != 1 or calculate_shanten(hand, open_melds) != 0:
        return result
    for tile in range(TILE_TYPES):
        if hand[tile] >= 4:
            continue
        hand[tile] += 1
        result[tile] = is_complete_hand(hand, open_melds)
        hand[tile] -= 1
    return result


def _remaining(rule_state: Any, seat: int, hand: np.ndarray) -> np.ndarray:
    if rule_state is None:
        return np.maximum(0, 4 - hand).astype(np.float32)
    return np.asarray(
        [
            max(0, 4 - int(rule_state.visible_family_count(seat, TILES_34[tile])))
            for tile in range(TILE_TYPES)
        ],
        dtype=np.float32,
    )


def analyze_hand_rules(
    state: Any,
    *,
    seat: int,
    rule_state: Any | None,
) -> HandRuleFacts:
    """Derive only exact rule facts; no strategy or value judgement is added."""

    hand = _array(state, "tehai")
    chis = _values(state, "chis")
    pons = _values(state, "pons")
    minkans = _values(state, "minkans")
    ankans = _values(state, "ankans")
    open_melds = len(chis) + len(pons) + len(minkans) + len(ankans)
    complete = int(hand.sum()) % 3 == 2 and is_complete_hand(hand, open_melds)
    shanten = calculate_shanten(hand, open_melds)

    base_hand = hand.copy()
    if int(base_hand.sum()) % 3 == 2 and not complete:
        # There is no single canonical 13-tile hand during a discard decision.
        structural_waits = np.zeros(TILE_TYPES, dtype=bool)
    else:
        structural_waits = _waits(base_hand, open_melds)

    bakaze = 27 + int(getattr(rule_state, "bakaze", 0))
    oya = int(getattr(rule_state, "oya", 0))
    jikaze = 27 + ((seat - oya) % 4)
    riichi = bool(
        getattr(state, "self_riichi_declared", False)
        or getattr(state, "self_riichi_accepted", False)
    )
    ron_yaku = np.zeros(TILE_TYPES, dtype=bool)
    tsumo_yaku = np.zeros(TILE_TYPES, dtype=bool)
    for tile in np.flatnonzero(structural_waits):
        completed = base_hand.copy()
        completed[tile] += 1
        common = {
            "chis": chis,
            "pons": pons,
            "minkans": minkans,
            "ankans": ankans,
            "bakaze": bakaze,
            "jikaze": jikaze,
            "winning_tile": int(tile),
        }
        ron_yaku[tile] = riichi or has_ron_yaku(completed, **common)
        tsumo_yaku[tile] = riichi or has_tsumo_yaku(completed, **common)

    if rule_state is None:
        furiten = (bool(getattr(state, "at_furiten", False)), False, False)
    else:
        furiten = rule_state.furiten_causes(seat, structural_waits)

    remaining = _remaining(rule_state, seat, hand)
    effective_draws = np.zeros(TILE_TYPES, dtype=bool)
    effective_remaining = np.zeros(TILE_TYPES, dtype=np.float32)
    if int(hand.sum()) % 3 == 1:
        draw_tiles = np.flatnonzero(hand < 4)
        draw_hands = np.repeat(hand[None, :], len(draw_tiles), axis=0)
        draw_hands[np.arange(len(draw_tiles)), draw_tiles] += 1
        draw_shanten = calculate_shanten_batch(draw_hands, open_melds)
        for tile, result in zip(draw_tiles, draw_shanten, strict=True):
            improved = int(result) < shanten
            if improved:
                effective_draws[tile] = True
                effective_remaining[tile] = remaining[tile]

    discard_result = np.full(TILE_TYPES, 7, dtype=np.int8)
    discard_ukeire = np.zeros(TILE_TYPES, dtype=np.float32)
    discard_creates_furiten = np.zeros(TILE_TYPES, dtype=bool)
    if int(hand.sum()) % 3 == 2:
        discard_tiles = np.flatnonzero(hand)
        discard_hands = np.repeat(hand[None, :], len(discard_tiles), axis=0)
        discard_hands[np.arange(len(discard_tiles)), discard_tiles] -= 1
        result_shantens = calculate_shanten_batch(discard_hands, open_melds)
        for discard, discarded_hand, raw_shanten in zip(
            discard_tiles, discard_hands, result_shantens, strict=True
        ):
            result_shanten = int(raw_shanten)
            discard_result[discard] = max(-1, min(6, result_shanten))
            result_waits = _waits(discarded_hand.copy(), open_melds)
            draw_tiles = np.flatnonzero(discarded_hand < 4)
            draw_hands = np.repeat(discarded_hand[None, :], len(draw_tiles), axis=0)
            draw_hands[np.arange(len(draw_tiles)), draw_tiles] += 1
            draw_results = calculate_shanten_batch(draw_hands, open_melds)
            improving = draw_tiles[draw_results < result_shanten]
            discard_ukeire[discard] = float(remaining[improving].sum())
            if rule_state is not None:
                causes = rule_state.furiten_causes(seat, result_waits)
                discard_creates_furiten[discard] = bool(
                    causes[0] or TILES_34[discard] in {
                        TILES_34[index]
                        for index in np.flatnonzero(result_waits)
                    }
                )

    return HandRuleFacts(
        complete=complete,
        shanten=max(0, min(6, shanten)),
        structural_waits=structural_waits,
        ron_yaku=ron_yaku,
        tsumo_yaku=tsumo_yaku,
        effective_draws=effective_draws,
        effective_remaining=effective_remaining,
        discard_result_shanten=discard_result,
        discard_ukeire=discard_ukeire,
        discard_creates_furiten=discard_creates_furiten,
        discard_furiten=furiten[0],
        temporary_furiten=furiten[1],
        riichi_furiten=furiten[2],
    )
