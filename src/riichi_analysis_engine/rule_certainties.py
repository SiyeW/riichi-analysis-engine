from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constants import RED_TILES, TILES_34, deaka


def _seat(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if 0 <= parsed < 4 else None


def _tile(value: Any) -> str | None:
    parsed = str(value or "")
    return parsed if parsed in TILES_34 or parsed in RED_TILES else None


def constrain_distribution(
    probabilities: np.ndarray,
    minimum: int,
    maximum: int,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64).copy()
    if values.ndim != 1:
        raise ValueError("probability distribution must be one-dimensional")
    minimum = max(0, int(minimum))
    maximum = min(len(values) - 1, int(maximum))
    if minimum > maximum:
        return values
    if minimum == 0 and maximum == len(values) - 1:
        return values
    if minimum == maximum:
        values.fill(0.0)
        values[minimum] = 1.0
        return values
    values[:minimum] = 0.0
    values[maximum + 1 :] = 0.0
    total = float(values.sum())
    if total > 0.0:
        values /= total
    else:
        values[minimum : maximum + 1] = 1.0 / (maximum - minimum + 1)
    return values


@dataclass
class PublicRuleState:
    concealed_sizes: list[int] = field(default_factory=lambda: [0] * 4)
    known_concealed: list[Counter[str]] = field(
        default_factory=lambda: [Counter() for _ in range(4)]
    )
    exposed: Counter[str] = field(default_factory=Counter)
    wall_size: int = 0
    forbidden_tiles: list[set[str]] = field(
        default_factory=lambda: [set() for _ in range(4)]
    )
    riichi: list[bool] = field(default_factory=lambda: [False] * 4)
    _own_discards: list[set[str]] = field(
        default_factory=lambda: [set() for _ in range(4)]
    )
    _temporary_passes: list[set[str]] = field(
        default_factory=lambda: [set() for _ in range(4)]
    )
    _riichi_passes: list[set[str]] = field(
        default_factory=lambda: [set() for _ in range(4)]
    )
    _pending_discard: dict[str, Any] | None = None

    @classmethod
    def from_events(cls, events: list[dict[str, Any]]) -> PublicRuleState:
        state = cls()
        for event in events:
            state.process(event)
        state._refresh_forbidden_tiles()
        return state

    def _reset(self, event: dict[str, Any]) -> None:
        raw_hands = event.get("tehais")
        hands = raw_hands if isinstance(raw_hands, list) else []
        self.concealed_sizes = [
            len(hands[seat])
            if seat < len(hands) and isinstance(hands[seat], list)
            else 0
            for seat in range(4)
        ]
        self.known_concealed = [Counter() for _ in range(4)]
        for seat, hand in enumerate(hands[:4]):
            if not isinstance(hand, list):
                continue
            for raw_tile in hand:
                tile = _tile(raw_tile)
                if tile is not None:
                    self.known_concealed[seat][tile] += 1
        self.exposed = Counter()
        marker = _tile(event.get("dora_marker"))
        if marker is not None:
            self.exposed[marker] += 1
        self.wall_size = max(
            0,
            136 - sum(self.concealed_sizes) - sum(self.exposed.values()),
        )
        self.forbidden_tiles = [set() for _ in range(4)]
        self.riichi = [False] * 4
        self._own_discards = [set() for _ in range(4)]
        self._temporary_passes = [set() for _ in range(4)]
        self._riichi_passes = [set() for _ in range(4)]
        self._pending_discard = None

    def _resolve_pending_discard(self, event: dict[str, Any]) -> None:
        if self._pending_discard is None:
            return
        kind = str(event.get("type") or "")
        pending_actor = _seat(self._pending_discard.get("actor"))
        tile = _tile(self._pending_discard.get("pai"))
        is_ron = kind == "hora" and _seat(event.get("target")) == pending_actor
        if not is_ron and pending_actor is not None and tile is not None:
            changed_actor = (
                _seat(event.get("actor"))
                if kind in {"tsumo", "chi", "pon", "ankan", "kakan", "daiminkan"}
                else None
            )
            family = deaka(tile)
            for seat in range(4):
                if seat == pending_actor:
                    continue
                if self.riichi[seat]:
                    self._riichi_passes[seat].add(family)
                elif seat != changed_actor:
                    self._temporary_passes[seat].add(family)
        self._pending_discard = None

    def _expose_from_hand(self, seat: int, tile: str) -> None:
        hand = self.known_concealed[seat]
        if hand[tile] > 0:
            hand[tile] -= 1
            if hand[tile] == 0:
                del hand[tile]
        self.exposed[tile] += 1

    def process(self, event: dict[str, Any]) -> None:
        kind = str(event.get("type") or "")
        if kind == "start_kyoku":
            self._reset(event)
            return

        self._resolve_pending_discard(event)
        actor = _seat(event.get("actor"))
        if (
            kind in {"tsumo", "chi", "pon", "ankan", "kakan", "daiminkan"}
            and actor is not None
        ):
            self._temporary_passes[actor].clear()

        if kind == "tsumo" and actor is not None:
            self.wall_size = max(0, self.wall_size - 1)
            self.concealed_sizes[actor] += 1
            tile = _tile(event.get("pai"))
            if tile is not None:
                self.known_concealed[actor][tile] += 1
        elif kind == "dahai" and actor is not None:
            tile = _tile(event.get("pai"))
            self.concealed_sizes[actor] = max(0, self.concealed_sizes[actor] - 1)
            if tile is not None:
                self._expose_from_hand(actor, tile)
                self._own_discards[actor].add(deaka(tile))
                self._pending_discard = event
        elif kind in {"chi", "pon", "daiminkan", "ankan"} and actor is not None:
            consumed = event.get("consumed")
            tiles = consumed if isinstance(consumed, list) else []
            self.concealed_sizes[actor] = max(0, self.concealed_sizes[actor] - len(tiles))
            for raw_tile in tiles:
                tile = _tile(raw_tile)
                if tile is not None:
                    self._expose_from_hand(actor, tile)
        elif kind == "kakan" and actor is not None:
            tile = _tile(event.get("pai"))
            self.concealed_sizes[actor] = max(0, self.concealed_sizes[actor] - 1)
            if tile is not None:
                self._expose_from_hand(actor, tile)
        elif kind == "dora":
            marker = _tile(event.get("dora_marker"))
            self.wall_size = max(0, self.wall_size - 1)
            if marker is not None:
                self.exposed[marker] += 1
        elif kind in {"reach", "reach_accepted"} and actor is not None:
            self.riichi[actor] = True

    def _refresh_forbidden_tiles(self) -> None:
        self.forbidden_tiles = [
            self._own_discards[seat]
            | self._temporary_passes[seat]
            | self._riichi_passes[seat]
            for seat in range(4)
        ]

    def _known_family_in_hand(self, seat: int, family: str) -> int:
        return sum(
            count
            for tile, count in self.known_concealed[seat].items()
            if deaka(tile) == family
        )

    def _known_family_total(self, family: str) -> int:
        concealed = sum(self._known_family_in_hand(seat, family) for seat in range(4))
        exposed = sum(
            count for tile, count in self.exposed.items() if deaka(tile) == family
        )
        return concealed + exposed

    def _known_red_in_hand(self, seat: int, red_tile: str) -> int:
        return self.known_concealed[seat][red_tile]

    def _known_red_total(self, red_tile: str) -> int:
        return (
            sum(self._known_red_in_hand(seat, red_tile) for seat in range(4))
            + self.exposed[red_tile]
        )

    def _unknown_hand_capacity(self, seat: int) -> int:
        return max(
            0,
            self.concealed_sizes[seat] - sum(self.known_concealed[seat].values()),
        )

    @staticmethod
    def _range(
        total_copies: int,
        known_total: int,
        known_target: int,
        target_capacity: int,
        other_capacity: int,
    ) -> tuple[int, int]:
        remaining = max(0, total_copies - known_total)
        minimum = known_target + max(0, remaining - max(0, other_capacity))
        maximum = known_target + min(remaining, max(0, target_capacity))
        return min(total_copies, minimum), min(total_copies, maximum)

    def concealed_range(self, seat: int, family: str) -> tuple[int, int]:
        known_total = self._known_family_total(family)
        if known_total > 4:
            return 0, 4
        known_target = self._known_family_in_hand(seat, family)
        target_capacity = self._unknown_hand_capacity(seat)
        other_capacity = self.wall_size + sum(
            self._unknown_hand_capacity(other) for other in range(4) if other != seat
        )
        return self._range(
            4,
            known_total,
            known_target,
            target_capacity,
            other_capacity,
        )

    def wall_range(self, family: str) -> tuple[int, int]:
        known_total = self._known_family_total(family)
        if known_total > 4:
            return 0, 4
        other_capacity = sum(self._unknown_hand_capacity(seat) for seat in range(4))
        return self._range(4, known_total, 0, self.wall_size, other_capacity)

    def concealed_red_range(self, seat: int, red_tile: str) -> tuple[int, int]:
        known_total = self._known_red_total(red_tile)
        if known_total > 1:
            return 0, 1
        known_target = self._known_red_in_hand(seat, red_tile)
        target_capacity = self._unknown_hand_capacity(seat)
        other_capacity = self.wall_size + sum(
            self._unknown_hand_capacity(other) for other in range(4) if other != seat
        )
        return self._range(
            1,
            known_total,
            known_target,
            target_capacity,
            other_capacity,
        )

    def wall_red_range(self, red_tile: str) -> tuple[int, int]:
        known_total = self._known_red_total(red_tile)
        if known_total > 1:
            return 0, 1
        other_capacity = sum(self._unknown_hand_capacity(seat) for seat in range(4))
        return self._range(1, known_total, 0, self.wall_size, other_capacity)


def apply_opponent_rule_certainties(
    shanten_probabilities: np.ndarray,
    wait_probabilities: np.ndarray,
    *,
    is_riichi: bool,
    forbidden_tiles: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    shanten = np.asarray(shanten_probabilities, dtype=np.float64).copy()
    waits = np.asarray(wait_probabilities, dtype=np.float64).copy()
    if is_riichi:
        shanten.fill(0.0)
        shanten[0] = 1.0
    for index, tile in enumerate(TILES_34):
        if tile in forbidden_tiles:
            waits[index] = 0.0
    return shanten, waits
