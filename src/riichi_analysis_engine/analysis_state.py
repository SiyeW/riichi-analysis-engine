"""Public-history state used by the analysis observation.

This is deliberately independent from target generation and from libriichi's
decision state.  It retains exactly the visible timeline needed by the proven
v2.3 opponent-analysis input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PLAYERS = 4
TILE_DIM = 37
VISIBILITY_HIDDEN = "hidden"
FRAME_EVENTS = {
    "start_kyoku",
    "tsumo",
    "dahai",
    "reach",
    "reach_accepted",
    "chi",
    "pon",
    "ankan",
    "kakan",
    "daiminkan",
    "dora",
}

_TILE_NAMES = (
    "1m", "2m", "3m", "4m", "5m", "5mr", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "5pr", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "5sr", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
)
_TILE_INDEX = {name: index for index, name in enumerate(_TILE_NAMES)}
_BAKAZE_MAP = {"E": 0, "S": 1, "W": 2, "N": 3}


def tile_index(name: str) -> int:
    return _TILE_INDEX[name]


def tile37_to_34(index: int) -> int:
    if index == 5:
        return 4
    if index == 15:
        return 13
    if index == 25:
        return 22
    if index <= 4:
        return index
    if index <= 9:
        return index - 1
    if index <= 14:
        return index - 1
    if index <= 19:
        return index - 2
    if index <= 24:
        return index - 2
    return index - 3


def normalize_visibility_mode(mode: str) -> str:
    if mode != VISIBILITY_HIDDEN:
        raise ValueError("analysis observations are hidden-only")
    return mode


@dataclass(frozen=True)
class CallContext:
    kind: str
    actor: int
    target: int
    called_tile: int
    consumed: tuple[int, ...]


@dataclass(frozen=True)
class KanContext:
    kind: str
    actor: int
    target: int | None
    tiles: tuple[int, ...]


@dataclass(frozen=True)
class KawaItem:
    discard: int
    tsumogiri: bool
    is_riichi: bool
    is_dora: bool
    call: CallContext | None
    kans: tuple[KanContext, ...]


class PublicHistoryState:
    """Track one kyoku's visible history for all four rotated perspectives."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.tehais = [np.zeros(TILE_DIM, dtype=np.int32) for _ in range(PLAYERS)]
        self.rivers: list[list[tuple[int, bool]]] = [[] for _ in range(PLAYERS)]
        self.kawa_views: list[list[list[KawaItem | None]]] = [
            [[] for _ in range(PLAYERS)] for _ in range(PLAYERS)
        ]
        self.pending_call: list[CallContext | None] = [None] * PLAYERS
        self.pending_kans: list[list[KanContext]] = [[] for _ in range(PLAYERS)]
        self.reach_declared_pending = [False] * PLAYERS
        self.melds: list[list[tuple[np.ndarray, bool]]] = [
            [] for _ in range(PLAYERS)
        ]
        self.riichi = [False] * PLAYERS
        self.riichi_accepted = [False] * PLAYERS
        self.dora_indicators: list[int] = []
        self.public_seen = np.zeros(TILE_DIM, dtype=np.int32)
        self.wall_remaining = 70
        self.ankans: list[list[int]] = [[] for _ in range(PLAYERS)]
        self.oya = 0
        self.bakaze = 0
        self.kyoku = 1
        self.honba = 0
        self.kyotaku = 0
        self.scores = [25_000] * PLAYERS

    @staticmethod
    def _tile(name: str | None) -> int | None:
        if not name or name == "?":
            return None
        return tile_index(name)

    def _add_public(self, name: str | None) -> None:
        index = self._tile(name)
        if index is not None:
            self.public_seen[index] += 1

    def _remove_hand(self, player: int, names: list[str]) -> None:
        for name in names:
            index = self._tile(name)
            if index is not None:
                self.tehais[player][index] = max(
                    0, self.tehais[player][index] - 1
                )

    def _counts(self, names: list[str]) -> np.ndarray:
        counts = np.zeros(TILE_DIM, dtype=np.int32)
        for name in names:
            index = self._tile(name)
            if index is not None:
                counts[index] += 1
        return counts

    def _pad_kawa_for_call(self, actor: int, target: int) -> None:
        skipped = (target + 1) % PLAYERS
        while skipped != actor:
            for perspective in range(PLAYERS):
                relative = (skipped - perspective + PLAYERS) % PLAYERS
                self.kawa_views[perspective][relative].append(None)
            skipped = (skipped + 1) % PLAYERS

    def _is_dora(self, tile37: int) -> bool:
        tile34 = tile37_to_34(tile37)
        for marker37 in self.dora_indicators:
            marker = tile37_to_34(marker37)
            if marker < 27:
                base = marker // 9 * 9
                dora = base + (marker - base + 1) % 9
            elif marker <= 30:
                dora = 27 + (marker - 27 + 1) % 4
            else:
                dora = 31 + (marker - 31 + 1) % 3
            if tile34 == dora:
                return True
        return False

    def _start_kyoku(self, event: dict[str, Any]) -> None:
        self.reset()
        self.oya = int(event["oya"])
        self.bakaze = _BAKAZE_MAP[event["bakaze"]]
        self.kyoku = int(event["kyoku"])
        self.honba = int(event.get("honba", 0))
        self.kyotaku = int(event.get("kyotaku", 0))
        self.scores = list(event.get("scores", [25_000] * PLAYERS))
        for perspective in range(PLAYERS):
            dealer_relative = (self.oya - perspective + PLAYERS) % PLAYERS
            for relative in range(dealer_relative):
                self.kawa_views[perspective][relative].append(None)
        marker = self._tile(event.get("dora_marker"))
        if marker is not None:
            self.dora_indicators = [marker]
        self._add_public(event.get("dora_marker"))
        for player, hand in enumerate(event["tehais"]):
            for name in hand:
                index = self._tile(name)
                if index is not None:
                    self.tehais[player][index] += 1

    def _tsumo(self, event: dict[str, Any]) -> None:
        actor = int(event["actor"])
        index = self._tile(event.get("pai"))
        if index is not None:
            self.tehais[actor][index] += 1
        self.wall_remaining = max(0, self.wall_remaining - 1)

    def _dahai(self, event: dict[str, Any]) -> None:
        actor = int(event["actor"])
        index = tile_index(event["pai"])
        is_riichi = self.reach_declared_pending[actor]
        self.tehais[actor][index] = max(0, self.tehais[actor][index] - 1)
        self.rivers[actor].append((index, bool(event.get("tsumogiri", False))))
        self._add_public(event["pai"])
        item = KawaItem(
            discard=index,
            tsumogiri=bool(event.get("tsumogiri", False)),
            is_riichi=is_riichi,
            is_dora=self._is_dora(index),
            call=self.pending_call[actor],
            kans=tuple(self.pending_kans[actor]),
        )
        for perspective in range(PLAYERS):
            relative = (actor - perspective + PLAYERS) % PLAYERS
            self.kawa_views[perspective][relative].append(item)
        self.pending_call[actor] = None
        self.pending_kans[actor].clear()
        self.reach_declared_pending[actor] = False

    def _call(self, event: dict[str, Any]) -> None:
        actor = int(event["actor"])
        kind = event["type"]
        target = int(event["target"])
        if kind == "pon":
            self._pad_kawa_for_call(actor, target)
        consumed = list(event["consumed"])
        called = event["pai"]
        self._remove_hand(actor, consumed)
        for name in consumed:
            self._add_public(name)
        self.melds[actor].append((self._counts(consumed + [called]), False))
        self.pending_call[actor] = CallContext(
            kind=kind,
            actor=actor,
            target=target,
            called_tile=tile_index(called),
            consumed=tuple(tile_index(name) for name in consumed),
        )

    def _kan(self, event: dict[str, Any]) -> None:
        actor = int(event["actor"])
        kind = event["type"]
        consumed = list(event.get("consumed", []))
        called = event.get("pai")
        target = event.get("target")
        if kind == "daiminkan" and isinstance(target, int):
            self._pad_kawa_for_call(actor, target)
        names = consumed + ([called] if called else [])
        self.pending_kans[actor].append(
            KanContext(
                kind=kind,
                actor=actor,
                target=int(target) if isinstance(target, int) else None,
                tiles=tuple(
                    index
                    for index in (self._tile(name) for name in names)
                    if index is not None
                ),
            )
        )
        if kind == "ankan":
            self._remove_hand(actor, consumed)
            for name in consumed:
                self._add_public(name)
            tile34 = tile37_to_34(tile_index(consumed[0]))
            self.ankans[actor].append(tile34)
            self.melds[actor].append((self._counts(consumed), True))
            return
        if kind == "kakan":
            if called:
                self._remove_hand(actor, [called])
                self._add_public(called)
            tile34 = tile37_to_34(tile_index(called))
            for slot, (counts, closed) in enumerate(self.melds[actor]):
                family = sum(
                    int(count)
                    for index, count in enumerate(counts)
                    if tile37_to_34(index) == tile34
                )
                if not closed and family >= 3:
                    updated = counts.copy()
                    updated[tile_index(called)] += 1
                    self.melds[actor][slot] = (updated, False)
                    break
            return
        self._remove_hand(actor, consumed)
        for name in consumed:
            self._add_public(name)
        self.melds[actor].append((self._counts(names), False))

    def process(self, event: dict[str, Any]) -> None:
        kind = event["type"]
        if kind == "start_kyoku":
            self._start_kyoku(event)
        elif kind == "tsumo":
            self._tsumo(event)
        elif kind == "dahai":
            self._dahai(event)
        elif kind in {"chi", "pon"}:
            self._call(event)
        elif kind in {"ankan", "kakan", "daiminkan"}:
            self._kan(event)
        elif kind == "reach":
            actor = int(event["actor"])
            self.riichi[actor] = True
            self.reach_declared_pending[actor] = True
        elif kind == "reach_accepted":
            actor = int(event["actor"])
            self.riichi_accepted[actor] = True
            self.kyotaku += 1
            scores = event.get("scores")
            if isinstance(scores, list) and len(scores) == PLAYERS:
                self.scores = list(scores)
            else:
                self.scores[actor] -= 1_000
        elif kind == "dora":
            marker = self._tile(event.get("dora_marker"))
            if marker is not None:
                self.dora_indicators.append(marker)
            self._add_public(event.get("dora_marker"))


# Names retained for the proven encoder source, whose semantics are checked by
# element-for-element parity tests.
GameState = PublicHistoryState
_FRAME_EVENTS = FRAME_EVENTS
_t37_to_t34 = tile37_to_34
