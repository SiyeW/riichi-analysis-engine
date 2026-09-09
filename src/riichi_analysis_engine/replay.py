from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .kyoku_outcome import outcome_class_index

from .constants import (
    PLAYERS,
    RED_TILES,
    RED_TILE_TO_INDEX,
    TILE37_TO_ACTION,
    deaka,
    relative_players,
    tile34_index,
)
from .prediction_values import SCORE_VALUE_SET
from .yaku import has_ron_yaku, is_complete_hand

FRAME_EVENTS = {
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
}


def read_events(source: str) -> list[dict[str, Any]]:
    if source.startswith("zip://"):
        archive_path, member = source[6:].split("!", 1)
        with zipfile.ZipFile(archive_path) as archive:
            text = archive.read(member).decode("utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    path = Path(source)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _remove_one(counter: Counter[str], tile: str) -> None:
    if counter[tile] <= 0:
        raise ValueError(f"tile {tile!r} is not in the concealed hand")
    counter[tile] -= 1
    if counter[tile] == 0:
        del counter[tile]


def _dora_from_marker(marker: str) -> str:
    marker = deaka(marker)
    if marker[-1:] in {"m", "p", "s"}:
        number = int(marker[0])
        return f"{number % 9 + 1}{marker[-1]}"
    cycles = ("E", "S", "W", "N"), ("P", "F", "C")
    for cycle in cycles:
        if marker in cycle:
            return cycle[(cycle.index(marker) + 1) % len(cycle)]
    raise ValueError(f"unknown dora marker {marker!r}")


@dataclass
class FullState:
    hands: list[Counter[str]] = field(default_factory=lambda: [Counter() for _ in range(4)])
    melds: list[list[list[str]]] = field(default_factory=lambda: [[] for _ in range(4)])
    wall: np.ndarray = field(default_factory=lambda: np.zeros(34, dtype=np.uint8))
    wall_red: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.uint8))
    scores: np.ndarray = field(default_factory=lambda: np.full(4, 25_000, dtype=np.int32))
    dora_markers: list[str] = field(default_factory=list)
    riichi: list[bool] = field(default_factory=lambda: [False] * 4)
    honba: int = 0
    kyotaku: int = 0
    oya: int = 0
    bakaze: str = "E"
    last_discard: str | None = None
    last_kan_tile: str | None = None

    def process(self, event: dict[str, Any]) -> None:
        kind = event["type"]
        if kind == "start_kyoku":
            self.hands = [Counter(hand) for hand in event["tehais"]]
            self.melds = [[] for _ in range(4)]
            self.scores = np.asarray(event.get("scores", [25_000] * 4), dtype=np.int32)
            self.dora_markers = [event["dora_marker"]]
            self.riichi = [False] * 4
            self.honba = int(event.get("honba", 0))
            self.kyotaku = int(event.get("kyotaku", 0))
            self.oya = int(event["oya"])
            self.bakaze = event["bakaze"]
            self.last_discard = None
            self.last_kan_tile = None
            self.wall = np.full(34, 4, dtype=np.int16)
            self.wall_red = np.ones(3, dtype=np.int8)
            for hand in event["tehais"]:
                for tile in hand:
                    self.wall[tile34_index(tile)] -= 1
                    if tile in RED_TILE_TO_INDEX:
                        self.wall_red[RED_TILE_TO_INDEX[tile]] -= 1
            self.wall[tile34_index(event["dora_marker"])] -= 1
            if event["dora_marker"] in RED_TILE_TO_INDEX:
                self.wall_red[RED_TILE_TO_INDEX[event["dora_marker"]]] -= 1
            if (self.wall < 0).any() or (self.wall_red < 0).any():
                raise ValueError("negative wall count at start_kyoku")
            self.wall = self.wall.astype(np.uint8)
            self.wall_red = self.wall_red.astype(np.uint8)
        elif kind == "tsumo":
            actor, tile = int(event["actor"]), event["pai"]
            self.hands[actor][tile] += 1
            index = tile34_index(tile)
            if self.wall[index] == 0:
                raise ValueError(f"negative wall count after drawing {tile}")
            self.wall[index] -= 1
            if tile in RED_TILE_TO_INDEX:
                red_index = RED_TILE_TO_INDEX[tile]
                if self.wall_red[red_index] == 0:
                    raise ValueError(f"negative red wall count after drawing {tile}")
                self.wall_red[red_index] -= 1
            self.last_kan_tile = None
        elif kind == "dahai":
            actor, tile = int(event["actor"]), event["pai"]
            _remove_one(self.hands[actor], tile)
            self.last_discard = tile
            self.last_kan_tile = None
        elif kind in {"chi", "pon", "daiminkan"}:
            actor = int(event["actor"])
            consumed = list(event["consumed"])
            for tile in consumed:
                _remove_one(self.hands[actor], tile)
            called = event["pai"]
            self.melds[actor].append(consumed + [called])
            self.last_kan_tile = called if kind == "daiminkan" else None
        elif kind == "ankan":
            actor = int(event["actor"])
            consumed = list(event["consumed"])
            for tile in consumed:
                _remove_one(self.hands[actor], tile)
            self.melds[actor].append(consumed)
            self.last_kan_tile = consumed[0]
        elif kind == "kakan":
            actor, tile = int(event["actor"]), event["pai"]
            _remove_one(self.hands[actor], tile)
            family = deaka(tile)
            for meld in self.melds[actor]:
                if len(meld) == 3 and all(deaka(value) == family for value in meld):
                    meld.append(tile)
                    break
            else:
                raise ValueError(f"kakan without a matching pon: {event}")
            self.last_kan_tile = tile
        elif kind == "reach":
            self.riichi[int(event["actor"])] = True
        elif kind == "reach_accepted":
            actor = int(event["actor"])
            self.kyotaku += 1
            if isinstance(event.get("scores"), list):
                self.scores = np.asarray(event["scores"], dtype=np.int32)
            else:
                self.scores[actor] -= 1_000
        elif kind == "dora":
            marker = event["dora_marker"]
            self.dora_markers.append(marker)
            index = tile34_index(marker)
            if self.wall[index] == 0:
                raise ValueError(f"negative wall count after dora marker {marker}")
            self.wall[index] -= 1
            if marker in RED_TILE_TO_INDEX:
                red_index = RED_TILE_TO_INDEX[marker]
                if self.wall_red[red_index] == 0:
                    raise ValueError(f"negative red wall count after dora marker {marker}")
                self.wall_red[red_index] -= 1
        elif kind in {"hora", "ryukyoku"}:
            deltas = event.get("deltas")
            if isinstance(deltas, list):
                self.scores = self.scores + np.asarray(deltas, dtype=np.int32)

    def concealed_counts(self, player: int) -> np.ndarray:
        counts = np.zeros(34, dtype=np.uint8)
        for tile, count in self.hands[player].items():
            counts[tile34_index(tile)] += count
        return counts

    def concealed_red_counts(self, player: int) -> np.ndarray:
        return np.asarray(
            [self.hands[player].get(tile, 0) for tile in RED_TILES],
            dtype=np.uint8,
        )

    def winning_tiles(self, actor: int, target: int) -> list[str]:
        tiles = list(self.hands[actor].elements())
        for meld in self.melds[actor]:
            tiles.extend(meld)
        if actor != target:
            winning_tile = self.last_kan_tile or self.last_discard
            if winning_tile is None:
                raise ValueError("ron without a preceding discard or kakan")
            tiles.append(winning_tile)
        return tiles

    def dora_count(self, actor: int, target: int, ura_markers: list[str]) -> int:
        tiles = self.winning_tiles(actor, target)
        dora_tiles = [_dora_from_marker(marker) for marker in self.dora_markers]
        dora_tiles.extend(_dora_from_marker(marker) for marker in ura_markers)
        normal = sum(dora_tiles.count(deaka(tile)) for tile in tiles)
        reds = sum(tile.endswith("r") for tile in tiles)
        return normal + reds


def hand_score(event: dict[str, Any], state: FullState, *, first_winner: bool) -> int:
    actor, target = int(event["actor"]), int(event["target"])
    deltas = np.asarray(event["deltas"], dtype=np.int32)
    if actor == target:
        value = -int(deltas[np.arange(PLAYERS) != actor].sum())
    else:
        value = -int(deltas[target])
    if first_winner:
        value -= state.honba * 300
    if value not in SCORE_VALUE_SET:
        raise ValueError(f"hora produced an unsupported hand score: {value}")
    return value


@dataclass(frozen=True)
class FutureAnnotation:
    final_kyoku_scores: np.ndarray
    final_match_scores: np.ndarray
    draw: int
    win: np.ndarray
    deal_in: np.ndarray
    targets: np.ndarray
    dora: np.ndarray
    score: np.ndarray


@dataclass
class _KyokuBuilder:
    frame_indices: list[int] = field(default_factory=list)
    draw: int = 0
    win: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.uint8))
    deal_in: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.uint8))
    targets: np.ndarray = field(default_factory=lambda: np.full(4, -1, dtype=np.int8))
    dora: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.uint8))
    score: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.uint32))
    first_winner_seen: bool = False


def annotate_game(events: list[dict[str, Any]]) -> dict[int, FutureAnnotation]:
    state = FullState()
    builders: list[tuple[_KyokuBuilder, np.ndarray]] = []
    active: _KyokuBuilder | None = None

    for index, event in enumerate(events):
        kind = event["type"]
        if kind == "start_kyoku":
            active = _KyokuBuilder()
        if active is not None and kind in FRAME_EVENTS:
            active.frame_indices.append(index)

        if kind == "hora" and active is not None:
            actor, target = int(event["actor"]), int(event["target"])
            active.win[actor] = 1
            active.targets[actor] = target
            if actor != target:
                active.deal_in[target] = 1
            active.dora[actor] = state.dora_count(actor, target, list(event.get("ura_markers", [])))
            active.score[actor] = hand_score(
                event, state, first_winner=not active.first_winner_seen
            )
            active.first_winner_seen = True
        elif kind == "ryukyoku" and active is not None:
            active.draw = 1

        state.process(event)
        if kind == "end_kyoku" and active is not None:
            builders.append((active, state.scores.copy()))
            active = None

    final_match_scores = state.scores.copy()
    annotations: dict[int, FutureAnnotation] = {}
    for builder, final_kyoku_scores in builders:
        annotation = FutureAnnotation(
            final_kyoku_scores=final_kyoku_scores,
            final_match_scores=final_match_scores,
            draw=builder.draw,
            win=builder.win,
            deal_in=builder.deal_in,
            targets=builder.targets,
            dora=builder.dora,
            score=builder.score,
        )
        annotations.update((index, annotation) for index in builder.frame_indices)
    return annotations


class ExactTargetTracker:
    """Build exact shanten and legal-ron labels from four libriichi states."""

    def __init__(self) -> None:
        self.bakaze = 27
        self.oya = 0
        self.wall_remaining = 70
        self.last_tsumo_actor: int | None = None
        self.chankan_tile: int | None = None
        self.discarded = np.zeros((4, 34), dtype=bool)
        self.temporary_furiten = np.zeros(4, dtype=bool)
        self.riichi_pass_furiten = np.zeros(4, dtype=bool)
        self.pending_pass: tuple[dict[str, Any], list[int]] | None = None

    def _player_target(
        self,
        player: int,
        state: Any,
    ) -> tuple[int, int, np.ndarray]:
        hand_raw = state.tehai
        hand = (
            np.frombuffer(hand_raw, dtype=np.uint8)
            if isinstance(hand_raw, bytes)
            else np.asarray(hand_raw, dtype=np.uint8)
        ).astype(np.int16)
        drawn = state.last_self_tsumo()
        can_improve_after_discard = int(hand.sum()) % 3 == 2 and bool(
            state.has_next_shanten_discard
        )
        shanten = int(state.shanten)
        if can_improve_after_discard:
            shanten -= 1
        shanten = max(0, min(6, shanten))
        ron_waits = np.zeros(34, dtype=np.uint8)
        if shanten != 0:
            return shanten, 0, ron_waits

        if drawn is not None and int(hand.sum()) % 3 == 2:
            hand[tile34_index(drawn)] -= 1
        if int(hand.sum()) % 3 != 1:
            return shanten, 0, ron_waits

        open_melds = len(state.chis) + len(state.pons) + len(state.minkans) + len(state.ankans)
        waits = np.zeros(34, dtype=bool)
        for tile in range(34):
            if hand[tile] >= 4:
                continue
            hand[tile] += 1
            waits[tile] = is_complete_hand(hand, open_melds)
            hand[tile] -= 1
        if not waits.any():
            return shanten, 0, ron_waits
        if (
            bool((waits & self.discarded[player]).any())
            or bool(self.temporary_furiten[player])
            or bool(self.riichi_pass_furiten[player])
        ):
            return shanten, 1, ron_waits
        if bool(state.self_riichi_declared or state.self_riichi_accepted):
            return shanten, 0, waits.astype(np.uint8)
        if self.wall_remaining == 0 and player != self.last_tsumo_actor:
            return shanten, 0, waits.astype(np.uint8)

        any_yaku = False
        for tile in np.flatnonzero(waits):
            tile = int(tile)
            if tile == self.chankan_tile:
                ron_waits[tile] = 1
                any_yaku = True
                continue
            complete = hand.copy()
            complete[tile] += 1
            if has_ron_yaku(
                complete,
                chis=state.chis,
                pons=state.pons,
                minkans=state.minkans,
                ankans=state.ankans,
                bakaze=self.bakaze,
                jikaze=27 + ((player - self.oya) % 4),
                winning_tile=tile,
            ):
                ron_waits[tile] = 1
                any_yaku = True
        return shanten, int(not any_yaku), ron_waits

    def process(self, event: dict[str, Any], states: list[Any]) -> np.ndarray | None:
        kind = event["type"]
        if self.pending_pass is not None:
            discard, players = self.pending_pass
            if not (kind == "hora" and event.get("target") == discard.get("actor")):
                for player in players:
                    if bool(
                        states[player].self_riichi_declared
                        or states[player].self_riichi_accepted
                    ):
                        self.riichi_pass_furiten[player] = True
                    else:
                        self.temporary_furiten[player] = True
            self.pending_pass = None
        self.chankan_tile = None
        if kind == "start_kyoku":
            self.bakaze = 27 + "ESWN".index(event["bakaze"])
            self.oya = int(event["oya"])
            self.wall_remaining = 70
            self.last_tsumo_actor = None
            self.discarded.fill(False)
            self.temporary_furiten.fill(False)
            self.riichi_pass_furiten.fill(False)
        elif kind == "tsumo":
            self.wall_remaining = max(0, self.wall_remaining - 1)
            self.last_tsumo_actor = int(event["actor"])
            self.temporary_furiten[int(event["actor"])] = False
        elif kind == "dahai":
            self.discarded[int(event["actor"]), tile34_index(event["pai"])] = True
        elif kind == "kakan":
            self.chankan_tile = tile34_index(event["pai"])
        if kind not in FRAME_EVENTS:
            return None

        absolute = [self._player_target(player, states[player]) for player in range(4)]
        rotated = np.zeros((4, 126), dtype=np.float32)
        for perspective in range(4):
            for position, player in enumerate(relative_players(perspective)):
                shanten, stuck, waits = absolute[player]
                base = position * 8
                rotated[perspective, base + shanten] = 1
                rotated[perspective, base + 7] = stuck
                rotated[perspective, 24 + position * 34 : 24 + (position + 1) * 34] = waits
        if kind == "dahai":
            discarder = int(event["actor"])
            tile = tile34_index(event["pai"])
            players = [
                player
                for player in range(4)
                if player != discarder and absolute[player][2][tile]
            ]
            self.pending_pass = (event, players)
        return rotated


def action_label(
    player: int,
    player_state: Any,
    cans: Any,
    events: list[dict[str, Any]],
    index: int,
) -> tuple[int | None, int | None]:
    window = events[index + 1 : index + 4]
    while len(window) < 3:
        window.append({"type": "end_game"})
    immediate = window[0]
    next_event = window[1] if immediate["type"] in {"reach_accepted", "dora"} else immediate
    kind = next_event["type"]
    kan_select: int | None = None

    if kind == "dahai" and int(next_event["actor"]) == player:
        return TILE37_TO_ACTION[next_event["pai"]], None
    if kind == "reach" and int(next_event["actor"]) == player:
        return 37, None
    if kind == "chi" and int(next_event["actor"]) == player:
        called = tile34_index(next_event["pai"])
        consumed = sorted(tile34_index(tile) for tile in next_event["consumed"])
        label = 38 if called < consumed[0] else 39 if called < consumed[1] else 40
        return label, None
    if kind == "pon" and int(next_event["actor"]) == player:
        return 41, None
    if kind == "daiminkan" and int(next_event["actor"]) == player:
        return 42, None
    if kind == "kakan" and int(next_event["actor"]) == player:
        candidates = player_state.kakan_candidates
        candidates = candidates() if callable(candidates) else candidates
        if len(candidates) > 1:
            kan_select = tile34_index(next_event["pai"])
        return 42, kan_select
    if kind == "ankan" and int(next_event["actor"]) == player:
        candidates = player_state.ankan_candidates
        candidates = candidates() if callable(candidates) else candidates
        if len(candidates) > 1:
            kan_select = tile34_index(next_event["consumed"][0])
        return 42, kan_select
    if kind == "ryukyoku" and bool(cans.can_ryukyoku):
        return 44, None

    has_any_ron = immediate["type"] == "hora"
    if has_any_ron:
        for event in window:
            if event["type"] == "end_kyoku":
                break
            if event["type"] == "hora" and int(event["actor"]) == player:
                return 43, None
    if (
        bool(cans.can_chi_low or cans.can_chi_mid or cans.can_chi_high)
        and kind == "tsumo"
    ) or (
        bool(cans.can_pon or cans.can_daiminkan or cans.can_ron_agari)
        and not has_any_ron
    ):
        return 45, None
    return None, None


def passive_perspective(source_id: str, event_index: int) -> int:
    digest = hashlib.blake2b(
        f"{source_id}:{event_index}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") % PLAYERS


def rotate_absolute(values: np.ndarray, perspective: int) -> np.ndarray:
    order = [(perspective + offset) % PLAYERS for offset in range(PLAYERS)]
    return np.asarray(values)[order]


PERMUTATIONS = tuple(itertools.permutations(range(PLAYERS)))


def placement_label(scores: np.ndarray, perspective: int) -> int:
    absolute_order = sorted(range(PLAYERS), key=lambda player: (-int(scores[player]), player))
    rank_by_player = np.empty(PLAYERS, dtype=np.uint8)
    for rank, player in enumerate(absolute_order):
        rank_by_player[player] = rank
    relative = tuple(int(value) for value in rotate_absolute(rank_by_player, perspective))
    return PERMUTATIONS.index(relative)


def rotated_future(
    annotation: FutureAnnotation,
    current_scores: np.ndarray,
    perspective: int,
) -> dict[str, np.ndarray | int]:
    order = [(perspective + offset) % PLAYERS for offset in range(PLAYERS)]
    opponents = relative_players(perspective)
    target_relative = np.full(4, -1, dtype=np.int8)
    inverse = {absolute: relative for relative, absolute in enumerate(order)}
    for relative_actor, absolute_actor in enumerate(order):
        absolute_target = int(annotation.targets[absolute_actor])
        if absolute_target >= 0:
            target_relative[relative_actor] = inverse[absolute_target]
    return {
        "draw": annotation.draw,
        "win": annotation.win[order],
        "deal_in_player": annotation.deal_in[order],
        "target": target_relative,
        "outcome": outcome_class_index(annotation.win[order], target_relative),
        "dora": annotation.dora[list(opponents)],
        "score": annotation.score[list(opponents)],
        "kyoku_delta": (annotation.final_kyoku_scores - current_scores)[order],
        "placement": placement_label(annotation.final_match_scores, perspective),
        "match_score": annotation.final_match_scores[order],
    }
