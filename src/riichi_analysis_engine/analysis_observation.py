"""Public-history observation planes used by the separated v9 input.

The layout exactly preserves the proven v2.3 analysis encoder: Mortal's
tile-aligned public state and KawaItem timeline, plus only the minimal event
cursor required by a predictor that runs after every mjai event.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .analysis_state import (
    _FRAME_EVENTS,
    PLAYERS,
    TILE_DIM,
    VISIBILITY_HIDDEN,
    GameState,
    KawaItem,
    _t37_to_t34,
    normalize_visibility_mode,
    tile_index,
)

TILE_AXIS = 34
FIRST_KAWA = 6
RECENT_KAWA = 18
MAX_FUURO = 4
PLANE_SCHEMA_ID = "riichi-analysis-public-history-v1"

KAWA_FIELDS = (
    "call_consumed_0",
    "call_consumed_1",
    "kan",
    "discard",
    "red",
    "dora",
    "tedashi",
    "riichi",
)
EVENT_TYPES = tuple(sorted(_FRAME_EVENTS))


def _make_channel_names() -> tuple[str, ...]:
    names: list[str] = []
    names.extend(f"hand_count_{count}" for count in range(1, 5))
    names.extend(f"hand_red_{suit}" for suit in "mps")

    for player in range(PLAYERS):
        names.extend((f"score_p{player}", f"score_low_p{player}"))
    for player in range(PLAYERS):
        names.extend(f"rank_p{player}_r{rank}" for rank in range(PLAYERS))
    names.extend(f"kyoku_{index}" for index in range(4))
    names.extend(("honba", "kyotaku", "bakaze", "jikaze", "grand_kyoku"))

    names.extend(f"dora_indicator_count_{count}" for count in range(1, 5))
    names.extend(f"dora_indicator_red_{suit}" for suit in "mps")

    for player in range(PLAYERS):
        for slot in range(FIRST_KAWA):
            names.extend(f"kawa_p{player}_first_{slot}_{field}" for field in KAWA_FIELDS)
        for slot in range(RECENT_KAWA):
            names.extend(f"kawa_p{player}_recent_{slot}_{field}" for field in KAWA_FIELDS)
        names.append(f"kawa_decay_p{player}_discard")
        names.extend(
            (
                f"kawa_decay_p{player}_tedashi",
                f"kawa_decay_p{player}_riichi",
            )
        )

    names.append("wall")
    for player in range(PLAYERS):
        names.extend(f"river_p{player}_count_{count}" for count in range(1, 5))
        names.extend(f"river_p{player}_red_{suit}" for suit in "mps")
    for player in range(PLAYERS):
        for slot in range(MAX_FUURO):
            names.extend(f"fuuro_p{player}_slot_{slot}_count_{count}" for count in range(1, 5))
            names.append(f"fuuro_p{player}_slot_{slot}_red")
    names.extend(f"ankan_p{player}" for player in range(PLAYERS))
    names.append("tiles_seen")

    for player in range(PLAYERS):
        names.extend(
            (
                f"last_tedashi_p{player}",
                f"last_tedashi_red_p{player}",
                f"last_tedashi_dora_p{player}",
                f"riichi_discard_p{player}",
                f"riichi_discard_red_p{player}",
                f"riichi_discard_dora_p{player}",
                f"riichi_declared_p{player}",
                f"riichi_accepted_p{player}",
            )
        )

    names.extend(f"current_{event_type}" for event_type in EVENT_TYPES)
    names.extend(f"current_actor_p{player}" for player in range(PLAYERS))
    names.extend(f"current_tile_count_{count}" for count in range(1, 5))
    names.extend(f"current_tile_red_{suit}" for suit in "mps")
    names.extend(("current_called_tile", "current_called_red"))
    names.extend(f"current_target_p{player}" for player in range(PLAYERS))
    return tuple(names)


PLANE_CHANNEL_NAMES = _make_channel_names()
PLANE_CHANNEL_INDEX = {name: index for index, name in enumerate(PLANE_CHANNEL_NAMES)}
PLANE_CHANNELS = len(PLANE_CHANNEL_NAMES)
_TILE37_TO_34 = np.asarray([_t37_to_t34(index) for index in range(TILE_DIM)])


def _channel(plane: np.ndarray, name: str) -> np.ndarray:
    return plane[PLANE_CHANNEL_INDEX[name]]


def _broadcast(plane: np.ndarray, name: str, value: float = 1.0) -> None:
    _channel(plane, name).fill(value)


def _red_suit(tile37: int) -> int | None:
    return {5: 0, 15: 1, 25: 2}.get(tile37)


def _encode_tile_multiset(
    plane: np.ndarray,
    prefix: str,
    tiles: Iterable[int],
    *,
    red_channels: bool = True,
) -> None:
    counts = np.zeros(TILE_AXIS, dtype=np.uint8)
    reds = [False, False, False]
    for tile37 in tiles:
        if tile37 is None:
            continue
        tile = int(tile37)
        tile34 = _t37_to_t34(tile)
        counts[tile34] += 1
        red = _red_suit(tile)
        if red is not None:
            reds[red] = True
    for tile34, count in enumerate(counts):
        for copy in range(int(count)):
            _channel(plane, f"{prefix}_count_{copy + 1}")[tile34] = 1.0
    if red_channels:
        for suit, present in zip("mps", reds):
            if present:
                _broadcast(plane, f"{prefix}_red_{suit}")


def _encode_kawa_item(
    plane: np.ndarray,
    prefix: str,
    item: KawaItem | None,
) -> None:
    if item is None:
        return
    if item.call is not None:
        consumed = sorted(_t37_to_t34(tile) for tile in item.call.consumed)
        if consumed:
            _channel(plane, f"{prefix}_call_consumed_0")[consumed[0]] = 1.0
        if len(consumed) > 1:
            _channel(plane, f"{prefix}_call_consumed_1")[consumed[-1]] = 1.0
    for kan in item.kans:
        for tile in kan.tiles:
            _channel(plane, f"{prefix}_kan")[_t37_to_t34(tile)] = 1.0
    tile34 = _t37_to_t34(item.discard)
    _channel(plane, f"{prefix}_discard")[tile34] = 1.0
    if _red_suit(item.discard) is not None:
        _broadcast(plane, f"{prefix}_red")
    if item.is_dora:
        _broadcast(plane, f"{prefix}_dora")
    if not item.tsumogiri:
        _broadcast(plane, f"{prefix}_tedashi")
    if item.is_riichi:
        _broadcast(plane, f"{prefix}_riichi")


def _latest_item(kawa: list[KawaItem | None], predicate) -> KawaItem | None:
    for item in reversed(kawa):
        if item is not None and predicate(item):
            return item
    return None


def _event_tiles(ev: dict, perspective: int) -> list[int]:
    event_type = ev.get("type")
    if event_type == "tsumo":
        # Match Mortal's decision observation: the complete post-draw hand and
        # the tsumo/actor cursor are present, but the just-drawn tile does not
        # receive a second tile-local pointer. Opponent draws were already
        # hidden; this removes only the redundant self-draw shortcut.
        return []
    if event_type == "dahai":
        return [tile_index(ev["pai"])]
    if event_type in ("chi", "pon", "ankan", "kakan", "daiminkan"):
        names = list(ev.get("consumed", []))
        if ev.get("pai"):
            names.append(ev["pai"])
        return [tile_index(name) for name in names if name != "?"]
    if event_type == "dora":
        marker = ev.get("dora_marker")
        return [] if not marker or marker == "?" else [tile_index(marker)]
    return []


def _encode_current_event(
    plane: np.ndarray,
    ev: dict,
    perspective: int,
) -> None:
    rel = lambda absolute: (absolute - perspective + PLAYERS) % PLAYERS
    event_type = ev.get("type")
    if event_type in EVENT_TYPES:
        _broadcast(plane, f"current_{event_type}")
    actor = ev.get("actor")
    if isinstance(actor, int):
        _broadcast(plane, f"current_actor_p{rel(actor)}")
    _encode_tile_multiset(plane, "current_tile", _event_tiles(ev, perspective))
    if event_type in ("chi", "pon", "kakan", "daiminkan"):
        called = ev.get("pai")
        if called and called != "?":
            called37 = tile_index(called)
            _channel(plane, "current_called_tile")[_t37_to_t34(called37)] = 1.0
            if _red_suit(called37) is not None:
                _broadcast(plane, "current_called_red")
    target = ev.get("target")
    if isinstance(target, int):
        _broadcast(plane, f"current_target_p{rel(target)}")


def encode_tile_plane(
    ev: dict,
    state: GameState,
    perspective: int,
    visibility_mode: str = VISIBILITY_HIDDEN,
) -> np.ndarray:
    """Encode one post-event observation. v2.3 never materializes oracle input."""
    if normalize_visibility_mode(visibility_mode) != VISIBILITY_HIDDEN:
        raise ValueError("v2.3 observations are hidden-only")
    plane = np.zeros((PLANE_CHANNELS, TILE_AXIS), dtype=np.float32)
    rel = lambda absolute: (absolute - perspective + PLAYERS) % PLAYERS

    hand = state.tehais[perspective]
    hand34 = np.bincount(
        _TILE37_TO_34,
        weights=hand.astype(np.float32),
        minlength=TILE_AXIS,
    ).astype(np.uint8)
    for tile34, count in enumerate(hand34):
        for copy in range(int(count)):
            _channel(plane, f"hand_count_{copy + 1}")[tile34] = 1.0
    for tile37, count in enumerate(hand):
        red = _red_suit(tile37)
        if red is not None and count:
            _broadcast(plane, f"hand_red_{'mps'[red]}")

    relative_scores = [state.scores[(perspective + player) % PLAYERS] for player in range(PLAYERS)]
    for player, score in enumerate(relative_scores):
        _broadcast(plane, f"score_p{player}", float(np.clip(score, 0, 100000)) / 100000.0)
        _broadcast(plane, f"score_low_p{player}", float(np.clip(score, 0, 30000)) / 30000.0)
    absolute_by_rank = sorted(
        range(PLAYERS),
        key=lambda absolute: (-state.scores[absolute], absolute),
    )
    rank_by_absolute = {
        absolute: rank for rank, absolute in enumerate(absolute_by_rank)
    }
    for player in range(PLAYERS):
        absolute = (perspective + player) % PLAYERS
        _broadcast(plane, f"rank_p{player}_r{rank_by_absolute[absolute]}")
    _broadcast(plane, f"kyoku_{max(0, min(3, state.kyoku - 1))}")
    _broadcast(plane, "honba", min(state.honba, 10) / 10.0)
    _broadcast(plane, "kyotaku", min(state.kyotaku, 10) / 10.0)
    _channel(plane, "bakaze")[27 + state.bakaze] = 1.0
    _channel(plane, "jikaze")[27 + ((perspective - state.oya + PLAYERS) % PLAYERS)] = 1.0
    _broadcast(
        plane,
        "grand_kyoku",
        min(7, state.bakaze * 4 + max(0, state.kyoku - 1)) / 7.0,
    )
    _encode_tile_multiset(plane, "dora_indicator", state.dora_indicators)

    kawa_view = state.kawa_views[perspective]
    max_kawa_len = max((len(kawa) for kawa in kawa_view), default=1)
    for player, kawa in enumerate(kawa_view):
        for slot, item in enumerate(kawa[:FIRST_KAWA]):
            _encode_kawa_item(
                plane,
                f"kawa_p{player}_first_{slot}",
                item,
            )
        for slot, item in enumerate(reversed(kawa[-RECENT_KAWA:])):
            _encode_kawa_item(
                plane,
                f"kawa_p{player}_recent_{slot}",
                item,
            )
        for turn, item in enumerate(kawa):
            if item is None:
                continue
            value = float(np.exp(-0.2 * (max_kawa_len - 1 - turn)))
            tile34 = _t37_to_t34(item.discard)
            _channel(plane, f"kawa_decay_p{player}_discard")[tile34] = max(
                _channel(plane, f"kawa_decay_p{player}_discard")[tile34],
                value,
            )
            if not item.tsumogiri:
                _channel(plane, f"kawa_decay_p{player}_tedashi")[tile34] = max(
                    _channel(plane, f"kawa_decay_p{player}_tedashi")[tile34],
                    value,
                )
            if item.is_riichi:
                _channel(plane, f"kawa_decay_p{player}_riichi")[tile34] = max(
                    _channel(plane, f"kawa_decay_p{player}_riichi")[tile34],
                    value,
                )

    _broadcast(plane, "wall", min(state.wall_remaining, 69) / 69.0)
    for absolute in range(PLAYERS):
        player = rel(absolute)
        _encode_tile_multiset(
            plane,
            f"river_p{player}",
            (tile for tile, _ in state.rivers[absolute]),
        )
        open_melds = [meld for meld in state.melds[absolute] if not meld[1]]
        for slot, (counts37, _closed) in enumerate(open_melds[:MAX_FUURO]):
            tiles = [tile for tile, count in enumerate(counts37) for _ in range(int(count))]
            _encode_tile_multiset(
                plane,
                f"fuuro_p{player}_slot_{slot}",
                tiles,
                red_channels=False,
            )
            if any(_red_suit(tile) is not None for tile in tiles):
                _broadcast(plane, f"fuuro_p{player}_slot_{slot}_red")
        for tile34 in state.ankans[absolute]:
            _channel(plane, f"ankan_p{player}")[tile34] = 1.0

    seen34 = np.bincount(
        _TILE37_TO_34,
        weights=state.public_seen.astype(np.float32),
        minlength=TILE_AXIS,
    )
    _channel(plane, "tiles_seen")[:] = seen34 / 4.0

    for player in range(PLAYERS):
        kawa = kawa_view[player]
        last_tedashi = _latest_item(kawa, lambda item: not item.tsumogiri)
        riichi_item = _latest_item(kawa, lambda item: item.is_riichi)
        for prefix, item in (
            (f"last_tedashi_p{player}", last_tedashi),
            (f"riichi_discard_p{player}", riichi_item),
        ):
            if item is None:
                continue
            _channel(plane, prefix)[_t37_to_t34(item.discard)] = 1.0
            if _red_suit(item.discard) is not None:
                _broadcast(plane, prefix.replace(f"_p{player}", f"_red_p{player}"))
            if item.is_dora:
                _broadcast(plane, prefix.replace(f"_p{player}", f"_dora_p{player}"))
        absolute = (perspective + player) % PLAYERS
        if state.riichi[absolute]:
            _broadcast(plane, f"riichi_declared_p{player}")
        if state.riichi_accepted[absolute]:
            _broadcast(plane, f"riichi_accepted_p{player}")

    _encode_current_event(plane, ev, perspective)
    return plane


_HAND_CHANNELS = tuple(
    [f"hand_count_{count}" for count in range(1, 5)]
    + [f"hand_red_{suit}" for suit in "mps"]
)
_DORA_CHANNELS = tuple(
    [f"dora_indicator_count_{count}" for count in range(1, 5)]
    + [f"dora_indicator_red_{suit}" for suit in "mps"]
)
_CURRENT_CHANNEL_INDICES = np.asarray(
    [
        index
        for index, name in enumerate(PLANE_CHANNEL_NAMES)
        if name.startswith("current_")
    ],
    dtype=np.int64,
)


def _clear_named_channels(plane: np.ndarray, names: Iterable[str]) -> None:
    for name in names:
        _channel(plane, name).fill(0.0)


def _refresh_hand(plane: np.ndarray, state: GameState, perspective: int) -> None:
    _clear_named_channels(plane, _HAND_CHANNELS)
    hand = state.tehais[perspective]
    hand34 = np.bincount(
        _TILE37_TO_34,
        weights=hand.astype(np.float32),
        minlength=TILE_AXIS,
    ).astype(np.uint8)
    for tile34, count in enumerate(hand34):
        for copy in range(int(count)):
            _channel(plane, f"hand_count_{copy + 1}")[tile34] = 1.0
    for tile37, count in enumerate(hand):
        red = _red_suit(tile37)
        if red is not None and count:
            _broadcast(plane, f"hand_red_{'mps'[red]}")


def _refresh_metadata(plane: np.ndarray, state: GameState, perspective: int) -> None:
    names: list[str] = []
    for player in range(PLAYERS):
        names.extend((f"score_p{player}", f"score_low_p{player}"))
        names.extend(f"rank_p{player}_r{rank}" for rank in range(PLAYERS))
    names.extend(f"kyoku_{index}" for index in range(4))
    names.extend(("honba", "kyotaku", "bakaze", "jikaze", "grand_kyoku"))
    _clear_named_channels(plane, names)

    relative_scores = [
        state.scores[(perspective + player) % PLAYERS] for player in range(PLAYERS)
    ]
    for player, score in enumerate(relative_scores):
        _broadcast(
            plane,
            f"score_p{player}",
            float(np.clip(score, 0, 100000)) / 100000.0,
        )
        _broadcast(
            plane,
            f"score_low_p{player}",
            float(np.clip(score, 0, 30000)) / 30000.0,
        )
    absolute_by_rank = sorted(
        range(PLAYERS),
        key=lambda absolute: (-state.scores[absolute], absolute),
    )
    rank_by_absolute = {
        absolute: rank for rank, absolute in enumerate(absolute_by_rank)
    }
    for player in range(PLAYERS):
        absolute = (perspective + player) % PLAYERS
        _broadcast(plane, f"rank_p{player}_r{rank_by_absolute[absolute]}")
    _broadcast(plane, f"kyoku_{max(0, min(3, state.kyoku - 1))}")
    _broadcast(plane, "honba", min(state.honba, 10) / 10.0)
    _broadcast(plane, "kyotaku", min(state.kyotaku, 10) / 10.0)
    _channel(plane, "bakaze")[27 + state.bakaze] = 1.0
    _channel(plane, "jikaze")[
        27 + ((perspective - state.oya + PLAYERS) % PLAYERS)
    ] = 1.0
    _broadcast(
        plane,
        "grand_kyoku",
        min(7, state.bakaze * 4 + max(0, state.kyoku - 1)) / 7.0,
    )


def _refresh_dora(plane: np.ndarray, state: GameState) -> None:
    _clear_named_channels(plane, _DORA_CHANNELS)
    _encode_tile_multiset(plane, "dora_indicator", state.dora_indicators)


def _refresh_wall(plane: np.ndarray, state: GameState) -> None:
    _broadcast(plane, "wall", min(state.wall_remaining, 69) / 69.0)


def _refresh_river(
    plane: np.ndarray,
    state: GameState,
    perspective: int,
    absolute: int,
) -> None:
    player = (absolute - perspective + PLAYERS) % PLAYERS
    names = tuple(
        [f"river_p{player}_count_{count}" for count in range(1, 5)]
        + [f"river_p{player}_red_{suit}" for suit in "mps"]
    )
    _clear_named_channels(plane, names)
    _encode_tile_multiset(
        plane,
        f"river_p{player}",
        (tile for tile, _ in state.rivers[absolute]),
    )


def _refresh_melds(
    plane: np.ndarray,
    state: GameState,
    perspective: int,
    absolute: int,
) -> None:
    player = (absolute - perspective + PLAYERS) % PLAYERS
    names: list[str] = []
    for slot in range(MAX_FUURO):
        names.extend(
            f"fuuro_p{player}_slot_{slot}_count_{count}" for count in range(1, 5)
        )
        names.append(f"fuuro_p{player}_slot_{slot}_red")
    names.append(f"ankan_p{player}")
    _clear_named_channels(plane, names)

    open_melds = [meld for meld in state.melds[absolute] if not meld[1]]
    for slot, (counts37, _closed) in enumerate(open_melds[:MAX_FUURO]):
        tiles = [
            tile
            for tile, count in enumerate(counts37)
            for _ in range(int(count))
        ]
        _encode_tile_multiset(
            plane,
            f"fuuro_p{player}_slot_{slot}",
            tiles,
            red_channels=False,
        )
        if any(_red_suit(tile) is not None for tile in tiles):
            _broadcast(plane, f"fuuro_p{player}_slot_{slot}_red")
    for tile34 in state.ankans[absolute]:
        _channel(plane, f"ankan_p{player}")[tile34] = 1.0


def _refresh_public_seen(plane: np.ndarray, state: GameState) -> None:
    seen34 = np.bincount(
        _TILE37_TO_34,
        weights=state.public_seen.astype(np.float32),
        minlength=TILE_AXIS,
    )
    _channel(plane, "tiles_seen")[:] = seen34 / 4.0


def _refresh_player_status(
    plane: np.ndarray,
    state: GameState,
    perspective: int,
    absolute: int,
) -> None:
    player = (absolute - perspective + PLAYERS) % PLAYERS
    names = (
        f"last_tedashi_p{player}",
        f"last_tedashi_red_p{player}",
        f"last_tedashi_dora_p{player}",
        f"riichi_discard_p{player}",
        f"riichi_discard_red_p{player}",
        f"riichi_discard_dora_p{player}",
        f"riichi_declared_p{player}",
        f"riichi_accepted_p{player}",
    )
    _clear_named_channels(plane, names)
    kawa = state.kawa_views[perspective][player]
    last_tedashi = _latest_item(kawa, lambda item: not item.tsumogiri)
    riichi_item = _latest_item(kawa, lambda item: item.is_riichi)
    for prefix, item in (
        (f"last_tedashi_p{player}", last_tedashi),
        (f"riichi_discard_p{player}", riichi_item),
    ):
        if item is None:
            continue
        _channel(plane, prefix)[_t37_to_t34(item.discard)] = 1.0
        if _red_suit(item.discard) is not None:
            _broadcast(plane, prefix.replace(f"_p{player}", f"_red_p{player}"))
        if item.is_dora:
            _broadcast(plane, prefix.replace(f"_p{player}", f"_dora_p{player}"))
    if state.riichi[absolute]:
        _broadcast(plane, f"riichi_declared_p{player}")
    if state.riichi_accepted[absolute]:
        _broadcast(plane, f"riichi_accepted_p{player}")


class IncrementalTilePlaneEncoder:
    """Maintain four post-event observation bases while a kyoku is replayed."""

    def __init__(self, visibility_mode: str = VISIBILITY_HIDDEN):
        if normalize_visibility_mode(visibility_mode) != VISIBILITY_HIDDEN:
            raise ValueError("v2.3 observations are hidden-only")
        self._bases: np.ndarray | None = None
        self._kawa_lengths = np.zeros((PLAYERS, PLAYERS), dtype=np.int16)
        self._kawa_max = np.zeros(PLAYERS, dtype=np.int16)
        self._decay_turns = np.full(
            (PLAYERS, PLAYERS, 3, TILE_AXIS),
            -1,
            dtype=np.int16,
        )
        self._river_counts = np.zeros((PLAYERS, TILE_AXIS), dtype=np.uint8)
        self._river_reds = np.zeros((PLAYERS, 3), dtype=np.bool_)

    def _initialize(self, state: GameState) -> None:
        self._bases = np.stack(
            [
                encode_tile_plane({}, state, perspective)
                for perspective in range(PLAYERS)
            ],
            axis=0,
        )
        self._kawa_lengths.fill(0)
        self._kawa_max.fill(0)
        self._decay_turns.fill(-1)
        self._river_counts.fill(0)
        self._river_reds.fill(False)
        for absolute, river in enumerate(state.rivers):
            for tile37, _tsumogiri in river:
                self._river_counts[absolute, _t37_to_t34(tile37)] += 1
                red = _red_suit(tile37)
                if red is not None:
                    self._river_reds[absolute, red] = True
        for perspective in range(PLAYERS):
            kawa_view = state.kawa_views[perspective]
            lengths = [len(kawa) for kawa in kawa_view]
            self._kawa_lengths[perspective] = lengths
            self._kawa_max[perspective] = max(lengths, default=0)
            for player, kawa in enumerate(kawa_view):
                for turn, item in enumerate(kawa):
                    self._record_decay_turn(perspective, player, turn, item)

    def _record_decay_turn(
        self,
        perspective: int,
        player: int,
        turn: int,
        item: KawaItem | None,
    ) -> None:
        if item is None:
            return
        tile34 = _t37_to_t34(item.discard)
        self._decay_turns[perspective, player, 0, tile34] = turn
        if not item.tsumogiri:
            self._decay_turns[perspective, player, 1, tile34] = turn
        if item.is_riichi:
            self._decay_turns[perspective, player, 2, tile34] = turn

    def _refresh_decay(self, perspective: int, players: Iterable[int]) -> None:
        assert self._bases is not None
        plane = self._bases[perspective]
        max_len = int(self._kawa_max[perspective])
        suffixes = ("discard", "tedashi", "riichi")
        for player in players:
            for field, suffix in enumerate(suffixes):
                turns = self._decay_turns[perspective, player, field]
                channel = _channel(plane, f"kawa_decay_p{player}_{suffix}")
                channel.fill(0.0)
                present = turns >= 0
                if np.any(present):
                    channel[present] = np.exp(
                        -0.2 * (max_len - 1 - turns[present].astype(np.float64))
                    ).astype(np.float32)

    def _append_kawa_item(
        self,
        perspective: int,
        player: int,
        turn: int,
        item: KawaItem | None,
    ) -> None:
        assert self._bases is not None
        plane = self._bases[perspective]
        if turn < FIRST_KAWA:
            _encode_kawa_item(
                plane,
                f"kawa_p{player}_first_{turn}",
                item,
            )

        recent_start = PLANE_CHANNEL_INDEX[
            f"kawa_p{player}_recent_0_{KAWA_FIELDS[0]}"
        ]
        recent_end = recent_start + RECENT_KAWA * len(KAWA_FIELDS)
        field_count = len(KAWA_FIELDS)
        plane[recent_start + field_count : recent_end] = plane[
            recent_start : recent_end - field_count
        ].copy()
        plane[recent_start : recent_start + field_count].fill(0.0)
        _encode_kawa_item(plane, f"kawa_p{player}_recent_0", item)
        self._record_decay_turn(perspective, player, turn, item)

    def _sync_kawa(self, state: GameState) -> None:
        assert self._bases is not None
        for perspective in range(PLAYERS):
            kawa_view = state.kawa_views[perspective]
            old_lengths = self._kawa_lengths[perspective].copy()
            new_lengths = np.asarray([len(kawa) for kawa in kawa_view], dtype=np.int16)
            if np.any(new_lengths < old_lengths):
                raise RuntimeError("Kawa history moved backwards inside a kyoku")
            changed = np.flatnonzero(new_lengths > old_lengths)
            if not len(changed):
                continue
            old_max = int(self._kawa_max[perspective])
            new_max = int(new_lengths.max(initial=0))
            for player in changed:
                for turn in range(int(old_lengths[player]), int(new_lengths[player])):
                    self._append_kawa_item(
                        perspective,
                        int(player),
                        turn,
                        kawa_view[int(player)][turn],
                    )
            self._kawa_lengths[perspective] = new_lengths
            self._kawa_max[perspective] = new_max
            refresh_players: Iterable[int]
            if new_max != old_max:
                refresh_players = range(PLAYERS)
            else:
                refresh_players = (int(player) for player in changed)
            self._refresh_decay(perspective, refresh_players)

    def _append_river(self, absolute: int, tile37: int) -> None:
        assert self._bases is not None
        tile34 = _t37_to_t34(tile37)
        self._river_counts[absolute, tile34] += 1
        count = int(self._river_counts[absolute, tile34])
        red = _red_suit(tile37)
        if red is not None:
            self._river_reds[absolute, red] = True
        for perspective, plane in enumerate(self._bases):
            player = (absolute - perspective + PLAYERS) % PLAYERS
            if 1 <= count <= 4:
                _channel(plane, f"river_p{player}_count_{count}")[tile34] = 1.0
            if red is not None:
                _broadcast(plane, f"river_p{player}_red_{'mps'[red]}")

    def advance(self, ev: dict, state: GameState) -> None:
        event_type = ev.get("type")
        if self._bases is None or event_type == "start_kyoku":
            self._initialize(state)
            return

        if event_type in ("dahai", "pon", "daiminkan"):
            self._sync_kawa(state)
        actor = ev.get("actor")
        if event_type == "tsumo":
            if isinstance(actor, int):
                _refresh_hand(self._bases[actor], state, actor)
            for plane in self._bases:
                _refresh_wall(plane, state)
        elif event_type == "dahai":
            if isinstance(actor, int):
                _refresh_hand(self._bases[actor], state, actor)
                self._append_river(actor, tile_index(ev["pai"]))
                for perspective, plane in enumerate(self._bases):
                    _refresh_player_status(plane, state, perspective, actor)
            for plane in self._bases:
                _refresh_public_seen(plane, state)
        elif event_type in ("chi", "pon", "ankan", "kakan", "daiminkan"):
            if isinstance(actor, int):
                _refresh_hand(self._bases[actor], state, actor)
                for perspective, plane in enumerate(self._bases):
                    _refresh_melds(plane, state, perspective, actor)
            for plane in self._bases:
                _refresh_public_seen(plane, state)
        elif event_type == "dora":
            for plane in self._bases:
                _refresh_dora(plane, state)
                _refresh_public_seen(plane, state)
        elif event_type == "reach":
            if isinstance(actor, int):
                for perspective, plane in enumerate(self._bases):
                    _refresh_player_status(plane, state, perspective, actor)
        elif event_type == "reach_accepted":
            for perspective, plane in enumerate(self._bases):
                _refresh_metadata(plane, state, perspective)
                if isinstance(actor, int):
                    _refresh_player_status(plane, state, perspective, actor)

    def encode(self, ev: dict, perspective: int) -> np.ndarray:
        if self._bases is None:
            raise RuntimeError("Incremental encoder has not seen start_kyoku")
        if perspective < 0 or perspective >= PLAYERS:
            raise ValueError(f"Invalid perspective {perspective}")
        plane = self._bases[perspective].copy()
        plane[_CURRENT_CHANNEL_INDICES] = 0.0
        _encode_current_event(plane, ev, perspective)
        return plane


def channel_index(name: str) -> int:
    return PLANE_CHANNEL_INDEX[name]
