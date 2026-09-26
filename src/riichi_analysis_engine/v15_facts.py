"""Exact public scalar facts for the v15 entity-owned input contract.

These values are derived only from the visible state and the current player's
rule context. They are stored separately from the legacy tile-plane transport
so no clipped score or round counter can be mistaken for an exact value.
"""

from __future__ import annotations

import numpy as np

from .analysis_state import PublicHistoryState
from .rule_context import RULE_GLOBAL_FEATURE_NAMES

V15_FACTS_SCHEMA_ID = "riichi-analysis-v15-public-facts-v1"
PLAYER_FIELDS = (
    "score_points",
    "rank_index",
    "seat_wind_index",
    "riichi_declared",
    "riichi_accepted",
    "concealed_tile_count",
)
GLOBAL_RULE_FIELDS = tuple(
    name
    for name in RULE_GLOBAL_FEATURE_NAMES
    if name not in {"riichi_declared", "riichi_accepted"}
)
GLOBAL_FIELDS = (
    "round_number",
    "prevailing_wind_index",
    "honba_count",
    "riichi_stick_count",
    *GLOBAL_RULE_FIELDS,
)
PLAYER_COUNT = 4
PLAYER_WIDTH = len(PLAYER_FIELDS)
GLOBAL_WIDTH = len(GLOBAL_FIELDS)
WALL_WIDTH = 1
V15_FACTS_WIDTH = PLAYER_COUNT * PLAYER_WIDTH + GLOBAL_WIDTH + WALL_WIDTH

assert (PLAYER_WIDTH, GLOBAL_WIDTH, V15_FACTS_WIDTH) == (6, 32, 57)


def encode_v15_facts(
    state: PublicHistoryState, perspective: int, rule_context: np.ndarray
) -> np.ndarray:
    """Return 57 original-unit facts, with no clipping or private tile identity."""

    if not 0 <= perspective < PLAYER_COUNT:
        raise ValueError("perspective must be one of four players")
    if rule_context.shape[0] < len(RULE_GLOBAL_FEATURE_NAMES):
        raise ValueError("rule context has no global rule facts")
    scores = [int(state.scores[(perspective + i) % 4]) for i in range(4)]
    ranks = sorted(
        range(4), key=lambda absolute: (-int(state.scores[absolute]), absolute)
    )
    rank_of = {absolute: rank for rank, absolute in enumerate(ranks)}
    players = np.empty((4, PLAYER_WIDTH), dtype=np.float32)
    for relative in range(4):
        absolute = (perspective + relative) % 4
        players[relative] = (
            scores[relative],
            rank_of[absolute],
            (absolute - state.oya) % 4,
            int(state.riichi[absolute]),
            int(state.riichi_accepted[absolute]),
            int(state.concealed_sizes[absolute]),
        )
    rule_values = rule_context[: len(RULE_GLOBAL_FEATURE_NAMES), 0]
    selected_rules = np.asarray(
        [
            rule_values[RULE_GLOBAL_FEATURE_NAMES.index(name)]
            for name in GLOBAL_RULE_FIELDS
        ],
        dtype=np.float32,
    )
    global_values = np.concatenate(
        (
            np.asarray(
                [state.kyoku, state.bakaze, state.honba, state.kyotaku],
                dtype=np.float32,
            ),
            selected_rules,
        )
    )
    return np.concatenate(
        (
            players.reshape(-1),
            global_values,
            np.asarray([state.wall_remaining], dtype=np.float32),
        )
    )
