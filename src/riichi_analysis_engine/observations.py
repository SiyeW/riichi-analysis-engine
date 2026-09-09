from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .constants import MORTAL_OBS_CHANNELS, OBS_CHANNELS, RANK_FEATURE_CHANNELS, TILE_TYPES
from .observation_layout import MORTAL_ANALYSIS_CHANNELS


def all_player_rank_features(scores: Sequence[int]) -> np.ndarray:
    """Encode every relative player's rank as four one-hot tile planes."""

    if len(scores) != 4:
        raise ValueError("exactly four scores are required")
    order = sorted(range(4), key=lambda player: (-int(scores[player]), player))
    ranks = np.empty(4, dtype=np.int8)
    for rank, player in enumerate(order):
        ranks[player] = rank
    features = np.zeros((RANK_FEATURE_CHANNELS, TILE_TYPES), dtype=np.float32)
    for player, rank in enumerate(ranks):
        features[player * 4 + int(rank)].fill(1.0)
    return features


def add_all_player_ranks(observation: np.ndarray, scores: Sequence[int]) -> np.ndarray:
    """Insert all-player rank features before Mortal's decision-only tail."""

    source = np.asarray(observation, dtype=np.float32)
    if source.shape != (MORTAL_OBS_CHANNELS, TILE_TYPES):
        raise ValueError(f"wrong Mortal v4 observation shape: {source.shape}")
    result = np.concatenate(
        (
            source[:MORTAL_ANALYSIS_CHANNELS],
            all_player_rank_features(scores),
            source[MORTAL_ANALYSIS_CHANNELS:],
        ),
        axis=0,
    )
    if result.shape != (OBS_CHANNELS, TILE_TYPES):
        raise RuntimeError("rank-augmented observation has the wrong shape")
    return result
