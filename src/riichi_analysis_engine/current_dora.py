"""Current visible-dora counting, separate from future winning-hand dora."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np

from .constants import TILE37_TO_ACTION, TILES_37, tile34_index
from .physical_tile_features import dora_tile_index
from .prediction_values import DORA_TAIL_START


def visible_dora_weights(markers: Iterable[str]) -> np.ndarray:
    """A physical tile's value under currently face-up indicators plus aka."""

    target_counts = Counter(dora_tile_index(tile34_index(marker)) for marker in markers)
    return np.asarray(
        [
            target_counts[tile34_index(tile)] + int(tile.endswith("r"))
            for tile in TILES_37
        ],
        dtype=np.int16,
    )


def current_concealed_dora(hand: Counter[str], markers: Iterable[str]) -> int:
    weights = visible_dora_weights(markers)
    return sum(
        int(count) * int(weights[TILE37_TO_ACTION[tile]])
        for tile, count in hand.items()
    )


def known_meld_dora(meld_counts: np.ndarray, markers: Iterable[str]) -> np.ndarray:
    """One exact known count per relative player, including exposed ankans."""

    counts = np.asarray(meld_counts)
    if counts.shape != (4, 37) or (counts < 0).any():
        raise ValueError("meld counts must be non-negative 4 x 37 physical tiles")
    return counts.astype(np.int32) @ visible_dora_weights(markers).astype(np.int32)


def add_known_dora_to_distribution(
    concealed_probabilities: np.ndarray, known_count: int
) -> np.ndarray:
    """Shift exact 0..6/7+ mass without inventing values below public knowledge."""

    source = np.asarray(concealed_probabilities)
    if source.shape != (DORA_TAIL_START + 1,) or known_count < 0:
        raise ValueError("dora probabilities or known count have the wrong shape")
    result = np.zeros_like(source)
    for concealed_value, probability in enumerate(source):
        result[min(concealed_value + known_count, DORA_TAIL_START)] += probability
    return result
