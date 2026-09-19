"""Deterministic 37-tile semantic features derived from the public observation.

The public-history encoder stores tile-aligned facts on the established 34-tile
axis and keeps red-five identity in dedicated channels.  New semantic models
need a single, audited boundary that restores 34 non-red tile identities plus
the three red fives without asking a learned layer to rediscover dora rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .analysis_observation import PLANE_CHANNEL_INDEX, PLANE_CHANNELS
from .constants import RED_TILES, TILE_TYPES, TILES_37

PHYSICAL_TILE_TYPES = len(TILES_37)
RED_BASE_TILE_INDICES = (4, 13, 22)
RED_ENTITY_FEATURE_NAMES = (
    "hand",
    "dora_indicator",
    "river_p0",
    "river_p1",
    "river_p2",
    "river_p3",
    "current_tile",
)


def dora_tile_index(marker: int) -> int:
    """Return the 34-tile identity indicated by one visible dora marker."""

    if not 0 <= marker < TILE_TYPES:
        raise ValueError("dora marker must be a 34-tile index")
    if marker < 27:
        suit_start = marker // 9 * 9
        return suit_start + (marker - suit_start + 1) % 9
    if marker <= 30:
        return 27 + (marker - 27 + 1) % 4
    return 31 + (marker - 31 + 1) % 3


DORA_TARGET_BY_MARKER = tuple(dora_tile_index(marker) for marker in range(TILE_TYPES))


@dataclass(frozen=True)
class PhysicalDoraFeatures:
    """Dora attributes attached to the canonical 37 physical tile identities."""

    indicator_count: Tensor
    dora_multiplier: Tensor
    aka_bonus: Tensor

    @property
    def visible_dora_weight(self) -> Tensor:
        return self.dora_multiplier + self.aka_bonus

    def stacked(self) -> Tensor:
        """Return ``[batch, 37, 4]`` features in their documented order."""

        return torch.stack(
            (
                self.indicator_count,
                self.dora_multiplier,
                self.aka_bonus,
                self.visible_dora_weight,
            ),
            dim=-1,
        )


def physical_dora_features(analysis_observation: Tensor) -> PhysicalDoraFeatures:
    """Restore entity-aligned dora facts from a batched public observation.

    ``indicator_count`` distinguishes a red five used as an indicator from a
    normal five.  ``dora_multiplier`` deliberately ignores that distinction:
    either five indicates six.  ``aka_bonus`` is intrinsic to the three red
    entities, so a red five that is also indicated as dora receives both values.
    """

    if analysis_observation.ndim != 3 or tuple(analysis_observation.shape[1:]) != (
        PLANE_CHANNELS,
        TILE_TYPES,
    ):
        raise ValueError(
            "physical dora features require a batched public-history observation"
        )

    count_channels = [
        PLANE_CHANNEL_INDEX[f"dora_indicator_count_{count}"]
        for count in range(1, 5)
    ]
    indicator_count34 = analysis_observation[:, count_channels].sum(dim=1)
    red_indicator_count = torch.stack(
        tuple(
            analysis_observation[:, PLANE_CHANNEL_INDEX[f"dora_indicator_red_{suit}"], 0]
            for suit in "mps"
        ),
        dim=1,
    )

    normal_indicator_count = indicator_count34.clone()
    normal_indicator_count[:, RED_BASE_TILE_INDICES] -= red_indicator_count
    indicator_count37 = torch.cat(
        (normal_indicator_count, red_indicator_count), dim=1
    )

    target = torch.tensor(
        DORA_TARGET_BY_MARKER,
        device=analysis_observation.device,
        dtype=torch.long,
    ).expand(len(analysis_observation), -1)
    dora_multiplier34 = analysis_observation.new_zeros(
        len(analysis_observation), TILE_TYPES
    )
    dora_multiplier34.scatter_add_(1, target, indicator_count34)
    dora_multiplier37 = torch.cat(
        (dora_multiplier34, dora_multiplier34[:, RED_BASE_TILE_INDICES]), dim=1
    )

    aka_bonus = analysis_observation.new_zeros(
        len(analysis_observation), PHYSICAL_TILE_TYPES
    )
    aka_bonus[:, TILE_TYPES:] = 1

    return PhysicalDoraFeatures(
        indicator_count=indicator_count37,
        dora_multiplier=dora_multiplier37,
        aka_bonus=aka_bonus,
    )


def physical_red_features(analysis_observation: Tensor) -> Tensor:
    """Attach broadcast red-five facts to their three physical entities.

    The proven 34-axis encoder broadcasts these flags because it has no red
    tile positions. A semantic model must not ask a learned layer to recover
    which physical five owns them, so this boundary localizes every
    suit-specific public flag before the first projection.
    """

    if analysis_observation.ndim != 3 or tuple(analysis_observation.shape[1:]) != (
        PLANE_CHANNELS,
        TILE_TYPES,
    ):
        raise ValueError(
            "physical red features require a batched public-history observation"
        )
    result = analysis_observation.new_zeros(
        len(analysis_observation), PHYSICAL_TILE_TYPES, len(RED_ENTITY_FEATURE_NAMES)
    )
    for suit_index, suit in enumerate("mps"):
        channels = [
            PLANE_CHANNEL_INDEX[f"hand_red_{suit}"],
            PLANE_CHANNEL_INDEX[f"dora_indicator_red_{suit}"],
            *(
                PLANE_CHANNEL_INDEX[f"river_p{player}_red_{suit}"]
                for player in range(4)
            ),
            PLANE_CHANNEL_INDEX[f"current_tile_red_{suit}"],
        ]
        result[:, TILE_TYPES + suit_index] = analysis_observation[:, channels, 0]
    return result


assert PHYSICAL_TILE_TYPES == TILE_TYPES + len(RED_TILES)
