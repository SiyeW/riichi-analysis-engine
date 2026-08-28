from __future__ import annotations

OBS_VERSION = 4
OBS_CHANNELS = 1012
TILE_TYPES = 34
ACTION_SPACE = 46
PLAYERS = 4
OPPONENTS = 3
SHANTEN_CLASSES = 7
COUNT_CLASSES = 5

TILES_34 = tuple(
    [f"{n}{s}" for s in "mps" for n in range(1, 10)]
    + ["E", "S", "W", "N", "P", "F", "C"]
)

TILES_37 = (
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C", "5mr", "5pr", "5sr",
)

RED_TILES = ("5mr", "5pr", "5sr")
RED_TILE_TO_INDEX = {tile: index for index, tile in enumerate(RED_TILES)}

TILE37_TO_ACTION = {tile: index for index, tile in enumerate(TILES_37)}
TILE34_TO_INDEX = {tile: index for index, tile in enumerate(TILES_34)}


def deaka(tile: str) -> str:
    return tile.removesuffix("r")


def tile34_index(tile: str) -> int:
    return TILE34_TO_INDEX[deaka(tile)]


def relative_players(perspective: int) -> tuple[int, int, int]:
    return tuple((perspective + offset) % PLAYERS for offset in (1, 2, 3))
