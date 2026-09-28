"""Full executable candidate identity for the next supervised policy schema."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .constants import TILE37_TO_ACTION

ACTION_KINDS = (
    "dahai",
    "reach",
    "chi",
    "pon",
    "daiminkan",
    "ankan",
    "kakan",
    "hora",
    "ryukyoku",
    "none",
)
ACTION_KIND_TO_ID = {name: index for index, name in enumerate(ACTION_KINDS)}

KIND_START = 0
TILE_START = KIND_START + len(ACTION_KINDS)
TARGET_START = TILE_START + 38
TSUMOGIRI_SLOT = TARGET_START + 5
CONSUMED_START = TSUMOGIRI_SLOT + 1
SKIP_ANKAN_SLOT = CONSUMED_START + 37
CANDIDATE_FEATURES = SKIP_ANKAN_SLOT + 1
CANDIDATE_CODE_WIDTH = 9
CANDIDATE_CAPACITY = 32


def candidate_signature(action: dict[str, Any], actor: int) -> tuple[Any, ...]:
    """Compare candidates independently of consumed order and opaque host ID."""

    kind = action.get("type")
    if kind not in ACTION_KIND_TO_ID:
        raise ValueError(f"unsupported action kind {kind!r}")
    if action.get("actor") != actor:
        raise ValueError("candidate actor differs from the controlled seat")
    if kind != "none" and "variant" in action:
        raise ValueError("variant is only defined for a none action")
    variant = action.get("variant") if kind == "none" else None
    if variant not in {None, "none", "skip-ankan"}:
        raise ValueError(f"unsupported none-action variant {variant!r}")
    if variant == "none":
        variant = None
    tile = action.get("pai")
    if tile is not None and tile not in TILE37_TO_ACTION:
        raise ValueError(f"unsupported physical tile {tile!r}")
    consumed = action.get("consumed", [])
    if not isinstance(consumed, list) or len(consumed) > 4:
        raise ValueError("consumed must be a list of at most four physical tiles")
    if any(item not in TILE37_TO_ACTION for item in consumed):
        raise ValueError("consumed contains an unsupported physical tile")
    target = action.get("target")
    if target is not None and (not isinstance(target, int) or not 0 <= target < 4):
        raise ValueError("candidate target must be a seat index")
    if kind == "dahai" and (
        tile is None or not isinstance(action.get("tsumogiri"), bool)
    ):
        raise ValueError("discard requires a physical tile and tsumogiri flag")
    if kind in {"chi", "pon", "daiminkan"}:
        required_count = 3 if kind == "daiminkan" else 2
        if (
            target is None
            or target == actor
            or tile is None
            or len(consumed) != required_count
        ):
            raise ValueError(
                f"{kind} requires target, tile and {required_count} consumed tiles"
            )
    if kind == "ankan" and len(consumed) != 4:
        raise ValueError("ankan requires four consumed tiles")
    if kind == "kakan" and (tile is None or len(consumed) != 3):
        raise ValueError("kakan requires a tile and three consumed tiles")
    if kind == "hora" and (target is None or tile is None):
        raise ValueError("hora requires target and winning tile")
    if variant == "skip-ankan" and (
        tile is None or action.get("tsumogiri") is not True
    ):
        raise ValueError("skip-ankan requires the forced drawn tile")
    if kind in {"reach", "ryukyoku"} and (
        tile is not None or target is not None or consumed
    ):
        raise ValueError(f"{kind} has unexpected tile or target fields")
    if (
        kind == "none"
        and variant is None
        and (tile is not None or target is not None or consumed)
    ):
        raise ValueError("plain none has unexpected tile or target fields")
    return (
        kind,
        tile,
        None if target is None else (target - actor) % 4,
        bool(action.get("tsumogiri", False)),
        tuple(sorted(Counter(consumed).items())),
        variant,
    )


def encode_candidate(action: dict[str, Any], actor: int) -> np.ndarray:
    """Fixed, injective fields for the protocol's supported action semantics."""

    kind, tile, relative_target, tsumogiri, consumed, variant = candidate_signature(
        action, actor
    )
    values = np.zeros(CANDIDATE_FEATURES, dtype=np.float32)
    values[KIND_START + ACTION_KIND_TO_ID[kind]] = 1
    values[TILE_START + (0 if tile is None else TILE37_TO_ACTION[tile] + 1)] = 1
    values[TARGET_START + (0 if relative_target is None else relative_target + 1)] = 1
    values[TSUMOGIRI_SLOT] = float(tsumogiri)
    for physical_tile, count in consumed:
        values[CONSUMED_START + TILE37_TO_ACTION[physical_tile]] = count / 4
    values[SKIP_ANKAN_SLOT] = float(variant == "skip-ankan")
    return values


def encode_candidate_codes(action: dict[str, Any], actor: int) -> np.ndarray:
    """Nine-byte storage form of one complete action, independent of list order."""

    kind, tile, relative_target, tsumogiri, consumed, variant = candidate_signature(
        action, actor
    )
    values = np.zeros(CANDIDATE_CODE_WIDTH, dtype=np.uint8)
    values[0] = ACTION_KIND_TO_ID[kind] + 1
    values[1] = 0 if tile is None else TILE37_TO_ACTION[tile] + 1
    values[2] = 0 if relative_target is None else relative_target + 1
    values[3] = int(tsumogiri)
    consumed_codes = sorted(
        (TILE37_TO_ACTION[physical_tile] + 1)
        for physical_tile, count in consumed
        for _ in range(count)
    )
    values[4 : 4 + len(consumed_codes)] = consumed_codes
    values[8] = int(variant == "skip-ankan")
    return values


def candidate_features_from_codes(codes: Tensor) -> Tensor:
    """Expand compact exact codes into the common candidate query fields."""

    if codes.shape[-1] != CANDIDATE_CODE_WIDTH:
        raise ValueError("candidate codes require nine fields")
    fields = codes.long()
    parts = [
        F.one_hot(fields[..., 0], len(ACTION_KINDS) + 1)[..., 1:],
        F.one_hot(fields[..., 1], 38),
        F.one_hot(fields[..., 2], 5),
        fields[..., 3:4],
    ]
    consumed = fields.new_zeros(*fields.shape[:-1], 38)
    consumed.scatter_add_(-1, fields[..., 4:8], torch.ones_like(fields[..., 4:8]))
    parts.append(consumed[..., 1:] / 4)
    parts.append(fields[..., 8:9])
    features = torch.cat(parts, dim=-1).float()
    return features * (fields[..., :1] != 0)


def encode_candidate_set(candidates: list[dict[str, Any]], actor: int) -> np.ndarray:
    if not candidates:
        raise ValueError("candidate set must be non-empty")
    signatures = [candidate_signature(action, actor) for action in candidates]
    if len(set(signatures)) != len(signatures):
        raise ValueError("candidate set contains duplicate executable actions")
    return np.stack([encode_candidate(action, actor) for action in candidates])


def pack_candidate_set(
    candidates: list[dict[str, Any]], actor: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-capacity compact row; overflow fails rather than truncating."""

    encode_candidate_set(candidates, actor)  # validate identity and duplicates
    if len(candidates) > CANDIDATE_CAPACITY:
        raise ValueError("legal candidate set exceeds the frozen capacity")
    codes = np.zeros((CANDIDATE_CAPACITY, CANDIDATE_CODE_WIDTH), dtype=np.uint8)
    mask = np.zeros(CANDIDATE_CAPACITY, dtype=bool)
    for index, action in enumerate(candidates):
        codes[index] = encode_candidate_codes(action, actor)
        mask[index] = True
    return codes, mask
