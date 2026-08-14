from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .constants import TILES_34, TILE37_TO_ACTION, relative_players, tile34_index
from .model import RiichiAnalysisModel

PERMUTATIONS = tuple(itertools.permutations(range(4)))


def _load_player_state() -> Any:
    try:
        from libriichi.state import PlayerState

        return PlayerState
    except ImportError:
        root = os.environ.get("RIICHI_LIBRIICHI_ROOT")
        if not root:
            raise RuntimeError(
                "libriichi is unavailable; set RIICHI_LIBRIICHI_ROOT for a development build"
            ) from None
        sys.path.insert(0, str(Path(root).resolve()))
        from libriichi.state import PlayerState

        return PlayerState


def _finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError("model returned a non-finite value")
    return value


def _distribution(probabilities: np.ndarray, *, first_value: int = 0) -> list[dict[str, float | int]]:
    return [
        {"value": index + first_value, "probability": _finite(probability)}
        for index, probability in enumerate(probabilities)
    ]


def _prediction_from_distribution(probabilities: np.ndarray, *, first_value: int = 0) -> dict[str, Any]:
    values = np.arange(len(probabilities), dtype=np.float32) + first_value
    return {
        "distribution": _distribution(probabilities, first_value=first_value),
        "expectedValue": _finite(np.dot(values, probabilities)),
    }


def _chi_action(action: dict[str, Any]) -> int:
    called = tile34_index(action["pai"])
    consumed = sorted(tile34_index(tile) for tile in action["consumed"])
    return 38 if called < consumed[0] else 39 if called < consumed[1] else 40


def candidate_action_index(action: dict[str, Any]) -> int:
    kind = action.get("type")
    if kind == "dahai":
        return TILE37_TO_ACTION[action["pai"]]
    if kind == "reach":
        return 37
    if kind == "chi":
        return _chi_action(action)
    if kind == "pon":
        return 41
    if kind in {"ankan", "kakan", "daiminkan"}:
        return 42
    if kind == "hora":
        return 43
    if kind == "ryukyoku":
        return 44
    if kind == "none":
        return 45
    raise ValueError(f"unsupported action candidate type: {kind!r}")


class AnalysisRuntime:
    def __init__(self, checkpoint: str | Path, device: str) -> None:
        self.device = torch.device(device)
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if payload.get("format") != "riichi-analysis-model-v1":
            raise RuntimeError("weight file has an unsupported format")
        self.model = RiichiAnalysisModel()
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device).eval()
        self.player_state_type = _load_player_state()
        with torch.inference_mode():
            self.model(torch.zeros(1, 1012, 34, device=self.device))

    def _observation(self, events: list[dict[str, Any]], controlled_seat: int) -> np.ndarray:
        state = self.player_state_type(controlled_seat)
        for event in events:
            state.update(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
        observation, _mask = state.encode_obs(4, False)
        return np.asarray(observation, dtype=np.float32)

    def predict(
        self,
        events: list[dict[str, Any]],
        controlled_seat: int,
        requested: list[dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], float]:
        started = time.perf_counter()
        observation = self._observation(events, controlled_seat)
        tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            raw = self.model(tensor)
        outputs = {name: value[0].float().cpu() for name, value in raw.items()}
        order = [(controlled_seat + offset) % 4 for offset in range(4)]
        opponents = relative_players(controlled_seat)
        results: dict[str, dict[str, Any]] = {}

        shanten = outputs["shanten"].softmax(-1).numpy()
        furiten = outputs["furiten_no_yaku"].sigmoid().numpy()
        results["opponent-shanten"] = {
            "players": [
                {
                    "seat": seat,
                    "shanten": _distribution(shanten[index]),
                    "furitenOrNoYaku": _finite(furiten[index]),
                }
                for index, seat in enumerate(opponents)
            ]
        }
        waits = outputs["deal_in_tile"].sigmoid().numpy()
        results["opponent-deal-in-probability"] = {
            "players": [
                {
                    "seat": seat,
                    "tiles": {tile: _finite(waits[index, tile_index]) for tile_index, tile in enumerate(TILES_34)},
                }
                for index, seat in enumerate(opponents)
            ]
        }
        concealed = outputs["concealed_count"].softmax(-1).numpy()
        results["opponent-concealed-tile-count"] = {
            "players": [
                {
                    "seat": seat,
                    "tiles": {
                        tile: _prediction_from_distribution(concealed[index, tile_index])
                        for tile_index, tile in enumerate(TILES_34)
                    },
                }
                for index, seat in enumerate(opponents)
            ]
        }
        wall = outputs["wall_count"].softmax(-1).numpy()
        results["wall-tile-count"] = {
            "tiles": {
                tile: _prediction_from_distribution(wall[tile_index])
                for tile_index, tile in enumerate(TILES_34)
            }
        }
        dora = F.softplus(outputs["dora"]).numpy()
        results["opponent-dora-count"] = {
            "players": [
                {"seat": seat, "prediction": {"expectedValue": _finite(dora[index])}}
                for index, seat in enumerate(opponents)
            ]
        }
        score = (F.softplus(outputs["score"]) * 1000.0).numpy()
        results["opponent-score"] = {
            "players": [
                {"seat": seat, "prediction": {"expectedValue": _finite(score[index])}}
                for index, seat in enumerate(opponents)
            ]
        }
        outcome = outputs["outcome"].softmax(-1).numpy()
        draw = _finite(outcome[0])
        win = np.asarray(
            [
                sum(outcome[mask] for mask in range(1, 16) if mask & (1 << relative))
                for relative in range(4)
            ],
            dtype=np.float32,
        )
        deal_player = outputs["deal_in_player"].sigmoid().numpy()
        target = outputs["target"].softmax(-1).numpy()
        results["kyoku-outcome"] = {
            "drawProbability": draw,
            "players": [
                {
                    "seat": seat,
                    "winProbability": _finite(win[relative]),
                    "dealInProbability": _finite(deal_player[relative]),
                    "targetGivenWin": [
                        {"seat": target_seat, "probability": _finite(target[relative, target_relative])}
                        for target_relative, target_seat in enumerate(order)
                    ],
                }
                for relative, seat in enumerate(order)
            ],
        }
        delta = (outputs["kyoku_delta"] * 10_000.0).numpy()
        results["kyoku-score-delta"] = {
            "players": [
                {"seat": seat, "prediction": {"expectedValue": _finite(delta[relative])}}
                for relative, seat in enumerate(order)
            ]
        }
        joint = outputs["placement"].softmax(-1).numpy()
        placement = np.zeros((4, 4), dtype=np.float32)
        for probability, permutation in zip(joint, PERMUTATIONS):
            for relative, rank in enumerate(permutation):
                placement[relative, rank] += probability
        results["match-placement"] = {
            "players": [
                {
                    "seat": seat,
                    "prediction": _prediction_from_distribution(placement[relative], first_value=1),
                }
                for relative, seat in enumerate(order)
            ]
        }
        match_score = (outputs["match_score"] * 10_000.0).numpy()
        results["match-score"] = {
            "players": [
                {"seat": seat, "prediction": {"expectedValue": _finite(match_score[relative])}}
                for relative, seat in enumerate(order)
            ]
        }

        policy_request = next(
            (item for item in requested if item.get("id") == "action-recommendation"), None
        )
        if policy_request is not None:
            candidates = policy_request.get("parameters", {}).get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError("action-recommendation requires non-empty candidates")
            grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for candidate in candidates:
                grouped[candidate_action_index(candidate["action"])].append(candidate)
            indices = list(grouped)
            group_probabilities = outputs["policy"][indices].softmax(0).numpy()
            values: dict[str, float] = {}
            for probability, index in zip(group_probabilities, indices):
                share = _finite(probability) / len(grouped[index])
                for candidate in grouped[index]:
                    values[candidate["candidateId"]] = share
            best = max(candidates, key=lambda item: values[item["candidateId"]])
            results["action-recommendation"] = {
                "bestCandidateId": best["candidateId"],
                "candidates": [
                    {
                        "candidateId": candidate["candidateId"],
                        "metrics": {"policy": values[candidate["candidateId"]]},
                    }
                    for candidate in candidates
                ],
            }
        return results, (time.perf_counter() - started) * 1000.0
