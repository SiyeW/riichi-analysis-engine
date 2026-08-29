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

from .constants import RED_TILES, TILE37_TO_ACTION, TILES_34, relative_players, tile34_index
from .kyoku_outcome import OUTCOME_CLASSES
from .model import RiichiAnalysisModel
from .prediction_values import DORA_VALUES, SCORE_VALUES

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


def _valued_distribution(
    probabilities: np.ndarray, values: tuple[int | str, ...]
) -> list[dict[str, float | int | str]]:
    return [
        {"value": value, "probability": _finite(probability)}
        for value, probability in zip(values, probabilities, strict=True)
    ]


def _valued_expected_value(probabilities: np.ndarray, values: tuple[int, ...]) -> float:
    return _finite(np.dot(np.asarray(values, dtype=np.float64), probabilities))


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
        model_format = payload.get("format")
        formats = {
            "riichi-analysis-model-v1": 1,
            "riichi-analysis-model-v2": 2,
            "riichi-analysis-model-v3": 3,
            "riichi-analysis-model-v4": 4,
        }
        if model_format not in formats:
            raise RuntimeError("weight file has an unsupported format")
        self.format_version = formats[model_format]
        if self.format_version >= 2:
            expected_values = {
                "dora": list(DORA_VALUES),
                "score": list(SCORE_VALUES),
            }
            architecture = payload.get("architecture")
            if not isinstance(architecture, dict) or architecture.get(
                "predictionValues"
            ) != expected_values:
                raise RuntimeError("weight file uses different prediction values")
        self.model = RiichiAnalysisModel(format_version=self.format_version)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device).eval()
        self.player_state_type = _load_player_state()
        with torch.inference_mode():
            self.model(torch.zeros(1, 1012, 34, device=self.device))

    def representations(self, output_id: str, protocol_minor: int) -> list[str]:
        if output_id not in {"opponent-dora-count", "opponent-score"}:
            raise ValueError(f"unsupported representation query: {output_id}")
        if self.format_version == 1:
            return ["expected-value"]
        if protocol_minor >= 2:
            return ["distribution", "point-estimate"]
        if output_id == "opponent-score":
            return ["distribution", "expected-value"]
        return ["expected-value"]

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
        protocol_minor: int = 1,
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
        concealed_red = (
            outputs["concealed_red_count"].softmax(-1).numpy()
            if self.format_version >= 3
            else None
        )
        results["opponent-concealed-tile-count"] = {
            "players": [
                {
                    "seat": seat,
                    "tiles": {
                        tile: _prediction_from_distribution(concealed[index, tile_index])
                        for tile_index, tile in enumerate(TILES_34)
                    },
                    **(
                        {
                            "redTiles": {
                                tile: _prediction_from_distribution(
                                    concealed_red[index, tile_index]
                                )
                                for tile_index, tile in enumerate(RED_TILES)
                            }
                        }
                        if concealed_red is not None and protocol_minor >= 2
                        else {}
                    ),
                }
                for index, seat in enumerate(opponents)
            ]
        }
        wall = outputs["wall_count"].softmax(-1).numpy()
        wall_red = (
            outputs["wall_red_count"].softmax(-1).numpy()
            if self.format_version >= 3
            else None
        )
        results["wall-tile-count"] = {
            "tiles": {
                tile: _prediction_from_distribution(wall[tile_index])
                for tile_index, tile in enumerate(TILES_34)
            },
            **(
                {
                    "redTiles": {
                        tile: _prediction_from_distribution(wall_red[tile_index])
                        for tile_index, tile in enumerate(RED_TILES)
                    }
                }
                if wall_red is not None and protocol_minor >= 2
                else {}
            ),
        }
        if self.format_version == 1:
            dora_predictions = [
                {"expectedValue": _finite(value)}
                for value in F.softplus(outputs["dora"]).numpy()
            ]
            score_predictions = [
                {"expectedValue": _finite(value)}
                for value in (F.softplus(outputs["score"]) * 1000.0).numpy()
            ]
        else:
            dora_distribution = outputs["dora_distribution"].softmax(-1).numpy()
            dora_point = F.softplus(outputs["dora_point"]).numpy()
            score_distribution = outputs["score_distribution"].softmax(-1).numpy()
            score_point = (F.softplus(outputs["score_point"]) * 1000.0).numpy()
            if protocol_minor >= 2:
                dora_predictions = [
                    {
                        "distribution": _valued_distribution(probabilities, DORA_VALUES),
                        "pointEstimate": _finite(dora_point[index]),
                    }
                    for index, probabilities in enumerate(dora_distribution)
                ]
                score_predictions = [
                    {
                        "distribution": _valued_distribution(probabilities, SCORE_VALUES),
                        "pointEstimate": _finite(score_point[index]),
                    }
                    for index, probabilities in enumerate(score_distribution)
                ]
            else:
                dora_predictions = [
                    {"expectedValue": _finite(value)} for value in dora_point
                ]
                score_predictions = [
                    {
                        "distribution": _valued_distribution(probabilities, SCORE_VALUES),
                        "expectedValue": _valued_expected_value(
                            probabilities, SCORE_VALUES
                        ),
                    }
                    for probabilities in score_distribution
                ]
        results["opponent-dora-count"] = {
            "players": [
                {"seat": seat, "prediction": dora_predictions[index]}
                for index, seat in enumerate(opponents)
            ]
        }
        results["opponent-score"] = {
            "players": [
                {"seat": seat, "prediction": score_predictions[index]}
                for index, seat in enumerate(opponents)
            ]
        }
        outcome_distribution: np.ndarray | None = None
        if self.format_version == 1:
            outcome = outputs["outcome"].softmax(-1).numpy()
            draw = _finite(outcome[0])
            win = np.asarray(
                [
                    sum(outcome[mask] for mask in range(1, 16) if mask & (1 << relative))
                    for relative in range(4)
                ],
                dtype=np.float32,
            )
        else:
            any_win = outputs["outcome_any_win"].sigmoid()
            draw = _finite(1.0 - any_win)
            win = (any_win * outputs["outcome_winner"].sigmoid()).numpy()
            if self.format_version >= 4:
                outcome_distribution = outputs["outcome"].softmax(-1).numpy()
        deal_player = outputs["deal_in_player"].sigmoid().numpy()
        outcome_result: dict[str, Any] = {
            "drawProbability": draw,
            "players": [
                {
                    "seat": seat,
                    "winProbability": _finite(win[relative]),
                    "dealInProbability": _finite(deal_player[relative]),
                }
                for relative, seat in enumerate(order)
            ],
        }
        if protocol_minor >= 2 and outcome_distribution is not None:
            serialized_outcomes: list[dict[str, Any]] = []
            for outcome_class, outcome_probability in zip(
                OUTCOME_CLASSES, outcome_distribution, strict=True
            ):
                item: dict[str, Any] = {
                    "type": outcome_class.kind,
                    "probability": _finite(outcome_probability),
                }
                if outcome_class.kind == "tsumo":
                    item["winner"] = order[outcome_class.winners[0]]
                elif outcome_class.kind == "ron":
                    assert outcome_class.target is not None
                    item["winners"] = [order[winner] for winner in outcome_class.winners]
                    item["target"] = order[outcome_class.target]
                serialized_outcomes.append(item)
            outcome_result["outcomes"] = serialized_outcomes
        elif protocol_minor < 2:
            if outcome_distribution is None:
                target = outputs["target"].softmax(-1).numpy()
            else:
                target = np.zeros((4, 4), dtype=np.float32)
                for outcome_class, outcome_probability in zip(
                    OUTCOME_CLASSES, outcome_distribution, strict=True
                ):
                    if outcome_class.kind == "draw":
                        continue
                    target_relative = (
                        outcome_class.winners[0]
                        if outcome_class.kind == "tsumo"
                        else outcome_class.target
                    )
                    assert target_relative is not None
                    for winner in outcome_class.winners:
                        target[winner, target_relative] += outcome_probability
                totals = target.sum(axis=-1, keepdims=True)
                target = np.divide(
                    target,
                    totals,
                    out=np.full_like(target, 0.25),
                    where=totals > 0,
                )
            for relative, player in enumerate(outcome_result["players"]):
                player["targetGivenWin"] = [
                    {
                        "seat": target_seat,
                        "probability": _finite(target[relative, target_relative]),
                    }
                    for target_relative, target_seat in enumerate(order)
                ]
        results["kyoku-outcome"] = outcome_result
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
