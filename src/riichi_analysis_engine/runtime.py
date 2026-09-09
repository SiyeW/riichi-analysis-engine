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

from .architecture import ModelArchitecture
from .constants import (
    MORTAL_OBS_CHANNELS,
    OBS_CHANNELS,
    RED_TILES,
    TILE37_TO_ACTION,
    TILES_34,
    relative_players,
    tile34_index,
)
from .kyoku_outcome import OUTCOME_CLASSES, outcome_marginals
from .model import RiichiAnalysisModel
from .observations import add_all_player_ranks
from .prediction_values import DORA_VALUES, SCORE_VALUES
from .rule_certainties import (
    PublicRuleState,
    apply_opponent_rule_certainties,
    constrain_distribution,
)
from .score_state import PublicScoreState

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


def _kan_selection_tile(action: dict[str, Any]) -> int | None:
    """Return Mortal's conditional kan-selection tile for a final action."""

    kind = action.get("type")
    if kind == "ankan":
        consumed = action.get("consumed")
        if not isinstance(consumed, list) or len(consumed) != 4:
            raise ValueError("ankan candidates require four consumed tiles")
        return tile34_index(consumed[0])
    if kind == "kakan":
        pai = action.get("pai")
        if not isinstance(pai, str):
            raise ValueError("kakan candidates require pai")
        return tile34_index(pai)
    return None


def _selected_softmax(logits: np.ndarray, indices: list[int]) -> dict[int, float]:
    if not indices:
        raise ValueError("cannot normalize an empty candidate set")
    values = np.asarray(logits, dtype=np.float64)[indices]
    shifted = values - values.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    return {index: _finite(probability) for index, probability in zip(indices, probabilities)}


def complete_candidate_policy(
    primary_logits: np.ndarray,
    candidates: list[dict[str, Any]],
    *,
    kan_selection_logits: np.ndarray | None = None,
    kan_selection_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Score final protocol candidates from Mortal's internal action factors.

    Mortal represents ankan/kakan as ``P(kan) * P(tile | kan)``.  The protocol
    represents a fully executable action, so this boundary is where those two
    internal steps are composed.  Studio never has to request the second step.
    """

    if not candidates:
        raise ValueError("action-recommendation requires non-empty candidates")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        candidate_id = candidate.get("candidateId")
        action = candidate.get("action")
        if not isinstance(candidate_id, str) or not isinstance(action, dict):
            raise ValueError("action candidates require candidateId and action")
        grouped[candidate_action_index(action)].append(candidate)

    primary = _selected_softmax(primary_logits, sorted(grouped))
    values: dict[str, float] = {}
    for action_index, grouped_candidates in grouped.items():
        if action_index != 42:
            share = primary[action_index] / len(grouped_candidates)
            for candidate in grouped_candidates:
                values[candidate["candidateId"]] = _finite(share)

    kan_candidates = grouped.get(42, [])
    if not kan_candidates:
        return values
    selection_candidates = [
        candidate
        for candidate in kan_candidates
        if _kan_selection_tile(candidate["action"]) is not None
    ]
    if not selection_candidates:
        share = primary[42] / len(kan_candidates)
        for candidate in kan_candidates:
            values[candidate["candidateId"]] = _finite(share)
        return values
    if len(selection_candidates) != len(kan_candidates):
        raise ValueError("cannot combine daiminkan with ankan/kakan in one action state")
    if kan_selection_logits is None or kan_selection_mask is None:
        raise ValueError("ankan/kakan candidates require conditional kan policy logits")

    by_tile: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for candidate in selection_candidates:
        tile = _kan_selection_tile(candidate["action"])
        assert tile is not None
        by_tile[tile].append(candidate)
    selection_tiles = sorted(by_tile)
    mask = np.asarray(kan_selection_mask, dtype=bool)
    unsupported = [tile for tile in selection_tiles if tile >= len(mask) or not mask[tile]]
    if unsupported:
        raise ValueError(f"kan candidates are unavailable in the engine state: {unsupported}")
    conditional = _selected_softmax(kan_selection_logits, selection_tiles)
    for tile, tile_candidates in by_tile.items():
        share = primary[42] * conditional[tile] / len(tile_candidates)
        for candidate in tile_candidates:
            values[candidate["candidateId"]] = _finite(share)
    return values


def best_candidate(candidates: list[dict[str, Any]], values: dict[str, float]) -> dict[str, Any]:
    """Choose deterministically without letting host array order break ties."""

    return min(
        candidates,
        key=lambda candidate: (
            -values[candidate["candidateId"]],
            json.dumps(candidate["action"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            candidate["candidateId"],
        ),
    )


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
            "riichi-analysis-model-v5": 5,
            "riichi-analysis-model-v6": 6,
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
        if self.format_version == 6:
            architecture = payload.get("architecture")
            if not isinstance(architecture, dict):
                raise RuntimeError("v6 weight file has no architecture metadata")
            try:
                model_architecture = ModelArchitecture.from_dict(architecture.get("model"))
            except ValueError as error:
                raise RuntimeError(f"v6 weight file has invalid architecture metadata: {error}") from error
            self.model = RiichiAnalysisModel(architecture=model_architecture)
        else:
            self.model = RiichiAnalysisModel(format_version=self.format_version)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device).eval()
        self.player_state_type = _load_player_state()
        observation_channels = OBS_CHANNELS if self.format_version == 6 else MORTAL_OBS_CHANNELS
        with torch.inference_mode():
            self.model(torch.zeros(1, observation_channels, 34, device=self.device))

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

    def _state(self, events: list[dict[str, Any]], controlled_seat: int) -> Any:
        state = self.player_state_type(controlled_seat)
        for event in events:
            state.update(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
        return state

    def _encode_observation(
        self, state: Any, score_state: PublicScoreState, *, at_kan_select: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        observation, mask = state.encode_obs(4, at_kan_select)
        if self.format_version == 6:
            observation = add_all_player_ranks(observation, score_state.relative(state.player_id))
        return np.asarray(observation, dtype=np.float32), np.asarray(mask, dtype=bool)

    def _observation(self, events: list[dict[str, Any]], controlled_seat: int) -> np.ndarray:
        state = self._state(events, controlled_seat)
        score_state = PublicScoreState()
        for event in events:
            score_state.process(event)
        observation, _mask = self._encode_observation(
            state, score_state, at_kan_select=False
        )
        return observation

    def predict(
        self,
        events: list[dict[str, Any]],
        controlled_seat: int,
        requested: list[dict[str, Any]],
        protocol_minor: int = 1,
    ) -> tuple[dict[str, dict[str, Any]], float]:
        started = time.perf_counter()
        state = self._state(events, controlled_seat)
        score_state = PublicScoreState()
        for event in events:
            score_state.process(event)
        observation, _primary_mask = self._encode_observation(
            state, score_state, at_kan_select=False
        )
        tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            raw = self.model(tensor)
        outputs = {name: value[0].float().cpu() for name, value in raw.items()}
        rule_state = PublicRuleState.from_events(events)
        order = [(controlled_seat + offset) % 4 for offset in range(4)]
        opponents = relative_players(controlled_seat)
        results: dict[str, dict[str, Any]] = {}

        shanten = outputs["shanten"].softmax(-1).numpy()
        furiten = outputs["furiten_no_yaku"].sigmoid().numpy()
        waits = outputs["deal_in_tile"].sigmoid().numpy()
        for index, seat in enumerate(opponents):
            shanten[index], waits[index] = apply_opponent_rule_certainties(
                shanten[index],
                waits[index],
                is_riichi=rule_state.riichi[seat],
                forbidden_tiles=rule_state.forbidden_tiles[seat],
            )
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
        for index, seat in enumerate(opponents):
            for tile_index, tile in enumerate(TILES_34):
                concealed[index, tile_index] = constrain_distribution(
                    concealed[index, tile_index],
                    *rule_state.concealed_range(seat, tile),
                )
            if concealed_red is not None:
                for tile_index, tile in enumerate(RED_TILES):
                    concealed_red[index, tile_index] = constrain_distribution(
                        concealed_red[index, tile_index],
                        *rule_state.concealed_red_range(seat, tile),
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
        for tile_index, tile in enumerate(TILES_34):
            wall[tile_index] = constrain_distribution(
                wall[tile_index],
                *rule_state.wall_range(tile),
            )
        if wall_red is not None:
            for tile_index, tile in enumerate(RED_TILES):
                wall_red[tile_index] = constrain_distribution(
                    wall_red[tile_index],
                    *rule_state.wall_red_range(tile),
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
            deal_player = outputs["deal_in_player"].sigmoid().numpy()
        elif self.format_version >= 5:
            outcome_distribution = outputs["outcome"].softmax(-1).numpy()
            draw, win, deal_player = outcome_marginals(outcome_distribution)
            draw = _finite(draw)
        else:
            any_win = outputs["outcome_any_win"].sigmoid()
            draw = _finite(1.0 - any_win)
            win = (any_win * outputs["outcome_winner"].sigmoid()).numpy()
            deal_player = outputs["deal_in_player"].sigmoid().numpy()
            if self.format_version >= 4:
                outcome_distribution = outputs["outcome"].softmax(-1).numpy()
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
            needs_kan_selection = any(
                isinstance(candidate, dict)
                and isinstance(candidate.get("action"), dict)
                and candidate["action"].get("type") in {"ankan", "kakan"}
                for candidate in candidates
            )
            kan_selection_logits: np.ndarray | None = None
            kan_selection_mask: np.ndarray | None = None
            if needs_kan_selection:
                selection_observation, kan_selection_mask = self._encode_observation(
                    state, score_state, at_kan_select=True
                )
                selection_tensor = torch.from_numpy(selection_observation).unsqueeze(0).to(self.device)
                with torch.inference_mode():
                    kan_selection_logits = self.model(selection_tensor)["policy"][0].float().cpu().numpy()
            values = complete_candidate_policy(
                outputs["policy"].numpy(),
                candidates,
                kan_selection_logits=kan_selection_logits,
                kan_selection_mask=kan_selection_mask,
            )
            best = best_candidate(candidates, values)
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
