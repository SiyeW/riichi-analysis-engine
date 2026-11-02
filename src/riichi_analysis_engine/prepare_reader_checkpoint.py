"""Publish a guarded V19 initialization with name-mapped inherited state.

This is not a new training contract. A completed parent's exposure and schedules
remain ancestry, not an active unread cursor. Training must reject this artifact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from .architecture import V18Architecture
from .atomic_io import publish_json, replace_file, temporary_sibling
from .corpus_extension import file_sha256
from .losses import LearnedUncertaintyBalancer
from .model import RiichiAnalysisModel, count_parameters
from .model_input import v16_model_input_metadata
from .prediction_values import DORA_VALUES, SCORE_VALUES
from .reader_growth import deepen_task_reader
from .semantic_input import semantic_input_metadata


def assert_finite_state(value) -> int:
    if isinstance(value, torch.Tensor):
        if not value.isfinite().all():
            raise ValueError("non-finite reader transfer state")
        return 1
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite reader transfer scalar")
    if isinstance(value, dict):
        return sum(assert_finite_state(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(assert_finite_state(item) for item in value)
    return 0


def inherited_optimizer(parent: dict, old_model, new_model, balancer) -> dict:
    old = parent["optimizer"]
    groups = old.get("param_groups", [])
    old_named = dict(old_model.named_parameters())
    new_named = dict(new_model.named_parameters())
    if (
        len(groups) != 2
        or len(groups[0]["params"]) != len(old_named)
        or len(groups[1]["params"]) != 1
    ):
        raise ValueError("unsupported model/balancer optimizer ownership")
    old_ids = groups[0]["params"] + groups[1]["params"]
    if len(set(old_ids)) != len(old_ids) or set(old["state"]) - set(old_ids):
        raise ValueError("optimizer contains duplicate or unowned state")
    old_by_name = dict(zip(old_named, groups[0]["params"], strict=True))
    new_by_name = {name: index for index, name in enumerate(new_named)}
    result = {"state": {}, "param_groups": copy.deepcopy(groups)}
    result["param_groups"][0]["params"] = list(range(len(new_named)))
    result["param_groups"][1]["params"] = [len(new_named)]
    if "param_names" in result["param_groups"][0]:
        result["param_groups"][0]["param_names"] = list(new_named)

    def checked_state(state, shape):
        if not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
            raise ValueError("incomplete inherited Adam state")
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in state and state[key].shape != shape:
                raise ValueError("inherited Adam tensor shape differs")
        step = state["step"]
        if isinstance(step, torch.Tensor):
            if step.numel() != 1 or step.item() < 0:
                raise ValueError("inherited Adam step is invalid")
        elif not isinstance(step, (int, float)) or step < 0:
            raise ValueError("inherited Adam step is invalid")
        return copy.deepcopy(state)

    for name, old_id in old_by_name.items():
        if name not in new_named or new_named[name].shape != old_named[name].shape:
            raise ValueError("optimizer transfer changed old parameter semantics")
        if old_id not in old["state"]:
            continue  # Absent Adam state remains absent, not zero-materialized.
        result["state"][new_by_name[name]] = checked_state(
            old["state"][old_id], old_named[name].shape
        )
    balance_id = groups[1]["params"][0]
    if balance_id in old["state"]:
        result["state"][len(new_named)] = checked_state(
            old["state"][balance_id], balancer.log_variance.shape
        )
    probe = torch.optim.AdamW(
        [{"params": new_model.parameters()}, {"params": balancer.parameters()}]
    )
    probe.load_state_dict(result)
    if any(
        parameter in probe.state
        for name, parameter in new_named.items()
        if name not in old_named
    ):
        raise ValueError("new reader inherited an old Adam state")
    return result


def prepare_reader_checkpoint(
    parent_path: Path, output: Path, *, parent_sha256: str, seed: int = 0
) -> dict:
    receipt_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError("reader initialization must not overwrite artifacts")
    if file_sha256(parent_path) != parent_sha256:
        raise ValueError("reader parent checksum differs")
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    if parent.get("format") != "riichi-analysis-model-v18":
        raise ValueError("reader transfer requires a full V18 checkpoint")
    for field in ("optimizer", "lossBalancer", "scaler", "randomState"):
        if not isinstance(parent.get(field), dict):
            raise TypeError("reader transfer requires full state: " + field)
    contracts = {
        "modelInput": v16_model_input_metadata(),
        "semanticInput": semantic_input_metadata(),
        "predictionValues": {"dora": list(DORA_VALUES), "score": list(SCORE_VALUES)},
    }
    for key, expected in contracts.items():
        if parent.get(key) != expected:
            raise ValueError("reader transfer changed " + key)
    architecture = V18Architecture.from_dict(parent.get("modelArchitecture"))
    if any(tensor.dtype != torch.float32 for tensor in parent["model"].values()):
        raise ValueError("reader transfer requires FP32 parent parameters")
    with torch.random.fork_rng(devices=[]):
        old_model = RiichiAnalysisModel(format_version=18, architecture=architecture)
        balancer = LearnedUncertaintyBalancer(tuple(parent["lossBalancer"]["terms"]))
    old_model.load_state_dict(parent["model"], strict=True)
    balancer.load_state_dict(parent["lossBalancer"]["state"], strict=True)
    assert_finite_state(parent)
    model, mapping = deepen_task_reader(old_model, seed=seed)
    optimizer = inherited_optimizer(parent, old_model, model, balancer)
    lineage = {
        "type": "task-reader-initialization-v1",
        "parent": {"path": str(parent_path.resolve()), "sha256": parent_sha256},
        "mapping": mapping,
        "newArchitectureUpdates": 0,
        "newArchitectureSamples": 0,
        "resumeAllowed": False,
        "modelParameterOrderSha256": hashlib.sha256(
            "\n".join(name for name, _ in model.named_parameters()).encode()
        ).hexdigest(),
        "optimizerPolicy": "old-name-mapped; new-parameters-no-moments-step0",
        "balancerScalerRngPolicy": "unchanged-parent-state",
    }
    inherited = {"lossBalancer", "scaler", "randomState", "lossObjective"}
    payload = {
        "format": "riichi-analysis-model-v19",
        "model": model.state_dict(),
        "modelArchitecture": model.architecture.to_dict(),
        "parameters": count_parameters(model),
        **contracts,
        **{key: copy.deepcopy(parent[key]) for key in inherited if key in parent},
        "optimizer": optimizer,
        "readerTransfer": lineage,
        "parentMetadata": {
            key: copy.deepcopy(value)
            for key, value in parent.items()
            if key not in {"model", "optimizer", *inherited}
        },
        "resumeAllowed": False,
        "step": 0,
        "samplesSeen": 0,
        "analysisSamplesSeen": 0,
        "training": {"initializationOnly": True, **lineage},
        "architecture": {
            "observationVersion": 4,
            "model": model.architecture.to_dict(),
            "parameters": count_parameters(model),
            **contracts,
        },
    }
    finite_count = assert_finite_state(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_sibling(output)
    # Failed unpublished artifacts are preserved for investigation, not erased.
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    if file_sha256(parent_path) != parent_sha256:
        raise ValueError("reader parent changed during publication")
    if output.exists():
        raise FileExistsError("reader publication target acquired another owner")
    replace_file(temporary, output)
    receipt = {
        "path": str(output.resolve()),
        "sha256": file_sha256(output),
        "bytes": output.stat().st_size,
        "parameters": count_parameters(model),
        "modelTensors": len(model.state_dict()),
        "finiteStateTensors": finite_count,
        "inheritedAdamStates": len(optimizer["state"]),
        "newReaderAdamStates": 0,
        "trainingUpdates": 0,
        "resumeAllowed": False,
        "parentSha256": parent_sha256,
        "numericalIdentityNotYetVerified": True,
    }
    publish_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_reader_checkpoint(
                args.parent,
                args.output,
                parent_sha256=args.parent_sha256,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
