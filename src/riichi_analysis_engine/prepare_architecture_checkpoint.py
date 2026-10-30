"""Turn a selected model initializer into an explicit, guarded warm-start state.

Only changed-shape model Adam moments are reset. The unchanged uncertainty
parameters and their Adam state, AMP scaler, RNG, data cursor and LR schedule
are inherited. Legacy exposure and new-shape updates remain separate.
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
from .architecture_transition import (
    TRANSITION_TYPE,
    base_cursor,
    restore_architecture_transition,
    transition_cursor,
    transition_fields,
)
from .atomic_io import publish_json, replace_file, temporary_sibling
from .corpus_extension import file_sha256
from .losses import LearnedUncertaintyBalancer
from .model import RiichiAnalysisModel, count_parameters
from .training_objective import saved_objective
from .training_phase import phase_totals


def _finite(value):
    if isinstance(value, torch.Tensor):
        if not value.isfinite().all():
            raise ValueError("non-finite warm-start state")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite scalar warm-start state")
    elif isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _finite(item)


def prepare_checkpoint(
    initializer_path: Path,
    parent_path: Path,
    output: Path,
    *,
    initializer_sha256: str,
    parent_sha256: str,
    source_revision: str,
) -> dict:
    proof_path = output.with_suffix(output.suffix + ".transition.json")
    receipt_path = output.with_suffix(output.suffix + ".json")
    if any(path.exists() for path in (output, proof_path, receipt_path)):
        raise FileExistsError("warm-start artifacts must not be overwritten")
    for path, expected in (
        (initializer_path, initializer_sha256),
        (parent_path, parent_sha256),
    ):
        if file_sha256(path) != expected:
            raise ValueError("selected warm-start artifact checksum changed")
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    initial = torch.load(initializer_path, map_location="cpu", weights_only=True)
    if (
        parent.get("format") != "riichi-analysis-model-v18"
        or initial.get("format") != parent["format"]
    ):
        raise ValueError("architecture warm-start requires V18 artifacts")
    cursor = base_cursor(parent)
    if (
        cursor.get("complete") is not False
        or parent.get("resumeAllowed", True) is not True
        or cursor.get("nextSample") != parent.get("samplesSeen")
        or cursor.get("batchesConsumed") != parent.get("step")
    ):
        raise ValueError("warm-start requires a full, unfinished exact parent cursor")
    if parent.get("architectureTransition") is not None:
        raise ValueError("stacking shape transitions requires a separate contract")
    metadata = initial.get("training", {})
    if (
        metadata.get("initializationOnly") is not True
        or initial.get("resumeAllowed") is not False
        or metadata.get("parentSha256") != parent_sha256
        or metadata.get("newArchitectureUpdates") != 0
    ):
        raise ValueError("initializer is not the selected zero-update parent transfer")
    for name in ("step", "samplesSeen", "analysisSamplesSeen"):
        if metadata.get("parentCounters", {}).get(name) != parent.get(name):
            raise ValueError("initializer belongs to a different parent cursor")
    for name, value in metadata.get("parentCounters", {}).items():
        if parent.get(name) != value:
            raise ValueError("initializer parent history changed: " + name)
    for name in ("modelInput", "semanticInput", "predictionValues"):
        if initial["architecture"].get(name) != parent.get(name):
            raise ValueError("architecture warm-start changed " + name)
    objective = saved_objective(parent)
    target = V18Architecture.from_dict(initial["architecture"]["model"])
    if target.to_dict() == parent["modelArchitecture"]:
        raise ValueError("this entrypoint requires a changed architecture")
    with torch.random.fork_rng(devices=[]):
        model = RiichiAnalysisModel(format_version=18, architecture=target)
        balancer = LearnedUncertaintyBalancer(tuple(parent["lossBalancer"]["terms"]))
    model.load_state_dict(initial["model"], strict=True)
    balancer.load_state_dict(parent["lossBalancer"]["state"], strict=True)
    _finite(parent)
    _finite(initial["model"])
    names = [name for name, _ in model.named_parameters()]
    if len(names) != len(model.state_dict()):
        raise ValueError("unexpected persistent buffers in V18 parameter mapping")
    old_optimizer = parent["optimizer"]
    groups = old_optimizer.get("param_groups", [])
    if (
        len(groups) != 2
        or len(groups[1]["params"]) != 1
        or len(groups[0]["params"]) != len(parent["model"])
    ):
        raise ValueError("unsupported model/balancer optimizer ownership")
    old_balance_id = groups[1]["params"][0]
    new_balance_id = len(names)
    optimizer = {"state": {}, "param_groups": copy.deepcopy(groups)}
    optimizer["param_groups"][0]["params"] = list(range(len(names)))
    optimizer["param_groups"][1]["params"] = [new_balance_id]
    if old_balance_id in old_optimizer["state"]:
        balance_adam = copy.deepcopy(old_optimizer["state"][old_balance_id])
        for name in ("exp_avg", "exp_avg_sq"):
            if balance_adam[name].shape != balancer.log_variance.shape:
                raise ValueError("balancer Adam tensors do not match unchanged tasks")
        optimizer["state"][new_balance_id] = balance_adam
    # Exercise the actual optimizer state loader; construction/verification stays CPU.
    probe = torch.optim.AdamW(
        [
            {"params": model.parameters()},
            {"params": balancer.parameters()},
        ]
    )
    probe.load_state_dict(optimizer)
    if any(parameter in probe.state for parameter in model.parameters()):
        raise ValueError("new model unexpectedly inherited old Adam moments")
    start = {
        name: int(parent[key])
        for name, key in (
            ("samples", "samplesSeen"),
            ("steps", "step"),
            ("analysis", "analysisSamplesSeen"),
        )
    }
    proof = {
        "type": TRANSITION_TYPE,
        "initializer": {
            "path": str(initializer_path.resolve()),
            "sha256": initializer_sha256,
        },
        "parent": {"path": str(parent_path.resolve()), "sha256": parent_sha256},
        "start": start,
        "parentCumulativeCounters": phase_totals(
            parent.get("trainingPhase"),
            start["samples"],
            start["steps"],
            start["analysis"],
        ),
        "sourceArchitecture": parent["modelArchitecture"],
        "targetArchitecture": target.to_dict(),
        "targetParameters": count_parameters(model),
        "modelParameterOrderSha256": hashlib.sha256(
            "\n".join(names).encode()
        ).hexdigest(),
        "manifests": {
            split: parent["datasets"][split]["manifestSha256"]
            for split in ("train", "validation")
        },
        "lossObjective": objective,
        "learningRateSchedule": parent["learningRateSchedule"],
        "optimizerPolicy": {
            "model": "fresh-zero-moments",
            "balancer": "bitwise-inherited-unchanged-parameter",
        },
        "scalerPolicy": "bitwise-parent",
        "randomStatePolicy": "bitwise-parent",
        "sourceRevisionAtTransition": source_revision,
        "notOldModelOptimizerTrajectory": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    publish_json(proof_path, proof)
    transition = {
        **proof,
        "proof": {"path": str(proof_path.resolve()), "sha256": file_sha256(proof_path)},
    }
    payload = {
        key: copy.deepcopy(value)
        for key, value in parent.items()
        if key not in {"model", "optimizer", "parameters", "environment", "validation"}
    }
    payload.update(
        model=model.state_dict(),
        modelArchitecture=target.to_dict(),
        optimizer=optimizer,
        parameters=count_parameters(model),
        validation=None,
        lossObjective=objective,
        resumeAllowed=True,
        environment={
            "sourceRevision": source_revision,
            "initializationOnly": True,
            "parentSourceRevision": parent.get("environment", {}).get("sourceRevision"),
        },
        trainingCursor=transition_cursor(cursor, transition),
        **transition_fields(
            transition, start["samples"], start["steps"], start["analysis"]
        ),
    )
    restore_architecture_transition(payload)
    temporary = temporary_sibling(output)
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    if (
        file_sha256(parent_path) != parent_sha256
        or file_sha256(initializer_path) != initializer_sha256
    ):
        raise ValueError("input changed during warm-start publication")
    replace_file(temporary, output, attempts=20, retry_delay=0.025)
    receipt = {
        "verified": True,
        "path": str(output.resolve()),
        "sha256": file_sha256(output),
        "bytes": output.stat().st_size,
        "proofSha256": transition["proof"]["sha256"],
        "parentSha256": parent_sha256,
        "initializerSha256": initializer_sha256,
        "parameters": count_parameters(model),
        "modelTensors": len(names),
        "newModelAdamStates": 0,
        "balancerAdamStates": len(optimizer["state"]),
        "balancerScalerRngInherited": True,
        "newShapeUpdates": 0,
        "sourceSamplesSeen": start["samples"],
        "sourceStep": start["steps"],
        "formalTrainingStarted": False,
    }
    publish_json(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("initializer", type=Path)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--initializer-sha256", required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_checkpoint(
                args.initializer,
                args.parent,
                args.output,
                initializer_sha256=args.initializer_sha256,
                parent_sha256=args.parent_sha256,
                source_revision=args.source_revision,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
