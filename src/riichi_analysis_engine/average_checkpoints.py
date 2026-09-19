from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

CONTRACT_FIELDS = (
    "format",
    "modelArchitecture",
    "modelInput",
    "semanticInput",
    "predictionValues",
    "datasets",
    "learningRateSchedule",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract(checkpoint: Mapping[str, object]) -> dict[str, object]:
    return {field: checkpoint.get(field) for field in CONTRACT_FIELDS}


def average_model_states(
    states: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Average floating model tensors while requiring exact structural buffers."""

    if len(states) < 2:
        raise ValueError("checkpoint averaging requires at least two model states")
    keys = tuple(states[0])
    sums: dict[str, torch.Tensor] = {}
    fixed: dict[str, torch.Tensor] = {}
    for index, state in enumerate(states):
        if tuple(state) != keys:
            raise RuntimeError("checkpoints contain different model-state keys")
        for name in keys:
            tensor = state[name].detach().cpu()
            reference = states[0][name]
            if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
                raise RuntimeError(
                    f"checkpoint tensor {name!r} has a different contract"
                )
            if torch.is_floating_point(tensor):
                value = tensor.to(torch.float64)
                sums[name] = value.clone() if index == 0 else sums[name] + value
            elif index == 0:
                fixed[name] = tensor.clone()
            elif not torch.equal(tensor, fixed[name]):
                raise RuntimeError(
                    f"non-floating checkpoint tensor {name!r} is not identical"
                )
    count = float(len(states))
    return {
        name: (
            (sums[name] / count).to(states[0][name].dtype)
            if name in sums
            else fixed[name]
        )
        for name in keys
    }


def create_average_checkpoint(
    sources: Sequence[Path], output: Path
) -> dict[str, object]:
    resolved = [path.resolve() for path in sources]
    if len(resolved) < 2:
        raise ValueError("checkpoint averaging requires at least two checkpoints")
    if len(set(resolved)) != len(resolved):
        raise ValueError("checkpoint paths must be unique")
    if output.resolve() in resolved:
        raise ValueError("the averaged checkpoint cannot overwrite a source")
    if output.exists():
        raise FileExistsError(output)

    checkpoints: list[dict[str, object]] = []
    identities: list[dict[str, object]] = []
    expected_contract: dict[str, object] | None = None
    for path in resolved:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("model"), Mapping
        ):
            raise TypeError(f"{path} is not a training checkpoint")
        if checkpoint.get("resumeAllowed", True) is not True:
            raise RuntimeError(f"{path} is already a validation-only candidate")
        contract = _contract(checkpoint)
        if expected_contract is None:
            expected_contract = contract
        elif contract != expected_contract:
            raise RuntimeError("checkpoints use different model or training contracts")
        checkpoints.append(checkpoint)
        identities.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "step": int(checkpoint.get("step", 0)),
                "samplesSeen": int(checkpoint.get("samplesSeen", 0)),
            }
        )

    base = max(
        checkpoints,
        key=lambda checkpoint: (
            int(checkpoint.get("samplesSeen", 0)),
            int(checkpoint.get("step", 0)),
        ),
    )
    averaged = dict(base)
    averaged["model"] = average_model_states(
        [checkpoint["model"] for checkpoint in checkpoints]
    )
    averaged["validation"] = None
    averaged["resumeAllowed"] = False
    averaged["averagedFrom"] = identities

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(averaged, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(output.resolve()),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "sources": identities,
        "resumeAllowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a validation-only arithmetic average of model checkpoints."
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    args = parser.parse_args()
    report = create_average_checkpoint(args.checkpoints, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
