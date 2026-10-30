"""Publish an isolated, inference-loadable architecture initialization artifact."""

import argparse
import json
import os
from pathlib import Path

import torch

from .architecture import V18Architecture
from .architecture_growth import grow_v18
from .atomic_io import publish_json, replace_file, temporary_sibling
from .export_weights import sha256
from .model import RiichiAnalysisModel, count_parameters
from .model_input import v16_model_input_metadata
from .prediction_values import DORA_VALUES, SCORE_VALUES
from .semantic_input import semantic_input_metadata


def publish_initialization(
    parent_path: Path,
    output: Path,
    architecture: V18Architecture,
    *,
    expected_parent_sha256: str,
    seed: int = 0,
    symmetry_scale: float = 0.001,
) -> dict:
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError("initialization outputs must not overwrite an artifact")
    parent_hash = sha256(parent_path)
    if parent_hash != expected_parent_sha256:
        raise ValueError("parent checksum does not match the selected identity")
    checkpoint = torch.load(parent_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "riichi-analysis-model-v18":
        raise ValueError(
            "initialization requires a V18 checkpoint or full inference file"
        )
    inference = checkpoint.get("architecture", {})
    source_architecture = V18Architecture.from_dict(
        checkpoint.get("modelArchitecture", inference.get("model"))
    )
    contracts = {
        "modelInput": v16_model_input_metadata(),
        "semanticInput": semantic_input_metadata(),
        "predictionValues": {"dora": list(DORA_VALUES), "score": list(SCORE_VALUES)},
    }
    for key, expected in contracts.items():
        if checkpoint.get(key, inference.get(key)) != expected:
            raise ValueError(f"parent uses a different {key} contract")
    with torch.random.fork_rng(devices=[]):
        parent = RiichiAnalysisModel(
            format_version=18, architecture=source_architecture
        )
    parent.load_state_dict(checkpoint["model"], strict=True)
    model, mapping = grow_v18(
        parent, architecture, seed=seed, symmetry_scale=symmetry_scale
    )
    parent_counters = {
        key: checkpoint[key]
        for key in (
            "step",
            "samplesSeen",
            "analysisSamplesSeen",
            "trainingPhase",
            "trainingCursor",
            "lossObjective",
            "objectiveTransition",
        )
        if key in checkpoint
    }
    lineage = {
        "type": "architecture-initialization-v1",
        "parentSha256": parent_hash,
        "parentCounters": parent_counters,
        "parentTrainingMetadata": checkpoint.get("training"),
        "newArchitectureUpdates": 0,
        "newArchitectureSamples": 0,
        "optimizerStateMapped": False,
        "resumeAllowed": False,
        "mapping": mapping,
    }
    payload = {
        "format": checkpoint["format"],
        "model": model.state_dict(),
        "resumeAllowed": False,
        "architecture": {
            "observationVersion": 4,
            "model": architecture.to_dict(),
            "parameters": count_parameters(model),
            **contracts,
        },
        "training": {"initializationOnly": True, **lineage},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_sibling(output)
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if sha256(parent_path) != parent_hash:
            raise ValueError("parent changed during migration")
        # No strict=False fallbacks; audit the exact file about to be published.
        saved = torch.load(temporary, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"], strict=True)
        if any(not value.isfinite().all() for value in saved["model"].values()):
            raise ValueError("initialization contains non-finite tensors")
        replace_file(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {
        "path": str(output.resolve()),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "finiteModelTensors": len(payload["model"]),
        **lineage,
    }
    # The receipt is the commit record. An artifact without it is not accepted.
    publish_json(report_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--shared-ff", type=int, required=True)
    parser.add_argument("--private-ff", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--symmetry-scale", type=float, default=0.001)
    args = parser.parse_args()
    receipt = publish_initialization(
        args.parent,
        args.output,
        V18Architecture(
            width=args.width,
            blocks=args.blocks,
            attention_heads=args.heads,
            feed_forward_width=args.shared_ff,
            decoder_width=args.private_ff,
        ),
        expected_parent_sha256=args.parent_sha256,
        seed=args.seed,
        symmetry_scale=args.symmetry_scale,
    )
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "path",
                    "bytes",
                    "sha256",
                    "finiteModelTensors",
                    "resumeAllowed",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
