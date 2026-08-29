from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch

from .model import RiichiAnalysisModel, count_parameters
from .prediction_values import DORA_VALUES, SCORE_VALUES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def public_dataset_metadata(checkpoint: dict[str, object]) -> object:
    datasets = checkpoint.get("datasets")
    if not isinstance(datasets, dict):
        return None
    return {
        split: {key: value for key, value in metadata.items() if key != "path"}
        if isinstance(metadata, dict)
        else metadata
        for split, metadata in datasets.items()
    }


def training_source_revision(checkpoint: dict[str, object]) -> str | None:
    environment = checkpoint.get("environment")
    if isinstance(environment, dict):
        revision = environment.get("sourceRevision")
        if isinstance(revision, str):
            return revision
    return source_revision()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export inference-only model weights.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--training-data", default="2025-1of20")
    parser.add_argument("--validation-data", default="2026-md5-holdout")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_format = checkpoint.get("format")
    formats = {
        "riichi-analysis-model-v1": 1,
        "riichi-analysis-model-v2": 2,
        "riichi-analysis-model-v3": 3,
        "riichi-analysis-model-v4": 4,
        "riichi-analysis-model-v5": 5,
    }
    if model_format not in formats:
        raise RuntimeError("checkpoint has an unsupported format")
    format_version = formats[model_format]
    model = RiichiAnalysisModel(format_version=format_version)
    model.load_state_dict(checkpoint["model"], strict=True)
    payload = {
        "format": model_format,
        "model": model.state_dict(),
        "architecture": {
            "observationVersion": 4,
            "channels": 256,
            "residualBlocks": 54,
            "stateWidth": 1024,
            "futureWidth": 768,
            "parameters": count_parameters(model),
        },
        "training": {
            "epoch": int(checkpoint.get("epoch", 0)),
            "step": int(checkpoint.get("step", 0)),
            "trainingData": args.training_data,
            "validationData": args.validation_data,
            "datasets": public_dataset_metadata(checkpoint),
            "environment": checkpoint.get("environment"),
            "validation": checkpoint.get("validation"),
            "sourceRevision": training_source_revision(checkpoint),
        },
    }
    if format_version >= 2:
        prediction_values = {
            "dora": list(DORA_VALUES),
            "score": list(SCORE_VALUES),
        }
        if checkpoint.get("predictionValues") != prediction_values:
            raise RuntimeError("checkpoint uses different prediction values")
        payload["architecture"]["predictionValues"] = prediction_values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "path": str(args.output.resolve()),
                "bytes": args.output.stat().st_size,
                "sha256": sha256(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
