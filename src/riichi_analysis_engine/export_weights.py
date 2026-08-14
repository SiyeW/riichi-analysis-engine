from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .model import RiichiAnalysisModel, count_parameters


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export inference-only model weights.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--training-data", default="2025-1of20")
    parser.add_argument("--validation-data", default="2026-md5-holdout")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "riichi-analysis-model-v1":
        raise RuntimeError("checkpoint has an unsupported format")
    model = RiichiAnalysisModel()
    model.load_state_dict(checkpoint["model"], strict=True)
    payload = {
        "format": "riichi-analysis-model-v1",
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
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
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
