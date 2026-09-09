import sys

import torch

from riichi_analysis_engine.architecture import ModelArchitecture
from riichi_analysis_engine.export_weights import (
    main,
    public_dataset_metadata,
    training_source_revision,
)
from riichi_analysis_engine.model import RiichiAnalysisModel
from riichi_analysis_engine.prediction_values import DORA_VALUES, SCORE_VALUES


def test_exported_provenance_omits_local_paths() -> None:
    checkpoint = {
        "datasets": {
            "train": {"path": "private-training-corpus", "games": 10},
            "validation": {"path": "private-validation-corpus", "games": 2},
        },
        "environment": {"sourceRevision": "abc123", "sourceDirty": False},
    }
    assert public_dataset_metadata(checkpoint) == {
        "train": {"games": 10},
        "validation": {"games": 2},
    }
    assert training_source_revision(checkpoint) == "abc123"


def test_v6_export_preserves_model_architecture(tmp_path, monkeypatch) -> None:
    architecture = ModelArchitecture(
        analysis_channels=16,
        analysis_blocks=1,
        analysis_latent_width=32,
        state_width=24,
        future_width=20,
        policy_context_channels=16,
        policy_context_blocks=1,
        policy_context_width=16,
        policy_width=24,
    )
    source = tmp_path / "checkpoint.pt"
    destination = tmp_path / "weights.pt"
    torch.save(
        {
            "format": "riichi-analysis-model-v6",
            "model": RiichiAnalysisModel(architecture=architecture).state_dict(),
            "modelArchitecture": architecture.to_dict(),
            "predictionValues": {"dora": list(DORA_VALUES), "score": list(SCORE_VALUES)},
            "environment": {},
        },
        source,
    )
    monkeypatch.setattr(sys, "argv", ["export_weights", str(source), str(destination)])

    main()

    exported = torch.load(destination, map_location="cpu", weights_only=True)
    assert exported["format"] == "riichi-analysis-model-v6"
    assert exported["architecture"]["model"] == architecture.to_dict()
