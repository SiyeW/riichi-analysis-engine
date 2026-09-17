import sys

import pytest
import torch

from riichi_analysis_engine.architecture import (
    ModelArchitecture,
    StructuredModelArchitecture,
)
from riichi_analysis_engine.export_weights import (
    main,
    public_dataset_metadata,
    training_source_revision,
)
from riichi_analysis_engine.model import RiichiAnalysisModel
from riichi_analysis_engine.model_input import model_input_metadata
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


@pytest.mark.parametrize(
    ("format_version", "architecture"),
    [
        (
            6,
            ModelArchitecture(
                analysis_channels=16,
                analysis_blocks=1,
                analysis_latent_width=32,
                state_width=24,
                future_width=20,
                policy_context_channels=16,
                policy_context_blocks=1,
                policy_context_width=16,
                policy_width=24,
            ),
        ),
        (
            8,
            StructuredModelArchitecture(
                shared_channels=8,
                shared_blocks=1,
                family_latent_width=16,
                opponent_latent_width=20,
                policy_latent_width=24,
                opponent_blocks=1,
                hidden_blocks=1,
                value_blocks=1,
                kyoku_blocks=1,
                match_blocks=1,
                policy_blocks=1,
                task_width=12,
                tile_width=6,
                policy_context_channels=4,
                policy_context_blocks=1,
                policy_context_width=8,
                policy_width=16,
            ),
        ),
        (
            9,
            StructuredModelArchitecture(
                shared_channels=8,
                shared_blocks=1,
                family_latent_width=16,
                opponent_latent_width=20,
                policy_latent_width=24,
                opponent_blocks=1,
                hidden_blocks=1,
                value_blocks=1,
                kyoku_blocks=1,
                match_blocks=1,
                policy_blocks=1,
                task_width=12,
                tile_width=6,
                policy_context_channels=4,
                policy_context_blocks=1,
                policy_context_width=8,
                policy_width=16,
            ),
        ),
    ],
)
def test_export_preserves_model_architecture(
    tmp_path, monkeypatch, format_version, architecture
) -> None:
    source = tmp_path / "checkpoint.pt"
    destination = tmp_path / "weights.pt"
    checkpoint = {
            "format": f"riichi-analysis-model-v{format_version}",
            "model": RiichiAnalysisModel(
                format_version=format_version, architecture=architecture
            ).state_dict(),
            "modelArchitecture": architecture.to_dict(),
            "predictionValues": {"dora": list(DORA_VALUES), "score": list(SCORE_VALUES)},
            "environment": {},
            "step": 10,
            "samplesSeen": 320,
            "trainingCursor": {
                "type": "single-pass-v1",
                "nextSample": 320,
                "batchesConsumed": 10,
                "batchSize": 32,
                "complete": False,
            },
        }
    if format_version == 9:
        checkpoint["modelInput"] = model_input_metadata()
    torch.save(checkpoint, source)
    monkeypatch.setattr(sys, "argv", ["export_weights", str(source), str(destination)])

    main()

    exported = torch.load(destination, map_location="cpu", weights_only=True)
    assert exported["format"] == f"riichi-analysis-model-v{format_version}"
    assert exported["architecture"]["model"] == architecture.to_dict()
    if format_version == 9:
        assert exported["architecture"]["modelInput"] == model_input_metadata()
    assert exported["training"]["step"] == 10
    assert exported["training"]["samplesSeen"] == 320
    assert exported["training"]["pass"] == {
        "type": "single-pass-v1",
        "nextSample": 320,
        "complete": False,
    }
    assert "epoch" not in exported["training"]
