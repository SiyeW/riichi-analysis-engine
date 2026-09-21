import sys

import pytest
import torch

from riichi_analysis_engine.architecture import (
    ModelArchitecture,
    SemanticModelArchitecture,
    StructuredModelArchitecture,
)
from riichi_analysis_engine.export_weights import (
    main,
    training_source_revision,
)
from riichi_analysis_engine.model import RiichiAnalysisModel
from riichi_analysis_engine.model_input import model_input_metadata
from riichi_analysis_engine.prediction_values import DORA_VALUES, SCORE_VALUES
from riichi_analysis_engine.semantic_input import semantic_input_metadata


def test_training_revision_uses_checkpoint_environment() -> None:
    checkpoint = {
        "environment": {"sourceRevision": "abc123", "sourceDirty": False},
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
        (
            10,
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
            11,
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
            12,
            SemanticModelArchitecture(
                backbone="cnn",
                width=16,
                stem_width=24,
                event_width=12,
                backbone_blocks=1,
                event_blocks=1,
                decoder_width=20,
                attention_heads=4,
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
        "analysisSamplesSeen": 224,
        "trainingCursor": {
            "type": "single-pass-v1",
            "nextSample": 320,
            "batchesConsumed": 10,
            "batchSize": 32,
            "complete": False,
        },
    }
    if format_version in {9, 10, 11, 12}:
        checkpoint["modelInput"] = model_input_metadata()
    if format_version == 12:
        checkpoint["semanticInput"] = semantic_input_metadata()
    torch.save(checkpoint, source)
    monkeypatch.setattr(sys, "argv", ["export_weights", str(source), str(destination)])

    main()

    exported = torch.load(destination, map_location="cpu", weights_only=True)
    assert exported["format"] == f"riichi-analysis-model-v{format_version}"
    assert exported["architecture"]["model"] == architecture.to_dict()
    if format_version in {9, 10, 11, 12}:
        assert exported["architecture"]["modelInput"] == model_input_metadata()
    if format_version == 12:
        assert exported["architecture"]["semanticInput"] == semantic_input_metadata()
    assert exported["training"]["step"] == 10
    assert exported["training"]["samplesSeen"] == 320
    assert exported["training"]["analysisSamplesSeen"] == 224
    assert exported["training"]["pass"] == {
        "type": "single-pass-v1",
        "nextSample": 320,
        "complete": False,
    }
    assert "datasets" not in exported["training"]
    assert "trainingData" not in exported["training"]
    assert "validationData" not in exported["training"]
    assert "epoch" not in exported["training"]
