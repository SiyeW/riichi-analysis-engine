from pathlib import Path

import pytest
import torch

from riichi_analysis_engine.average_checkpoints import (
    average_model_states,
    create_average_checkpoint,
)


def test_model_average_preserves_dtype_and_exact_structural_buffers() -> None:
    states = [
        {
            "weight": torch.tensor([1.0, 3.0], dtype=torch.float32),
            "indices": torch.tensor([0, 2], dtype=torch.int64),
        },
        {
            "weight": torch.tensor([3.0, 7.0], dtype=torch.float32),
            "indices": torch.tensor([0, 2], dtype=torch.int64),
        },
    ]

    averaged = average_model_states(states)

    assert averaged["weight"].dtype == torch.float32
    assert torch.equal(averaged["weight"], torch.tensor([2.0, 5.0]))
    assert torch.equal(averaged["indices"], states[0]["indices"])


def test_model_average_rejects_changed_structural_buffers() -> None:
    with pytest.raises(RuntimeError, match="not identical"):
        average_model_states(
            [
                {"indices": torch.tensor([0, 2])},
                {"indices": torch.tensor([0, 3])},
            ]
        )


def _write_checkpoint(path: Path, value: float, samples: int) -> None:
    torch.save(
        {
            "format": "riichi-analysis-model-v12",
            "step": samples // 10,
            "samplesSeen": samples,
            "model": {"weight": torch.tensor([value])},
            "modelArchitecture": {"backbone": "cnn"},
            "modelInput": {"schema": "input"},
            "semanticInput": {"schema": "semantic"},
            "predictionValues": {"dora": [0], "score": [1000]},
            "datasets": {"train": {"manifestSha256": "same"}},
            "learningRateSchedule": {"type": "sample-tail-linear-v1"},
            "trainingCursor": {"nextSample": samples},
            "validation": {"loss": value},
            "resumeAllowed": True,
        },
        path,
    )


def test_average_checkpoint_uses_latest_metadata_but_cannot_resume(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    output = tmp_path / "average.pt"
    _write_checkpoint(first, 1.0, 100)
    _write_checkpoint(second, 3.0, 200)

    report = create_average_checkpoint([first, second], output)
    payload = torch.load(output, map_location="cpu", weights_only=True)

    assert torch.equal(payload["model"]["weight"], torch.tensor([2.0]))
    assert payload["samplesSeen"] == 200
    assert payload["validation"] is None
    assert payload["resumeAllowed"] is False
    assert len(payload["averagedFrom"]) == 2
    assert report["resumeAllowed"] is False
