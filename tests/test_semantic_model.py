import pytest
import torch

from riichi_analysis_engine.architecture import SemanticModelArchitecture
from riichi_analysis_engine.constants import ACTION_SPACE, TILE_TYPES
from riichi_analysis_engine.kyoku_outcome import OUTCOME_COUNT
from riichi_analysis_engine.model import RiichiAnalysisModel, count_parameters
from riichi_analysis_engine.model_input import MODEL_INPUT_CHANNELS
from riichi_analysis_engine.prediction_values import DORA_VALUES, SCORE_VALUES
from riichi_analysis_engine.semantic_input import (
    EVENT_ACTOR,
    EVENT_FIELDS,
    EVENT_TILE,
    EVENT_TYPE,
    PUBLIC_EVENT_TYPE_TO_ID,
)


def _architecture(backbone: str) -> SemanticModelArchitecture:
    return SemanticModelArchitecture(
        backbone=backbone,
        width=32,
        stem_width=48,
        event_width=24,
        backbone_blocks=2,
        event_blocks=1,
        decoder_width=40,
        attention_heads=4,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    observation = torch.zeros(2, MODEL_INPUT_CHANNELS, TILE_TYPES)
    events = torch.zeros(2, 3, EVENT_FIELDS, dtype=torch.uint8)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    events[:, 0, EVENT_TYPE] = PUBLIC_EVENT_TYPE_TO_ID["start_kyoku"]
    events[:, 0, EVENT_TILE] = 1
    events[:, 1, EVENT_TYPE] = PUBLIC_EVENT_TYPE_TO_ID["tsumo"]
    events[:, 1, EVENT_ACTOR] = 1
    events[:, 1, EVENT_TILE] = 35
    events[:, 2, EVENT_TYPE] = PUBLIC_EVENT_TYPE_TO_ID["dahai"]
    events[:, 2, EVENT_ACTOR] = 1
    events[:, 2, EVENT_TILE] = 35
    return observation, events, mask


@pytest.mark.parametrize("backbone", ["cnn", "transformer"])
def test_v12_backbones_share_outputs_and_support_backward(backbone: str) -> None:
    architecture = _architecture(backbone)
    assert SemanticModelArchitecture.from_dict(architecture.to_dict()) == architecture
    model = RiichiAnalysisModel(format_version=12, architecture=architecture)
    observation, events, mask = _inputs()

    outputs = model(observation, events, mask)

    assert outputs["shanten"].shape == (2, 3, 7)
    assert outputs["deal_in_tile"].shape == (2, 3, TILE_TYPES)
    assert outputs["hidden_source_affinity"].shape == (2, 4, TILE_TYPES)
    assert outputs["hidden_red_source"].shape == (2, 3, 4)
    assert outputs["dora_distribution"].shape == (2, 3, len(DORA_VALUES))
    assert outputs["score_distribution"].shape == (2, 3, len(SCORE_VALUES))
    assert outputs["outcome"].shape == (2, OUTCOME_COUNT)
    assert outputs["placement"].shape == (2, 24)
    assert outputs["policy"].shape == (2, ACTION_SPACE)
    sum(value.float().sum() for value in outputs.values()).backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    counts = count_parameters(model)
    assert counts["total"] == counts["input"] + counts["backbone"] + counts["decoder"]


def test_v12_rejects_missing_event_memory() -> None:
    model = RiichiAnalysisModel(format_version=12, architecture=_architecture("cnn"))
    observation, _events, _mask = _inputs()

    with pytest.raises(ValueError, match="requires semantic event memory"):
        model(observation)
