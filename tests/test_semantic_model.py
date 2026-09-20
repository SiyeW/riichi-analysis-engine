import pytest
import torch

from riichi_analysis_engine.architecture import SemanticModelArchitecture
from riichi_analysis_engine.constants import ACTION_SPACE, TILE_TYPES
from riichi_analysis_engine.kyoku_outcome import OUTCOME_COUNT
from riichi_analysis_engine.model import RiichiAnalysisModel, count_parameters
from riichi_analysis_engine.model_input import (
    MODEL_INPUT_CHANNELS,
    RULE_CONTEXT_START,
    SHARED_MODEL_INPUT_CHANNELS,
)
from riichi_analysis_engine.prediction_values import DORA_VALUES, SCORE_VALUES
from riichi_analysis_engine.rule_context import RULE_TILE_CHANNELS
from riichi_analysis_engine.semantic_input import (
    EVENT_ACTOR,
    EVENT_FIELDS,
    EVENT_TILE,
    EVENT_TYPE,
    PUBLIC_EVENT_TYPE_TO_ID,
)
from riichi_analysis_engine.semantic_model import TileGraphBlock


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


def _prior_transformer_architecture() -> SemanticModelArchitecture:
    return SemanticModelArchitecture(
        backbone="transformer",
        width=32,
        stem_width=40,
        event_width=24,
        backbone_blocks=3,
        event_blocks=1,
        decoder_width=32,
        attention_heads=4,
        transformer_ff_multiplier=2,
        transformer_tile_prior_blocks=1,
        transformer_event_prior_blocks=2,
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


def test_v13_shared_rule_facts_reach_non_policy_outputs() -> None:
    torch.manual_seed(3)
    model = RiichiAnalysisModel(
        format_version=13, architecture=_architecture("cnn")
    ).eval()
    _, events, mask = _inputs()
    first = torch.zeros(2, SHARED_MODEL_INPUT_CHANNELS, TILE_TYPES)
    second = first.clone()
    second[:, RULE_CONTEXT_START + RULE_TILE_CHANNELS] = 1

    first_outputs = model(first, events, mask)
    second_outputs = model(second, events, mask)

    assert not torch.equal(first_outputs["outcome"], second_outputs["outcome"])
    assert not torch.equal(first_outputs["policy"], second_outputs["policy"])
    counts = count_parameters(model)
    assert counts["total"] == counts["input"] + counts["backbone"] + counts["decoder"]


def test_v12_rejects_missing_event_memory() -> None:
    model = RiichiAnalysisModel(format_version=12, architecture=_architecture("cnn"))
    observation, _events, _mask = _inputs()

    with pytest.raises(ValueError, match="requires semantic event memory"):
        model(observation)


def test_cnn_event_padding_cannot_change_a_short_history() -> None:
    torch.manual_seed(7)
    model = RiichiAnalysisModel(format_version=12, architecture=_architecture("cnn"))
    model.eval()
    observation, events, mask = _inputs()

    alone = model(observation[1:2], events[1:2, :2], mask[1:2, :2])
    batched = model(observation, events, mask)

    for name in alone:
        torch.testing.assert_close(alone[name][0], batched[name][1])


def test_transformer_event_order_changes_the_prediction() -> None:
    torch.manual_seed(11)
    model = RiichiAnalysisModel(
        format_version=12, architecture=_architecture("transformer")
    )
    model.eval()
    observation, events, mask = _inputs()
    reordered = events[0:1].clone()
    reordered[:, [1, 2]] = reordered[:, [2, 1]]

    original = model(observation[0:1], events[0:1], mask[0:1])
    changed = model(observation[0:1], reordered, mask[0:1])

    assert not torch.allclose(original["outcome"], changed["outcome"])


def test_prior_transformer_supports_backward_and_ignores_event_padding() -> None:
    torch.manual_seed(17)
    model = RiichiAnalysisModel(
        format_version=12, architecture=_prior_transformer_architecture()
    )
    model.eval()
    observation, events, mask = _inputs()

    alone = model(observation[1:2], events[1:2, :2], mask[1:2, :2])
    batched = model(observation, events, mask)

    for name in alone:
        torch.testing.assert_close(
            alone[name][0], batched[name][1], atol=2e-6, rtol=2e-5
        )
    sum(value.float().sum() for value in batched.values()).backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_old_transformer_architecture_metadata_keeps_historical_topology() -> None:
    current = _architecture("transformer").to_dict()
    for name in (
        "transformer_ff_multiplier",
        "transformer_tile_prior_blocks",
        "transformer_event_prior_blocks",
        "semantic_prior_version",
    ):
        current.pop(name)

    restored = SemanticModelArchitecture.from_dict(current)

    assert restored.transformer_ff_multiplier == 4
    assert restored.transformer_tile_prior_blocks == 0
    assert restored.transformer_event_prior_blocks == 0
    assert restored.semantic_prior_version == 1


def test_current_tile_graph_does_not_invent_honor_adjacency() -> None:
    current = TileGraphBlock(8, prior_version=2)
    legacy = TileGraphBlock(8, prior_version=1)

    assert current.adjacency[0, 1] > 0
    assert current.adjacency[4, TILE_TYPES] > 0
    assert current.adjacency[27, 28] == 0
    assert current.adjacency[31, 32] == 0
    assert legacy.adjacency[27, 28] > 0
    assert legacy.adjacency[31, 32] > 0


def test_current_cnn_event_blocks_cover_a_full_history() -> None:
    architecture = SemanticModelArchitecture(backbone="cnn")
    model = RiichiAnalysisModel(format_version=12, architecture=architecture)
    blocks = model.semantic_model.backbone.event_blocks

    dilations = [
        convolution.dilation[0]
        for block in blocks
        for convolution in (block.first, block.second)
    ]
    assert dilations == [1, 2, 4, 8, 16, 32, 1, 2]
    assert 1 + 2 * sum(dilations) >= 96


def test_legacy_cnn_metadata_preserves_local_event_blocks() -> None:
    metadata = SemanticModelArchitecture(backbone="cnn").to_dict()
    metadata.pop("semantic_prior_version")
    restored = SemanticModelArchitecture.from_dict(metadata)
    model = RiichiAnalysisModel(format_version=12, architecture=restored)

    assert restored.semantic_prior_version == 1
    assert all(
        convolution.dilation == (1,)
        for block in model.semantic_model.backbone.event_blocks
        for convolution in (block.first, block.second)
    )


def test_prior_transformer_candidate_stays_below_the_frozen_cnn_budget() -> None:
    candidate = SemanticModelArchitecture(
        backbone="transformer",
        width=256,
        stem_width=320,
        event_width=160,
        backbone_blocks=6,
        event_blocks=1,
        decoder_width=384,
        attention_heads=8,
        transformer_ff_multiplier=2,
        transformer_tile_prior_blocks=1,
        transformer_event_prior_blocks=2,
    )

    candidate_parameters = count_parameters(
        RiichiAnalysisModel(format_version=12, architecture=candidate)
    )
    cnn_parameters = count_parameters(
        RiichiAnalysisModel(
            format_version=12,
            architecture=SemanticModelArchitecture(backbone="cnn"),
        )
    )

    assert candidate_parameters["total"] == 7_989_137
    assert candidate_parameters["total"] < cnn_parameters["total"] == 8_084_881


def test_policy_context_can_change_without_recomputing_semantic_state() -> None:
    torch.manual_seed(13)
    model = RiichiAnalysisModel(format_version=12, architecture=_architecture("cnn"))
    model.eval()
    observation, events, mask = _inputs()
    selection_observation = observation[0:1].clone()
    selection_observation[:, -1] = 1.0

    state = model.semantic_model.encode(observation[0:1], events[0:1], mask[0:1])
    reused = model.semantic_model.decode_policy(state, selection_observation)
    full = model(selection_observation, events[0:1], mask[0:1])["policy"]

    torch.testing.assert_close(reused, full)
