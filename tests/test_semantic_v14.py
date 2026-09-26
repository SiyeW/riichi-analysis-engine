from dataclasses import replace
from math import comb

import pytest
import torch
from test_dataset import pack_directory

from riichi_analysis_engine.architecture import SemanticModelArchitecture
from riichi_analysis_engine.dataset import PackDataset
from riichi_analysis_engine.joint_counts import (
    joint_count_baseline,
    joint_count_marginals,
    project_joint_counts,
)
from riichi_analysis_engine.losses import (
    LOSS_TERMS_V8,
    LearnedUncertaintyBalancer,
    multitask_loss,
)
from riichi_analysis_engine.model import RiichiAnalysisModel
from riichi_analysis_engine.model_input import SHARED_MODEL_INPUT_CHANNELS
from riichi_analysis_engine.semantic_input import EVENT_FIELDS
from riichi_analysis_engine.semantic_v14 import HIDDEN_SOURCE_ROLES, V14Decoder
from riichi_analysis_engine.train import forward_batch, validate
from riichi_analysis_engine.training_schema import validate_semantic_training_batch


def architecture(backbone="cnn"):
    return SemanticModelArchitecture(
        backbone=backbone,
        width=16,
        stem_width=24,
        event_width=12,
        backbone_blocks=2,
        event_blocks=1,
        decoder_width=20,
        attention_heads=4,
        semantic_design_version=2,
    )


def constraints():
    inventory = torch.full((1, 37), 4)
    inventory[:, (4, 13, 22)] = 3
    inventory[:, 34:] = 1
    inventory[:, :3] = 0
    inventory[:, 3] = 2
    return inventory, torch.tensor([[13, 13, 13, 83]])


def inputs():
    observation = torch.randn(2, SHARED_MODEL_INPUT_CHANNELS, 34) * 0.01
    events = torch.zeros(2, 4, EVENT_FIELDS, dtype=torch.uint8)
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    return observation, events, mask


def test_joint_baseline_matches_total_five_hypergeometric_exactly():
    inventory, capacities = constraints()
    prediction = project_joint_counts(torch.zeros(1, 4, 34, 10), inventory, capacities)
    total, red, physical = prediction.marginals()
    for source, size in enumerate(capacities[0].tolist()):
        exact = torch.tensor(
            [comb(4, k) * comb(118, size - k) / comb(122, size) for k in range(5)]
        )
        torch.testing.assert_close(total[0, source, 4], exact, atol=2e-7, rtol=2e-6)
        assert abs(float(red[0, source, 0, 1]) - size / 122) < 2e-7
    torch.testing.assert_close(
        prediction.probability, prediction.baseline, atol=2e-7, rtol=2e-6
    )
    assert physical.shape == (1, 4, 37, 5)


@pytest.mark.parametrize("scale", [0.2, 1.0, 3.0])
def test_joint_projection_conserves_expectations_and_has_finite_gradients(scale):
    torch.manual_seed(53)
    inventory, capacities = constraints()
    residual = (torch.randn(1, 4, 34, 10) * scale).requires_grad_()
    prediction = project_joint_counts(residual, inventory, capacities)
    _, _, physical = prediction.marginals()
    expected = (physical * torch.arange(5)).sum(-1)
    torch.testing.assert_close(expected.sum(1), inventory.float(), atol=2e-3, rtol=0)
    torch.testing.assert_close(expected.sum(2), capacities.float(), atol=2e-3, rtol=0)
    assert (prediction.probability[prediction.baseline == 0] == 0).all()
    prediction.probability.square().sum().backward()
    assert torch.isfinite(residual.grad).all()
    assert residual.grad.abs().sum() > 0


def test_joint_marginals_preserve_bimodal_counts_and_red_dependence():
    p = torch.zeros(1, 4, 34, 10)
    p[..., 0] = 1
    p[:, :, 4] = 0
    # Half zero, half two fives (one normal + one red): never total one.
    p[:, :, 4, 0] = 0.5
    p[:, :, 4, 5] = 0.5
    total, red, physical = joint_count_marginals(p)
    torch.testing.assert_close(total[0, 0, 4], torch.tensor([0.5, 0, 0.5, 0, 0]))
    torch.testing.assert_close(red[0, 0, 0], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(physical[0, 0, 4], torch.tensor([0.5, 0.5, 0, 0, 0]))


def test_empty_inventory_is_deterministic_zero():
    inventory, capacities = torch.zeros(1, 37), torch.zeros(1, 4)
    p = project_joint_counts(torch.randn(1, 4, 34, 10), inventory, capacities)
    assert torch.isfinite(p.probability).all()
    assert (p.probability[..., 0] == 1).all()


@pytest.mark.parametrize("backbone", ["cnn", "transformer"])
def test_v14_reads_history_into_tiles_and_refines_task_queries(backbone):
    torch.manual_seed(23)
    model = RiichiAnalysisModel(
        format_version=14, architecture=architecture(backbone)
    ).eval()
    observation, events, mask = inputs()
    semantic = model.semantic_model
    state = semantic.encode(observation, events, mask)
    changed = events.clone()
    changed[:, 1, 0] = 2
    state2 = semantic.encode(observation, changed, mask)
    assert not torch.allclose(state.tiles, state2.tiles)
    assert not torch.allclose(state.players, state2.players)
    outputs = semantic.decode(state)
    assert outputs["hidden_joint_residual"].shape == (2, 4, 34, 10)
    assert torch.count_nonzero(outputs["hidden_joint_residual"]) == 0
    # Decoder query paths reach both tile detail and the retained history.
    query_state = replace(
        state,
        tiles=state.tiles.detach().requires_grad_(),
        events=state.events.detach().requires_grad_(),
    )
    semantic.decode(query_state)["furiten_no_yaku"].sum().backward()
    assert query_state.tiles.grad.abs().sum() > 0
    assert query_state.events.grad.abs().sum() > 0
    alone = model(observation[1:2], events[1:2, :2], mask[1:2, :2])
    for key in alone:
        torch.testing.assert_close(alone[key][0], outputs[key][1], atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError, match="exact re-encoding"):
        semantic.decode_policy(state, observation)


def test_hidden_sources_reuse_correct_opponents_and_separate_wall():
    model = RiichiAnalysisModel(format_version=14, architecture=architecture())
    state = model.semantic_model.encode(*inputs())
    assert HIDDEN_SOURCE_ROLES == ("downstream", "opposite", "upstream", "wall")
    torch.testing.assert_close(state.hidden_sources[:, :3], state.players[:, 1:])
    torch.testing.assert_close(state.hidden_sources[:, 3], state.wall)
    decoder = model.semantic_model.decoder
    seeds = decoder.question_seeds(state)
    # Source differences must be identical for every tile, with wall separate.
    expected = decoder.source_projection(state.hidden_sources)
    actual = seeds["hidden_joint_residual"]
    torch.testing.assert_close(
        actual[:, 1:] - actual[:, :1],
        (expected[:, 1:] - expected[:, :1])[:, :, None].expand(-1, -1, 34, -1),
    )


@pytest.mark.parametrize("name", [*V14Decoder.OUTPUT_WIDTHS, "dora_tail"])
def test_every_output_directly_reads_event_memory_without_backbone_help(name):
    torch.manual_seed(83)
    model = RiichiAnalysisModel(format_version=14, architecture=architecture()).eval()
    decoder = model.semantic_model.decoder
    # The zero residual initialization is deliberate. Check the learned route
    # after making its final map nonzero, not a nonexistent initialization gradient.
    if name == "hidden_joint_residual":
        torch.nn.init.normal_(decoder.outputs[name].weight, std=0.01)
    with torch.no_grad():
        state = model.semantic_model.encode(*inputs())
    state = replace(state, events=state.events.detach().requires_grad_())
    calls = []
    handle = decoder.task_attention.register_forward_pre_hook(
        lambda _module, args: calls.append(args[0].shape)
    )
    outputs = decoder(state)
    outputs[name].square().sum().backward()
    handle.remove()
    assert calls == [torch.Size([2, 300, architecture().width])]
    assert torch.isfinite(state.events.grad).all()
    assert state.events.grad[:, :2].abs().sum() > 0
    assert state.events.grad[1, 2:].abs().sum() == 0
    changed = replace(state, events=state.events.detach().clone())
    changed.events[:, 0, 0] += 2
    assert not torch.allclose(outputs[name], decoder(changed)[name])


def test_query_entities_permute_with_their_outputs():
    torch.manual_seed(47)
    model = RiichiAnalysisModel(format_version=14, architecture=architecture()).eval()
    decoder = model.semantic_model.decoder
    torch.nn.init.normal_(decoder.outputs["hidden_joint_residual"].weight, std=0.01)
    state = model.semantic_model.encode(*inputs())
    permuted = replace(state, players=state.players[:, [0, 3, 1, 2]])
    original, changed = decoder(state), decoder(permuted)
    for name in (
        "shanten",
        "furiten_no_yaku",
        "dora_distribution",
        "dora_tail",
        "score_distribution",
        "deal_in_tile",
    ):
        torch.testing.assert_close(changed[name], original[name][:, [2, 0, 1]])
    torch.testing.assert_close(
        changed["hidden_joint_residual"],
        original["hidden_joint_residual"][:, [2, 0, 1, 3]],
    )
    for name in ("outcome", "kyoku_accounts", "placement", "match_score", "policy"):
        torch.testing.assert_close(changed[name], original[name])


def test_v14_consumes_existing_v13_packs_and_all_losses(tmp_path):
    packs = pack_directory(
        tmp_path,
        games=2,
        chunks=1,
        samples=2,
        pack_samples=4,
        training_targets=True,
        model_format=13,
        semantic_history=True,
    )
    batch = next(iter(PackDataset(packs, batch_size=4)))
    batch["hidden_baseline_anchor"][::2] = False
    batch["concealed_count"][:, 0, 4] = 2
    batch["concealed_red_count"][:, 0, 0] = 1
    batch["concealed_count"][:, 1, 4] = 1
    batch["wall_count"][:, 4] = 1
    batch["winner_mask"][:, 0] = True
    batch["score"][:, 0] = 1000
    batch["dora"][:, 0] = 1
    validate_semantic_training_batch(batch, require_hidden_baseline_anchor=True)
    model = RiichiAnalysisModel(format_version=14, architecture=architecture())
    balancer = LearnedUncertaintyBalancer(LOSS_TERMS_V8)
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *balancer.parameters()], lr=1e-4
    )
    output = forward_batch(model, batch)
    total, losses, active, _ = multitask_loss(output, batch, balancer)
    total.backward()
    assert torch.isfinite(total)
    assert set(losses) == set(LOSS_TERMS_V8)
    assert active["hidden_allocation"]
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    untrained = [name for name, p in model.named_parameters() if p.grad is None]
    assert set(untrained) <= {
        "semantic_model.decoder.dora_tail.weight",
        "semantic_model.decoder.dora_tail.bias",
    }
    optimizer.step()
    metrics = validate(model, balancer, [batch], torch.device("cpu"), progress_every=0)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert any("hiddenJointLoss" in key for key in metrics)


def test_joint_anchor_is_soft_target_not_observed_opening_hand():
    inventory, capacities = constraints()
    prediction = project_joint_counts(torch.zeros(1, 4, 34, 10), inventory, capacities)
    # The anchor loss ignores the sampled allocation, using analytic soft labels.
    counts = torch.zeros(1, 4, 37, dtype=torch.long)
    loss = prediction.loss(counts, inventory, torch.tensor([True]))
    assert abs(float(loss)) < 1e-6
    assert torch.isfinite(joint_count_baseline(inventory, capacities)).all()


def test_v14_mixed_precision_five_fusion_and_count_projection():
    model = RiichiAnalysisModel(format_version=14, architecture=architecture())
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        outputs = model(*inputs())
    inventory, capacities = constraints()
    result = project_joint_counts(
        outputs["hidden_joint_residual"],
        inventory.expand(2, -1),
        capacities.expand(2, -1),
    )
    result.probability.square().sum().backward()
    assert torch.isfinite(result.probability).all()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
