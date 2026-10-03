"""Private query selection, reversible raw scaling, and strict format boundary."""

import sys

import pytest
import torch
from test_v17 import inputs, training_batch

from riichi_analysis_engine.architecture import V17Architecture, V18Architecture
from riichi_analysis_engine.export_weights import main as export_weights
from riichi_analysis_engine.losses import multitask_loss
from riichi_analysis_engine.model import RiichiAnalysisModel, count_parameters
from riichi_analysis_engine.model_input import v16_model_input_metadata
from riichi_analysis_engine.prediction_values import DORA_VALUES, SCORE_VALUES
from riichi_analysis_engine.runtime import AnalysisRuntime
from riichi_analysis_engine.semantic_input import semantic_input_metadata
from riichi_analysis_engine.semantic_v18 import restore_raw_facts, scale_raw_facts
from riichi_analysis_engine.train import forward_batch, validate_dataset_input_contract


def small_architecture():
    return V18Architecture(
        width=128, blocks=1, feed_forward_width=256, decoder_width=256
    )


def test_budget_and_no_redundant_identity_or_second_reader():
    model = RiichiAnalysisModel(format_version=18)
    assert count_parameters(model) == {
        "input": 300_800,
        "backbone": 4_209_664,
        "decoder": 6_385_562,
        "total": 10_896_026,
    }
    decoder = model.semantic_model.decoder
    assert not hasattr(decoder, "task_queries")
    assert not hasattr(decoder, "task_ff")
    for blocks in (decoder.query_branches, decoder.task_branches):
        assert set(blocks) == set(decoder.OUTPUT_WIDTHS)
        assert {
            sum(p.numel() for p in block.parameters()) for block in blocks.values()
        } == {263_424}
    assert (
        len(
            {
                block[1].weight.data_ptr()
                for blocks in (decoder.query_branches, decoder.task_branches)
                for block in blocks.values()
            }
        )
        == 22
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_raw_scaling_recovers_absolute_fields_and_preserves_padding(dtype):
    raw = torch.zeros(2, 6, 512, dtype=dtype)
    for owner in range(5):
        raw[:, owner, owner] = 1
        raw[:, owner, 5:12] = torch.tensor([-2.5, 0, 1, 4, 0.25, 10, 100], dtype=dtype)
    scaled = scale_raw_facts(raw)
    assert torch.isfinite(scaled).all()
    torch.testing.assert_close(restore_raw_facts(scaled), raw)
    torch.testing.assert_close(
        scaled[:, :5].float().square().mean(-1),
        torch.ones(2, 5),
        rtol=0.002,
        atol=0.002,
    )
    assert not scaled[:, 5].any()
    doubled = raw.clone()
    doubled[:, 0, 7] *= 2
    assert not torch.equal(scale_raw_facts(doubled)[:, 0], scaled[:, 0])


def test_every_task_reaches_its_own_pre_and_post_blocks_and_common_reader():
    model = RiichiAnalysisModel(format_version=18, architecture=small_architecture())
    decoder = model.semantic_model.decoder
    with torch.no_grad():
        decoder.outputs["hidden_joint_residual"].weight.fill_(0.01)
    for name in decoder.OUTPUT_WIDTHS:
        if name == "policy":
            continue
        model.zero_grad(set_to_none=True)
        forward_batch(model, training_batch())[name].square().sum().backward()
        for blocks in (decoder.query_branches, decoder.task_branches):
            assert blocks[name][1].weight.grad is not None
            assert blocks[name][1].weight.grad.abs().sum() > 0
            assert all(
                block[1].weight.grad is None or not block[1].weight.grad.any()
                for other, block in blocks.items()
                if other != name
            )
        assert decoder.task_attention.in_proj_weight.grad.abs().sum() > 0
        assert all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        )


def test_private_read_selection_and_one_attention_call():
    torch.manual_seed(18)
    model = RiichiAnalysisModel(
        format_version=18, architecture=small_architecture()
    ).eval()
    decoder = model.semantic_model.decoder
    calls = []
    hook = decoder.task_attention.register_forward_pre_hook(
        lambda _, args: calls.append(args[0].detach().clone())
    )
    with torch.no_grad():
        state = model.semantic_model.encode(*inputs())
        seeds = decoder.question_seeds(state)
        assert not torch.allclose(seeds["shanten"], seeds["score_distribution"])
        first = model(*inputs())
        decoder.query_branches["score_distribution"][3].bias[0].add_(1)
        second = model(*inputs())
        for name in first:
            if name != "score_distribution":
                torch.testing.assert_close(first[name], second[name])
        assert not torch.equal(
            first["score_distribution"], second["score_distribution"]
        )
    hook.remove()
    assert len(calls) == 2  # One complete read per forward, not one per task.


def test_real_losses_adam_and_policy_pre_branch(tmp_path):
    model = RiichiAnalysisModel(format_version=18)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    batch = training_batch()
    total, losses, _, _ = multitask_loss(forward_batch(model, batch), batch)
    assert torch.isfinite(total)
    assert all(torch.isfinite(value).all() for value in losses.values())
    total.backward()
    assert (
        model.semantic_model.decoder.query_branches["policy"][1].weight.grad.abs().sum()
        > 0
    )
    optimizer.step()
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    checkpoint = tmp_path / "optimizer-roundtrip.pt"
    torch.save(
        {
            "architecture": model.architecture.to_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        checkpoint,
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored = RiichiAnalysisModel(
        format_version=18,
        architecture=V18Architecture.from_dict(saved["architecture"]),
    )
    restored.load_state_dict(saved["model"], strict=True)
    resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=2e-5)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    # The next update agrees with uninterrupted execution, including Adam moments.
    for current, current_optimizer in (
        (model, optimizer),
        (restored, resumed_optimizer),
    ):
        current_optimizer.zero_grad(set_to_none=True)
        next_loss, _, _, _ = multitask_loss(forward_batch(current, batch), batch)
        next_loss.backward()
        current_optimizer.step()
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(
            restored.state_dict()[name], expected, rtol=0, atol=0
        )


def test_actual_raw_fields_padding_and_candidate_order():
    import numpy as np

    from riichi_analysis_engine.v16_candidates import encode_candidate_set

    model = RiichiAnalysisModel(
        format_version=18, architecture=small_architecture()
    ).eval()
    values = inputs()
    actions = [
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": True},
        {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": False},
    ]
    features = torch.from_numpy(np.asarray(encode_candidate_set(actions, 0)))[None]
    mask = torch.ones(1, 3, dtype=torch.bool)
    with torch.no_grad():
        state = model.semantic_model.encode(*values, features, mask)
        torch.testing.assert_close(
            restore_raw_facts(scale_raw_facts(state.raw_memory)), state.raw_memory
        )
        first = model(*values, features, mask)
        reordered = model(*values, features.flip(1), mask)
        torch.testing.assert_close(first["policy"], reordered["policy"].flip(1))
        padded_events = values[1].clone()
        padded_events[:, 2] = torch.tensor([11, 4, 3, 37, 36, 35, 34, 33, 1])
        padded = model(values[0], padded_events, *values[2:], features, mask)
        for name in first:
            torch.testing.assert_close(first[name], padded[name])
        masked = model(*values, features, torch.tensor([[True, False, True]]))["policy"]
        assert torch.isneginf(masked[0, 1])
        assert torch.isfinite(masked[0, [0, 2]]).all()


@pytest.mark.parametrize("width", [256, 512])
def test_export_load_strict_topology_and_unchanged_data(tmp_path, monkeypatch, width):
    architecture = V18Architecture(
        width=width, feed_forward_width=width * 4, decoder_width=width * 2
    )
    assert V18Architecture.from_dict(architecture.to_dict()) == architecture
    model = RiichiAnalysisModel(format_version=18, architecture=architecture)
    checkpoint, exported = tmp_path / "checkpoint.pt", tmp_path / "weights.pt"
    torch.save(
        {
            "format": "riichi-analysis-model-v18",
            "model": model.state_dict(),
            "modelArchitecture": model.architecture.to_dict(),
            "modelInput": v16_model_input_metadata(),
            "semanticInput": semantic_input_metadata(),
            "predictionValues": {
                "dora": list(DORA_VALUES),
                "score": list(SCORE_VALUES),
            },
            "step": 1,
            "samplesSeen": 32,
        },
        checkpoint,
    )
    monkeypatch.setattr(sys, "argv", ["export_weights", str(checkpoint), str(exported)])
    export_weights()
    monkeypatch.setattr(
        "riichi_analysis_engine.runtime._load_player_state", lambda: object
    )
    runtime = AnalysisRuntime(exported, "cpu")
    assert runtime.format_version == 18
    assert runtime.model.architecture == architecture
    if width == 512:
        assert count_parameters(runtime.model)["total"] == 43_287_706
    else:
        assert count_parameters(runtime.model)["total"] == 10_896_026
        assert 43_584_104 < exported.stat().st_size < 45_000_000
    with torch.no_grad():
        for name, expected in model(*inputs()).items():
            torch.testing.assert_close(runtime.model(*inputs())[name], expected)
    with pytest.raises(TypeError):
        RiichiAnalysisModel(format_version=18, architecture=V17Architecture())
    with pytest.raises(TypeError):
        RiichiAnalysisModel(format_version=17, architecture=small_architecture())
    from riichi_analysis_engine.model_input import SHARED_MODEL_INPUT_CHANNELS
    from riichi_analysis_engine.semantic_input import EVENT_MEMORY_SCHEMA_ID
    from riichi_analysis_engine.storage import V16_TRAINING_TARGET_SCHEMA_ID

    metadata = {
        "modelInputSchema": v16_model_input_metadata()["schema"],
        "observationChannels": SHARED_MODEL_INPUT_CHANNELS,
        "eventMemorySchema": EVENT_MEMORY_SCHEMA_ID,
        "trainingTargetSchema": V16_TRAINING_TARGET_SCHEMA_ID,
    }
    validate_dataset_input_contract({"train": metadata}, 18)
    with pytest.raises(RuntimeError):
        validate_dataset_input_contract(
            {"train": {**metadata, "trainingTargetSchema": "old"}}, 18
        )
