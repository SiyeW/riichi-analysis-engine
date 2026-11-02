"""Initial equivalence, active gradients, exact roles and portable V19 paths."""

import copy
import sys

import pytest
import torch
from test_v17 import inputs, training_batch

from riichi_analysis_engine import export_weights, train
from riichi_analysis_engine.architecture import V18Architecture, V19Architecture
from riichi_analysis_engine.losses import (
    LOSS_TERMS_V8,
    LearnedUncertaintyBalancer,
    multitask_loss,
)
from riichi_analysis_engine.model import RiichiAnalysisModel, count_parameters
from riichi_analysis_engine.model_input import v16_model_input_metadata
from riichi_analysis_engine.prediction_values import DORA_VALUES, SCORE_VALUES
from riichi_analysis_engine.reader_growth import (
    NEW_PARAMETERS,
    PREFIX,
    deepen_task_reader,
)
from riichi_analysis_engine.runtime import AnalysisRuntime
from riichi_analysis_engine.semantic_input import semantic_input_metadata
from riichi_analysis_engine.train import forward_batch
from riichi_analysis_engine.v16_candidates import encode_candidate_set


def parent_model():
    torch.manual_seed(19)
    return RiichiAnalysisModel(
        format_version=18,
        architecture=V18Architecture(
            width=128,
            blocks=1,
            attention_heads=8,
            feed_forward_width=256,
            decoder_width=256,
        ),
    )


def test_default_shape_budget_and_strict_format():
    model = RiichiAnalysisModel(format_version=19)
    assert model.architecture == V19Architecture()
    assert count_parameters(model)["total"] == 29_727_130
    assert len(model.state_dict()) == 319
    assert len(model.semantic_model.decoder.task_branches) == 11
    with pytest.raises(TypeError):
        RiichiAnalysisModel(format_version=18, architecture=V19Architecture())
    with pytest.raises(TypeError):
        RiichiAnalysisModel(format_version=19, architecture=V18Architecture())
    with pytest.raises(ValueError):
        V18Architecture.from_dict(V19Architecture().to_dict())
    with pytest.raises(ValueError):
        V19Architecture(readout_stages=1)
    assert V19Architecture.from_dict(V19Architecture().to_dict()) == V19Architecture()


def test_all_old_tensors_rng_and_full_initial_predictions_preserved():
    old = parent_model()
    before = copy.deepcopy(old.state_dict())
    rng = torch.get_rng_state().clone()
    new, mapping = deepen_task_reader(old)
    assert torch.equal(torch.get_rng_state(), rng)
    assert set(new.state_dict()) - set(before) == {
        PREFIX + name for name in NEW_PARAMETERS
    }
    assert mapping["newArchitectureUpdates"] == 0
    for name, expected in before.items():
        assert torch.equal(expected, old.state_dict()[name])
        assert torch.equal(expected, new.state_dict()[name])
    with torch.no_grad():
        first, second = old(*inputs()), new(*inputs())
    for name, expected in first.items():
        torch.testing.assert_close(expected, second[name], rtol=0, atol=0)


def test_reader_transfer_does_not_silently_cast_parent_precision():
    with pytest.raises(ValueError, match="FP32"):
        deepen_task_reader(parent_model().double())


def test_old_gradients_same_and_new_reader_opens_after_output_update():
    old = parent_model()
    new, _ = deepen_task_reader(old)
    batch = training_batch()
    for model in (old, new):
        total, _, _, _ = multitask_loss(forward_batch(model, batch), batch)
        total.backward()
    new_named = dict(new.named_parameters())
    for name, old_parameter in old.named_parameters():
        current = new_named[name]
        if old_parameter.grad is None:
            assert current.grad is None
        else:
            torch.testing.assert_close(old_parameter.grad, current.grad, rtol=0, atol=0)
    decoder = new.semantic_model.decoder
    assert decoder.refine_attention.out_proj.weight.grad.abs().sum() > 0
    assert not decoder.refine_attention.in_proj_weight.grad.any()
    assert not decoder.refine_query_norm.weight.grad.any()
    optimizer = torch.optim.AdamW(new.parameters(), lr=1e-4)
    optimizer.step()
    new.zero_grad(set_to_none=True)
    total, _, _, _ = multitask_loss(forward_batch(new, batch), batch)
    total.backward()
    assert decoder.refine_attention.in_proj_weight.grad.abs().sum() > 0
    assert decoder.refine_query_norm.weight.grad.abs().sum() > 0
    assert all(
        torch.isfinite(p.grad).all() for p in new.parameters() if p.grad is not None
    )


def test_every_task_has_two_common_reads_without_other_private_paths():
    model, _ = deepen_task_reader(parent_model())
    decoder = model.semantic_model.decoder
    # Open the new residual on a disposable model, so both stages are exercised.
    with torch.no_grad():
        decoder.refine_attention.out_proj.weight.copy_(
            decoder.task_attention.out_proj.weight
        )
        decoder.outputs["hidden_joint_residual"].weight.fill_(0.01)
    calls = []
    hooks = [
        attention.register_forward_pre_hook(
            lambda module, args: calls.append((module, args[0].shape[1], args[1]))
        )
        for attention in (decoder.task_attention, decoder.refine_attention)
    ]
    for task in decoder.OUTPUT_WIDTHS:
        if task == "policy":
            continue  # A real labeled nonempty policy batch is covered below.
        model.zero_grad(set_to_none=True)
        calls.clear()
        model(*inputs())[task].square().sum().backward()
        assert len(calls) == 2
        assert calls[0][1] == calls[1][1] == 254
        assert calls[0][2] is calls[1][2]
        assert decoder.refine_attention.in_proj_weight.grad.abs().sum() > 0
        for blocks in (decoder.query_branches, decoder.task_branches):
            assert blocks[task][1].weight.grad.abs().sum() > 0
            assert all(
                block[1].weight.grad is None or not block[1].weight.grad.any()
                for name, block in blocks.items()
                if name != task
            )
    for hook in hooks:
        hook.remove()


def test_second_query_depends_on_first_hidden_read_not_output_heads():
    model, _ = deepen_task_reader(parent_model())
    decoder = model.semantic_model.decoder
    state = model.semantic_model.encode(*inputs())
    with torch.no_grad():
        expected, _, _, memory, _ = decoder._first_task_read(state)
        expected = decoder.refine_query_norm(expected)
        captured = []
        hook = decoder.refine_attention.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].clone())
        )
        model(*inputs())
        decoder.outputs["score_distribution"].bias.add_(100)
        model(*inputs())
    hook.remove()
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(captured[1], expected, rtol=0, atol=0)
    assert memory.shape[1] == 2 * (43 + inputs()[1].shape[1])


def test_candidate_order_padding_mask_and_policy_loss_reach_new_reader():
    model, _ = deepen_task_reader(parent_model())
    decoder = model.semantic_model.decoder
    with torch.no_grad():
        decoder.refine_attention.out_proj.weight.copy_(
            decoder.task_attention.out_proj.weight
        )
    actions = [
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": True},
        {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": False},
    ]
    features = torch.from_numpy(encode_candidate_set(actions, 0))[None]
    mask = torch.ones(1, 3, dtype=torch.bool)
    first = model(*inputs(), features, mask)
    second = model(*inputs(), features.flip(1), mask)
    torch.testing.assert_close(first["policy"], second["policy"].flip(1))
    for name in first:
        if name != "policy":
            torch.testing.assert_close(first[name], second[name])
    values = list(inputs())
    original = model(*values)
    values[1] = values[1].clone()
    values[1][:, -1] = torch.tensor([11, 4, 3, 37, 36, 35, 34, 33, 1])
    padded = model(*values)
    for name in original:
        torch.testing.assert_close(original[name], padded[name])
    masked = model(*inputs(), features, torch.tensor([[True, False, True]]))
    assert torch.isneginf(masked["policy"][0, 1])
    policy_loss = torch.nn.functional.cross_entropy(first["policy"], torch.tensor([1]))
    policy_loss.backward()
    assert decoder.query_branches["policy"][1].weight.grad.abs().sum() > 0
    assert decoder.refine_attention.in_proj_weight.grad.abs().sum() > 0
    batch = training_batch()
    total, losses, _, _ = multitask_loss(forward_batch(model, batch), batch)
    assert torch.isfinite(total) and torch.isfinite(losses["policy"])


def test_export_runtime_and_backward_compatible_architecture_boundary(
    tmp_path, monkeypatch
):
    model, _ = deepen_task_reader(parent_model())
    checkpoint, output = tmp_path / "checkpoint.pt", tmp_path / "inference.pt"
    torch.save(
        {
            "format": "riichi-analysis-model-v19",
            "model": model.state_dict(),
            "modelArchitecture": model.architecture.to_dict(),
            "modelInput": v16_model_input_metadata(),
            "semanticInput": semantic_input_metadata(),
            "predictionValues": {
                "dora": list(DORA_VALUES),
                "score": list(SCORE_VALUES),
            },
            "readerTransfer": {"newArchitectureUpdates": 0},
            "resumeAllowed": False,
        },
        checkpoint,
    )
    monkeypatch.setattr(sys, "argv", ["export", str(checkpoint), str(output)])
    export_weights.main()
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert payload["training"]["initializationOnly"] is True
    monkeypatch.setattr(
        "riichi_analysis_engine.runtime._load_player_state", lambda: object
    )
    runtime = AnalysisRuntime(output, "cpu")
    assert runtime.format_version == 19
    for name, expected in model(*inputs()).items():
        torch.testing.assert_close(
            expected, runtime.model(*inputs())[name], rtol=0, atol=0
        )
    payload["architecture"]["model"]["readout_stages"] = 1
    bad = tmp_path / "wrong.pt"
    torch.save(payload, bad)
    with pytest.raises(RuntimeError, match="architecture"):
        AnalysisRuntime(bad, "cpu")


def test_trained_reader_adam_rng_and_next_update_survive_checkpoint(tmp_path):
    model, _ = deepen_task_reader(parent_model())
    balancer = LearnedUncertaintyBalancer(LOSS_TERMS_V8)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": 1e-5},
            {"params": balancer.parameters(), "lr": 1e-4},
        ]
    )
    batch = training_batch()
    loss = multitask_loss(forward_batch(model, batch), batch, balancer)[0]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert model.semantic_model.decoder.refine_attention.out_proj.weight.abs().sum() > 0
    checkpoint = tmp_path / "trained-reader.pt"
    train.save_checkpoint(
        checkpoint,
        model,
        optimizer,
        torch.amp.GradScaler("cuda", enabled=False),
        balancer,
        model.architecture,
        step=1,
        samples_seen=len(batch["policy"]),
        analysis_samples_seen=int(batch["analysis_active"].sum()),
        batch_size=len(batch["policy"]),
        pass_complete=False,
        parameters=count_parameters(model),
        datasets={},
        environment={},
        validation=None,
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["modelArchitecture"]["readout_stages"] == 2
    restored = RiichiAnalysisModel(
        format_version=19,
        architecture=V19Architecture.from_dict(saved["modelArchitecture"]),
    )
    restored.load_state_dict(saved["model"], strict=True)
    balance2 = LearnedUncertaintyBalancer(LOSS_TERMS_V8)
    balance2.load_state_dict(saved["lossBalancer"]["state"], strict=True)
    optimizer2 = torch.optim.AdamW(
        [{"params": restored.parameters()}, {"params": balance2.parameters()}]
    )
    optimizer2.load_state_dict(saved["optimizer"])
    for current, balance, opt in (
        (model, balancer, optimizer),
        (restored, balance2, optimizer2),
    ):
        train.restore_random_state(saved["randomState"])
        opt.zero_grad(set_to_none=True)
        total = multitask_loss(forward_batch(current, batch), batch, balance)[0]
        total.backward()
        assert (
            current.semantic_model.decoder.refine_attention.in_proj_weight.grad.abs().sum()
            > 0
        )
        opt.step()
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(
            expected, restored.state_dict()[name], rtol=0, atol=0
        )
    torch.testing.assert_close(
        balancer.log_variance, balance2.log_variance, rtol=0, atol=0
    )
