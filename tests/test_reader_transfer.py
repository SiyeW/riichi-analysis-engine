"""Full-state migration is explicit but cannot resume an unchosen new pass."""

import copy

import pytest
import torch
from test_training_phase import assert_nested_equal
from test_v17 import training_batch
from test_v19 import parent_model

from riichi_analysis_engine import train
from riichi_analysis_engine.architecture import V19Architecture
from riichi_analysis_engine.corpus_extension import file_sha256
from riichi_analysis_engine.losses import (
    LOSS_TERMS_V8,
    LearnedUncertaintyBalancer,
    multitask_loss,
)
from riichi_analysis_engine.model import RiichiAnalysisModel, count_parameters
from riichi_analysis_engine.prepare_reader_checkpoint import prepare_reader_checkpoint


def parent_checkpoint(tmp_path):
    model = parent_model()
    balancer = LearnedUncertaintyBalancer(LOSS_TERMS_V8)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": 1e-6, "weight_decay": 1e-4},
            {"params": balancer.parameters(), "lr": 1e-4, "weight_decay": 0},
        ]
    )
    batch = training_batch()
    total, _, _, _ = multitask_loss(train.forward_batch(model, batch), batch, balancer)
    total.backward()
    optimizer.step()
    path = tmp_path / "parent.pt"
    train.save_checkpoint(
        path,
        model,
        optimizer,
        torch.amp.GradScaler("cuda", enabled=False),
        balancer,
        model.architecture,
        step=1,
        samples_seen=16,
        analysis_samples_seen=8,
        batch_size=16,
        pass_complete=True,
        parameters=count_parameters(model),
        datasets={
            "train": {"manifestSha256": "train"},
            "validation": {"manifestSha256": "validation"},
        },
        environment={"sourceRevision": "parent"},
        validation=None,
    )
    return path


def test_named_adam_scaler_rng_and_balancer_are_inherited_without_new_cursor(tmp_path):
    parent_path = parent_checkpoint(tmp_path)
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    parent_hash = file_sha256(parent_path)
    rng = torch.get_rng_state().clone()
    output = tmp_path / "reader.pt"
    receipt = prepare_reader_checkpoint(parent_path, output, parent_sha256=parent_hash)
    assert torch.equal(rng, torch.get_rng_state())
    assert file_sha256(parent_path) == parent_hash
    saved = torch.load(output, map_location="cpu", weights_only=True)
    assert saved["format"] == "riichi-analysis-model-v19"
    assert saved["resumeAllowed"] is False
    assert "trainingCursor" not in saved
    assert "architectureTransition" not in saved
    assert "modelGeneration" not in saved
    assert saved["training"]["newArchitectureUpdates"] == 0
    assert saved["samplesSeen"] == saved["step"] == 0
    for field in ("lossBalancer", "randomState", "scaler"):
        assert_nested_equal(parent[field], saved[field])
    assert_nested_equal(
        parent["trainingCursor"], saved["parentMetadata"]["trainingCursor"]
    )
    for name, expected in parent["model"].items():
        assert torch.equal(expected, saved["model"][name])
    old_names = list(parent["model"])
    model = RiichiAnalysisModel(
        format_version=19,
        architecture=V19Architecture.from_dict(saved["modelArchitecture"]),
    )
    model.load_state_dict(saved["model"], strict=True)
    names = [name for name, _ in model.named_parameters()]
    old_ids = parent["optimizer"]["param_groups"][0]["params"]
    for name, old_id in zip(old_names, old_ids, strict=True):
        new_id = names.index(name)
        if old_id in parent["optimizer"]["state"]:
            assert_nested_equal(
                parent["optimizer"]["state"][old_id],
                saved["optimizer"]["state"][new_id],
            )
        else:
            assert new_id not in saved["optimizer"]["state"]
    for name in set(names) - set(old_names):
        assert names.index(name) not in saved["optimizer"]["state"]
    old_balance = parent["optimizer"]["param_groups"][1]["params"][0]
    new_balance = saved["optimizer"]["param_groups"][1]["params"][0]
    assert new_balance == len(names) and new_balance != old_balance
    assert_nested_equal(
        parent["optimizer"]["state"][old_balance],
        saved["optimizer"]["state"][new_balance],
    )
    assert receipt["newReaderAdamStates"] == receipt["trainingUpdates"] == 0
    assert receipt["numericalIdentityNotYetVerified"] is True
    with pytest.raises(FileExistsError):
        prepare_reader_checkpoint(parent_path, output, parent_sha256=parent_hash)


def test_old_parameter_next_update_matches_after_named_optimizer_restore(tmp_path):
    parent_path = parent_checkpoint(tmp_path)
    output = tmp_path / "reader.pt"
    prepare_reader_checkpoint(
        parent_path, output, parent_sha256=file_sha256(parent_path)
    )
    old_state = torch.load(parent_path, map_location="cpu", weights_only=True)
    new_state = torch.load(output, map_location="cpu", weights_only=True)
    old = parent_model()
    new = RiichiAnalysisModel(
        format_version=19,
        architecture=V19Architecture.from_dict(new_state["modelArchitecture"]),
    )
    for model, state in ((old, old_state), (new, new_state)):
        model.load_state_dict(state["model"], strict=True)
        balance = LearnedUncertaintyBalancer(LOSS_TERMS_V8)
        balance.load_state_dict(state["lossBalancer"]["state"], strict=True)
        optimizer = torch.optim.AdamW(
            [{"params": model.parameters()}, {"params": balance.parameters()}]
        )
        optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        loss, _, _, _ = multitask_loss(
            train.forward_batch(model, training_batch()), training_batch(), balance
        )
        loss.backward()
        optimizer.step()
    for name, expected in old.state_dict().items():
        torch.testing.assert_close(expected, new.state_dict()[name], rtol=0, atol=0)


def test_wrong_parent_hash_fails_before_publication(tmp_path):
    parent = parent_checkpoint(tmp_path)
    output = tmp_path / "no-output.pt"
    with pytest.raises(ValueError, match="checksum"):
        prepare_reader_checkpoint(parent, output, parent_sha256="0" * 64)
    assert not output.exists()


def test_incomplete_old_adam_is_rejected_instead_of_fabricated(tmp_path):
    parent = parent_checkpoint(tmp_path)
    value = torch.load(parent, map_location="cpu", weights_only=True)
    first_state = next(iter(value["optimizer"]["state"].values()))
    del first_state["exp_avg_sq"]
    broken = tmp_path / "broken-parent.pt"
    torch.save(value, broken)
    output = tmp_path / "not-published.pt"
    with pytest.raises(ValueError, match="incomplete inherited Adam"):
        prepare_reader_checkpoint(broken, output, parent_sha256=file_sha256(broken))
    assert not output.exists()
