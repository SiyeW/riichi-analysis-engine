"""New model format, unchanged V16 data, and independent task processing."""

import sys

import numpy as np
import pytest
import torch
from test_convert import _OneDiscardPlayerState

from riichi_analysis_engine.architecture import V15Architecture, V17Architecture
from riichi_analysis_engine.convert import convert_game
from riichi_analysis_engine.export_weights import main as export_weights
from riichi_analysis_engine.losses import multitask_loss
from riichi_analysis_engine.model import RiichiAnalysisModel, count_parameters
from riichi_analysis_engine.model_input import (
    SHARED_MODEL_INPUT_CHANNELS,
    v16_model_input_metadata,
)
from riichi_analysis_engine.prediction_values import DORA_VALUES, SCORE_VALUES
from riichi_analysis_engine.runtime import AnalysisRuntime
from riichi_analysis_engine.semantic_input import (
    materialize_event_memory,
    semantic_input_metadata,
)
from riichi_analysis_engine.train import forward_batch, validate_dataset_input_contract
from riichi_analysis_engine.training_schema import validate_v16_training_batch
from riichi_analysis_engine.v15_facts import V15_FACTS_WIDTH
from riichi_analysis_engine.v16_candidates import encode_candidate_set


def small_architecture():
    # Exact-fact event fields need at least their fixed 104 slots.
    return V17Architecture(
        width=128,
        blocks=1,
        attention_heads=8,
        feed_forward_width=256,
        decoder_width=256,
    )


def inputs():
    obs = torch.zeros(1, SHARED_MODEL_INPUT_CHANNELS, 34)
    facts = torch.zeros(1, V15_FACTS_WIDTH)
    facts[:, 24] = 1
    events = torch.zeros(1, 3, 9, dtype=torch.uint8)
    events[:, :2, 0] = 1
    mask = torch.tensor([[True, True, False]])
    return obs, events, mask, facts, torch.zeros(1, 4, 37)


def test_default_budget_and_independent_equal_branches():
    model = RiichiAnalysisModel(format_version=17)
    assert model.architecture == V17Architecture()
    assert count_parameters(model) == {
        "input": 1_125_888,
        "backbone": 16_807_936,
        "decoder": 13_797_018,
        "total": 31_730_842,
    }
    decoder = model.semantic_model.decoder
    assert not hasattr(decoder, "task_ff")
    assert set(decoder.task_branches) == set(decoder.OUTPUT_WIDTHS)
    assert len(decoder.task_branches) == 11
    assert {
        sum(p.numel() for p in block.parameters())
        for block in decoder.task_branches.values()
    } == {1_051_136}
    weights = [block[1].weight for block in decoder.task_branches.values()]
    assert len({weight.data_ptr() for weight in weights}) == 11


def test_each_task_uses_only_its_private_block_and_still_reaches_shared_reader():
    model = RiichiAnalysisModel(format_version=17, architecture=small_architecture())
    decoder = model.semantic_model.decoder
    for name in decoder.OUTPUT_WIDTHS:
        model.zero_grad(set_to_none=True)
        output = model(*inputs())
        # Hidden residual is initially zero by the retained baseline contract.
        if name == "hidden_joint_residual":
            with torch.no_grad():
                decoder.outputs[name].weight.fill_(0.01)
            output = model(*inputs())
        if name == "policy":
            continue  # Nonempty candidates are tested separately below.
        output[name].sum().backward()
        assert decoder.task_branches[name][1].weight.grad is not None
        assert decoder.task_attention.in_proj_weight.grad is not None
        assert all(
            block[1].weight.grad is None
            for other, block in decoder.task_branches.items()
            if other != name
        )
        assert all(
            torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.grad is not None
        )


def test_padding_candidates_and_private_changes_are_task_local():
    torch.manual_seed(17)
    model = RiichiAnalysisModel(
        format_version=17, architecture=small_architecture()
    ).eval()
    values = inputs()
    actions = [
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": True},
        {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": False},
    ]
    features = torch.from_numpy(encode_candidate_set(actions, 0))[None]
    mask = torch.ones(1, 3, dtype=torch.bool)
    with torch.no_grad():
        first = model(*values, features, mask)
        reordered = model(*values, features.flip(1), mask)
        torch.testing.assert_close(first["policy"], reordered["policy"].flip(1))
        changed_events = values[1].clone()
        changed_events[:, 2] = torch.tensor([11, 4, 3, 37, 36, 35, 34, 33, 1])
        padded = model(values[0], changed_events, *values[2:], features, mask)
        for name in first:
            torch.testing.assert_close(first[name], padded[name])
        model.semantic_model.decoder.task_branches["score_distribution"][3].bias.add_(1)
        changed = model(*values, features, mask)
        for name in first:
            if name != "score_distribution":
                torch.testing.assert_close(first[name], changed[name])
        assert not torch.equal(
            first["score_distribution"], changed["score_distribution"]
        )
    model.zero_grad(set_to_none=True)
    model(*values, features, mask)["policy"].sum().backward()
    assert (
        model.semantic_model.decoder.task_branches["policy"][1].weight.grad is not None
    )
    masked = model(*values, features, torch.tensor([[True, False, True]]))["policy"]
    assert torch.isneginf(masked[0, 1])
    assert torch.isfinite(masked[0, [0, 2]]).all()
    assert model(*values)["policy"].shape == (1, 0)


def converted_fixture():
    events = [
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "dora_marker": "1m",
            "scores": [25000] * 4,
            "tehais": [["5m"], [], [], []],
        },
        {"type": "tsumo", "actor": 0, "pai": "5m"},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        {"type": "ryukyoku", "deltas": [0] * 4},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]
    return convert_game(
        events, "fixture-v17", model_format=16, player_state_type=_OneDiscardPlayerState
    )


def training_batch():
    converted = converted_fixture()
    arrays = converted.arrays
    tokens, mask = materialize_event_memory(
        converted.event_catalog,
        arrays["history_start"],
        arrays["history_length"],
        arrays["perspective"],
    )
    return {
        **{name: torch.from_numpy(np.asarray(value)) for name, value in arrays.items()},
        "event_tokens": torch.from_numpy(tokens),
        "event_mask": torch.from_numpy(mask),
    }


@pytest.mark.parametrize("architecture", [small_architecture(), V17Architecture()])
def test_v16_converted_batch_trains_v17_with_real_losses_and_adam(architecture):
    batch = training_batch()
    validate_v16_training_batch(batch)
    model = RiichiAnalysisModel(format_version=17, architecture=architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    total, losses, _, _ = multitask_loss(forward_batch(model, batch), batch)
    assert torch.isfinite(total)
    assert all(torch.isfinite(loss).all() for loss in losses.values())
    total.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert torch.isfinite(multitask_loss(forward_batch(model, batch), batch)[0])


def test_export_load_and_strict_version_boundary(tmp_path, monkeypatch):
    model = RiichiAnalysisModel(format_version=17, architecture=small_architecture())
    checkpoint, exported = tmp_path / "checkpoint.pt", tmp_path / "weights.pt"
    torch.save(
        {
            "format": "riichi-analysis-model-v17",
            "model": model.state_dict(),
            "modelArchitecture": model.architecture.to_dict(),
            "modelInput": v16_model_input_metadata(),
            "semanticInput": semantic_input_metadata(),
            "predictionValues": {
                "dora": list(DORA_VALUES),
                "score": list(SCORE_VALUES),
            },
            "step": 1,
            "samplesSeen": 16,
            "optimizer": {"unneeded": 1},
        },
        checkpoint,
    )
    monkeypatch.setattr(sys, "argv", ["export_weights", str(checkpoint), str(exported)])
    export_weights()
    payload = torch.load(exported, weights_only=True)
    assert "optimizer" not in payload
    assert payload["architecture"]["modelInput"] == v16_model_input_metadata()
    monkeypatch.setattr(
        "riichi_analysis_engine.runtime._load_player_state", lambda: object
    )
    runtime = AnalysisRuntime(exported, "cpu")
    assert runtime.format_version == 17
    assert isinstance(runtime.model.architecture, V17Architecture)
    with torch.no_grad():
        for name, expected in model(*inputs()).items():
            torch.testing.assert_close(runtime.model(*inputs())[name], expected)
    payload["format"] = "riichi-analysis-model-v16"
    torch.save(payload, exported)
    with pytest.raises(RuntimeError):
        AnalysisRuntime(exported, "cpu")
    with pytest.raises(TypeError):
        RiichiAnalysisModel(format_version=17, architecture=V15Architecture())


def test_v17_reuses_v16_dataset_contract_and_rejects_old_schemas():
    from riichi_analysis_engine.semantic_input import EVENT_MEMORY_SCHEMA_ID
    from riichi_analysis_engine.storage import V16_TRAINING_TARGET_SCHEMA_ID

    metadata = {
        "modelInputSchema": v16_model_input_metadata()["schema"],
        "observationChannels": SHARED_MODEL_INPUT_CHANNELS,
        "eventMemorySchema": EVENT_MEMORY_SCHEMA_ID,
        "trainingTargetSchema": V16_TRAINING_TARGET_SCHEMA_ID,
    }
    validate_dataset_input_contract({"train": metadata}, 17)
    with pytest.raises(RuntimeError, match="model-input"):
        validate_dataset_input_contract(
            {"train": {**metadata, "modelInputSchema": "old"}}, 17
        )


@pytest.mark.parametrize("model_format", [17, 18])
def test_cli_checkpoint_cooperative_stop_and_exact_resume(
    tmp_path, monkeypatch, model_format
):
    from riichi_analysis_engine import train
    from riichi_analysis_engine.packing import (
        MANIFEST_FORMAT,
        build_pack,
        pack_slots,
        plan_corpus,
        seam_game,
        staged_games,
        write_manifest,
    )
    from riichi_analysis_engine.storage import (
        V16_TRAINING_TARGET_SCHEMA_ID,
        save_chunk_archive,
    )

    converted = converted_fixture()
    stage, packs = tmp_path / "stage", tmp_path / "packs"
    for index in range(2):
        save_chunk_archive(
            stage / f"game-{index:06d}.zip",
            converted.arrays,
            1,
            event_catalog=converted.event_catalog,
            training_target_schema=V16_TRAINING_TARGET_SCHEMA_ID,
        )
    paths = staged_games(stage)
    plan = plan_corpus(paths, 17)
    entries = [
        build_pack(paths, packs, plan, slot, 17, seam_game(plan, slot))[0]
        for slot in pack_slots(plan["length"], 32)
    ]
    write_manifest(
        packs / "manifest.json",
        {
            "format": MANIFEST_FORMAT,
            "seed": 17,
            "samples": int(plan["length"].sum()),
            "sourceGames": 2,
            "chunks": len(plan["length"]),
            "packs": entries,
            **{
                key: entries[0].get(key)
                for key in (
                    "modelInputSchema",
                    "observationChannels",
                    "eventMemorySchema",
                    "trainingTargetSchema",
                )
            },
        },
    )
    monkeypatch.setattr(train, "open_dashboard", lambda _run: None)
    stop = tmp_path / "stop.request"

    def run(name, samples, resume=None):
        arguments = [
            "train",
            "--train",
            str(packs),
            "--validation",
            str(packs),
            "--run",
            str(tmp_path / name),
            "--model-format",
            str(model_format),
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--max-train-samples",
            str(samples),
            "--learning-rate",
            "1e-5",
            "--semantic-width",
            "128",
            "--v15-blocks",
            "1",
            "--v15-feed-forward-width",
            "256",
            "--semantic-decoder-width",
            "256",
            "--stop-file",
            str(stop),
        ]
        if resume:
            arguments.extend(["--resume", str(resume)])
        monkeypatch.setattr(sys, "argv", arguments)
        if stop.exists():
            with pytest.raises(SystemExit) as interrupted:
                train.main()
            assert interrupted.value.code == 130
        else:
            train.main()
        checkpoint = train.resolve_resume_path(tmp_path / name)
        return checkpoint, torch.load(checkpoint, weights_only=True)

    first_path, first = run("first", 1)
    assert first["trainingCursor"]["nextSample"] == 1
    assert first["format"] == f"riichi-analysis-model-v{model_format}"
    stop.touch()
    stopped_path, stopped = run("stopped", 2, first_path)
    assert stopped["trainingCursor"]["nextSample"] == 1
    for key, value in first["model"].items():
        torch.testing.assert_close(value, stopped["model"][key], rtol=0, atol=0)
    stop.unlink()
    _, resumed = run("resumed", 2, stopped_path)
    assert resumed["trainingCursor"]["nextSample"] == 2
    assert resumed["step"] == 2
    assert all(torch.isfinite(value).all() for value in resumed["model"].values())
