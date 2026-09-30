from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from test_dataset import pack_directory
from test_pack_resume import pack_global
from test_packing import merge_packs
from test_training_single_pass import run_training

from riichi_analysis_engine.corpus_extension import (
    extend_tail_schedule,
    validate_corpus_extension,
)
from riichi_analysis_engine.dataset import PackDataset
from riichi_analysis_engine.packing import read_plan, write_plan
from riichi_analysis_engine.train import (
    dataset_metadata,
    resolve_learning_rate_schedule,
    resolve_resume_path,
)


@pytest.fixture()
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pack_directory(
        tmp_path, games=6, chunks=2, samples=16, pack_samples=192, training_targets=True
    )
    for name, options in (
        ("prefix", ["--limit-games", "3"]),
        ("suffix", ["--start-game", "3"]),
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "pack_global",
                "--stage",
                str(tmp_path / "stage"),
                "--output",
                str(tmp_path / name),
                "--pack-samples",
                "96",
                *options,
            ],
        )
        pack_global.main()
    merged = tmp_path / "merged"
    merged.mkdir()
    merge_packs.merge(
        merged,
        [str(tmp_path / "prefix"), str(tmp_path / "suffix")],
        expected_source_games=6,
        append_prefix=True,
    )
    return tmp_path / "prefix", merged


def test_extension_preserves_exact_prefix_and_unread_sample_order(corpus):
    prefix, merged = corpus
    old, new = dataset_metadata(prefix), dataset_metadata(merged)
    proof = validate_corpus_extension(old, new, 80)
    assert proof["prefixSamples"] == 96
    assert proof["samples"] == 192

    def tags(root, start):
        return torch.cat(
            [
                b["sample_tag"]
                for b in PackDataset(root, batch_size=8, start_sample=start)
            ]
        )

    assert torch.equal(tags(prefix, 80), tags(merged, 80)[:16])
    assert len(torch.unique(tags(merged, 0))) == 192


@pytest.mark.parametrize("mutation", ["order", "bytes", "repeat", "unverified"])
def test_extension_rejects_invalid_proof(corpus, mutation):
    prefix, merged = corpus
    old = dataset_metadata(prefix)
    path = merged / "manifest.json"
    manifest = json.loads(path.read_text())
    if mutation == "order":
        manifest["packs"][0], manifest["packs"][1] = (
            manifest["packs"][1],
            manifest["packs"][0],
        )
    elif mutation == "bytes":
        pack = prefix / "pack-00000.npz"
        with pack.open("ab") as handle:
            handle.write(b"changed")
    elif mutation == "repeat":
        plan = read_plan(merged / "plan.npz")
        boundary = len(read_plan(prefix / "plan.npz")["length"])
        plan["source_game"][boundary] = plan["source_game"][0]
        write_plan(merged / "plan.npz", plan)
    else:
        manifest["audit"]["verified"] = False
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError):
        validate_corpus_extension(old, dataset_metadata(merged), 80)


def test_resume_extension_keeps_optimizer_cursor_and_freezes_tail(
    corpus, tmp_path, monkeypatch
):
    prefix, merged = corpus
    first = run_training(monkeypatch, prefix, tmp_path / "first", max_samples=80)
    old = torch.load(first, weights_only=True)
    with pytest.raises(RuntimeError, match="different train manifest"):
        run_training(
            monkeypatch,
            merged,
            tmp_path / "refused",
            max_samples=192,
            validation_packs=prefix,
            resume=first,
        )
    # An already-present cooperative request saves without consuming a sample.
    stop = tmp_path / "stop.request"
    stop.touch()
    with pytest.raises(SystemExit) as stopped:
        run_training(
            monkeypatch,
            merged,
            tmp_path / "stopped",
            max_samples=192,
            validation_packs=prefix,
            resume=first,
            allow_corpus_extension=True,
            tail_decay_samples=32,
            stop_file=stop,
        )
    assert stopped.value.code == 130
    interrupted = resolve_resume_path(tmp_path / "stopped")
    state = torch.load(interrupted, weights_only=True)
    assert state["samplesSeen"] == 80
    assert state["step"] == old["step"]
    for name, value in old["model"].items():
        assert torch.equal(value, state["model"][name])
    for key, values in old["optimizer"]["state"].items():
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, state["optimizer"]["state"][key][name])
    final = run_training(
        monkeypatch,
        merged,
        tmp_path / "continued",
        max_samples=192,
        validation_packs=prefix,
        resume=interrupted,
        tail_decay_samples=32,
    )
    state = torch.load(final, weights_only=True)
    assert state["samplesSeen"] == 192
    assert state["step"] == 24
    assert state["trainingCursor"]["complete"] is True
    assert state["datasets"]["train"]["corpusExtensions"][0]["atSample"] == 80
    assert state["learningRateSchedule"]["corpusExtension"]["atSample"] == 80
    assert state["optimizer"]["param_groups"][0]["lr"] == pytest.approx(2e-5)


def test_extension_schedule_cannot_change_rates_or_an_existing_tail():
    old = {
        "warmupSteps": 0,
        "cooldownSteps": 0,
        "learningRate": 1e-5,
        "peakLearningRate": 1e-5,
        "lossBalanceLearningRate": 1e-3,
        "tailDecaySamples": 0,
        "tailLearningRateFactor": 0.1,
    }
    requested = {**old, "sampleLimit": 200, "tailDecaySamples": 20}
    extended = extend_tail_schedule(requested, old, 80)
    assert (
        resolve_learning_rate_schedule(
            requested,
            extended,
            next_sample=100,
            allow_transition=False,
            validate_only=False,
        )
        == extended
    )
    for invalid in (
        {**requested, "learningRate": 2e-5},
        {**requested, "tailDecaySamples": 150},
    ):
        with pytest.raises(RuntimeError):
            extend_tail_schedule(invalid, old, 80)
    with pytest.raises(RuntimeError):
        extend_tail_schedule(requested, {**old, "tailDecaySamples": 10}, 80)
