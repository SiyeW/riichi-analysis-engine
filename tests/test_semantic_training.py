import shutil
import uuid
from pathlib import Path

import pytest
import torch
from test_dataset import pack_directory

from riichi_analysis_engine.architecture import SemanticModelArchitecture
from riichi_analysis_engine.dataset import PackDataset
from riichi_analysis_engine.losses import (
    LOSS_TERMS_V8,
    LearnedUncertaintyBalancer,
    multitask_loss,
)
from riichi_analysis_engine.model import RiichiAnalysisModel
from riichi_analysis_engine.train import forward_batch
from riichi_analysis_engine.training_schema import validate_semantic_training_batch


@pytest.fixture()
def scratch() -> Path:
    root = (
        Path(__file__).resolve().parents[1]
        / "runs"
        / "test-semantic-training"
        / uuid.uuid4().hex
    )
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("backbone", ["cnn", "transformer"])
def test_v12_real_losses_complete_one_cpu_optimizer_step(
    scratch: Path, backbone: str
) -> None:
    packs = pack_directory(
        scratch,
        games=2,
        chunks=1,
        samples=2,
        pack_samples=4,
        training_targets=True,
        model_format=11,
        semantic_history=True,
    )
    batch = next(iter(PackDataset(packs, batch_size=4)))
    validate_semantic_training_batch(batch)
    architecture = SemanticModelArchitecture(
        backbone=backbone,
        width=16,
        stem_width=24,
        event_width=12,
        backbone_blocks=1,
        event_blocks=1,
        decoder_width=20,
        attention_heads=4,
    )
    model = RiichiAnalysisModel(format_version=12, architecture=architecture)
    balancer = LearnedUncertaintyBalancer(LOSS_TERMS_V8)
    optimizer = torch.optim.AdamW(
        [*model.parameters(), *balancer.parameters()], lr=1e-4
    )

    outputs = forward_batch(model, batch)
    total, losses, active, _weights = multitask_loss(outputs, batch, balancer)
    total.backward()
    optimizer.step()

    assert torch.isfinite(total)
    assert set(losses) == set(LOSS_TERMS_V8)
    assert any(active.values())
