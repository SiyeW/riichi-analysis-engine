from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from test_dataset import pack_directory

from riichi_analysis_engine.dataset import PackDataset
from riichi_analysis_engine.model_input import (
    SHARED_MODEL_INPUT_CHANNELS,
    SHARED_MODEL_INPUT_SCHEMA_ID,
)
from riichi_analysis_engine.observation_layout import JIKAZE_CHANNEL, WIND_TILE_START
from riichi_analysis_engine.semantic_input import (
    EVENT_ACTOR,
    EVENT_MEMORY_SCHEMA_ID,
    EVENT_TILE,
    EVENT_TYPE,
    PUBLIC_EVENT_TYPE_TO_ID,
)
from riichi_analysis_engine.train import (
    dataset_metadata,
    validate_dataset_input_contract,
)
from riichi_analysis_engine.training_schema import (
    validate_semantic_training_batch,
    validate_v8_training_batch,
)


@pytest.fixture()
def scratch() -> Iterator[Path]:
    root = (
        Path(__file__).resolve().parents[1]
        / "runs"
        / "test-training-schema"
        / uuid.uuid4().hex
    )
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_v8_training_schema_accepts_a_converted_contract_batch(scratch) -> None:
    packs = pack_directory(scratch, training_targets=True)
    batch = next(iter(PackDataset(packs, batch_size=4)))

    assert validate_v8_training_batch(batch) == {"samples": 4, "seatWinds": 1}


def test_v8_training_schema_rejects_missing_seat_wind(scratch) -> None:
    packs = pack_directory(scratch, training_targets=True)
    batch = next(iter(PackDataset(packs, batch_size=4)))
    batch["obs"][:, JIKAZE_CHANNEL, WIND_TILE_START : WIND_TILE_START + 4] = 0

    try:
        validate_v8_training_batch(batch)
    except ValueError as error:
        assert "seat wind" in str(error)
    else:
        raise AssertionError("missing seat wind was accepted")


def test_v10_training_schema_requires_the_analysis_row_mask(scratch) -> None:
    packs = pack_directory(scratch, training_targets=True, model_format=9)
    batch = next(iter(PackDataset(packs, batch_size=4)))

    with pytest.raises(ValueError, match="analysis_active"):
        validate_v8_training_batch(batch, require_analysis_active=True)

    batch["analysis_active"] = batch["policy"] >= 0
    result = validate_v8_training_batch(batch, require_analysis_active=True)
    assert result["analysisSamples"] == int(batch["analysis_active"].sum())


def test_v12_training_schema_validates_masked_event_memory(scratch) -> None:
    packs = pack_directory(
        scratch,
        training_targets=True,
        model_format=11,
        semantic_history=True,
    )
    batch = next(iter(PackDataset(packs, batch_size=4)))

    result = validate_semantic_training_batch(batch)
    metadata = dataset_metadata(packs)
    validate_dataset_input_contract({"validation": metadata}, 12)

    assert result["samples"] == 4
    assert result["eventTokens"] == int(batch["event_mask"].sum())
    assert 1 <= result["maxEventHistory"] <= 3

    opponent_draw = (
        (batch["event_tokens"][..., EVENT_TYPE] == PUBLIC_EVENT_TYPE_TO_ID["tsumo"])
        & (batch["event_tokens"][..., EVENT_ACTOR] != 1)
        & batch["event_mask"]
    )
    if opponent_draw.any():
        batch["event_tokens"][..., EVENT_TILE][opponent_draw] = 2
        with pytest.raises(ValueError, match="leaked"):
            validate_semantic_training_batch(batch)


def test_v13_dataset_contract_requires_engine_owned_rule_context() -> None:
    metadata = {
        "modelInputSchema": SHARED_MODEL_INPUT_SCHEMA_ID,
        "observationChannels": SHARED_MODEL_INPUT_CHANNELS,
        "eventMemorySchema": EVENT_MEMORY_SCHEMA_ID,
    }

    validate_dataset_input_contract({"train": metadata}, 13)

    with pytest.raises(RuntimeError, match="model-input contract"):
        validate_dataset_input_contract(
            {"train": {**metadata, "modelInputSchema": "mortal-observation"}}, 13
        )
