from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from test_dataset import pack_directory

from riichi_analysis_engine.dataset import PackDataset
from riichi_analysis_engine.observation_layout import JIKAZE_CHANNEL, WIND_TILE_START
from riichi_analysis_engine.training_schema import validate_v8_training_batch


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
