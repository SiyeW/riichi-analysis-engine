import hashlib
import importlib.util
import shutil
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
from test_storage import sample_arrays

from riichi_analysis_engine.packing import (
    build_pack,
    pack_slots,
    plan_corpus,
    seam_game,
    staged_games,
    write_plan,
)
from riichi_analysis_engine.storage import (
    read_packed_shard,
    save_chunk_archive,
    write_packed_shard,
)

PACK_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pack_global.py"
PACK_SPEC = importlib.util.spec_from_file_location("pack_global", PACK_SCRIPT)
assert PACK_SPEC is not None and PACK_SPEC.loader is not None
pack_global = importlib.util.module_from_spec(PACK_SPEC)
sys.modules[PACK_SPEC.name] = pack_global
PACK_SPEC.loader.exec_module(pack_global)


@pytest.fixture()
def scratch():
    root = (
        Path(__file__).resolve().parents[1]
        / "runs"
        / "test-pack-resume"
        / uuid.uuid4().hex
    )
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def stage_corpus(root: Path) -> Path:
    stage = root / "stage"
    for game in range(6):
        arrays = sample_arrays(32, kyoku=game)
        arrays["event_index"] = np.arange(32, dtype=np.int32) + game * 1000
        save_chunk_archive(stage / f"game-{game:06d}.zip", arrays, 8)
    return stage


def test_interrupted_pack_can_verify_and_reuse_an_existing_slot(scratch: Path) -> None:
    stage = stage_corpus(scratch)
    output = scratch / "packs"
    games = staged_games(stage)
    plan = plan_corpus(games, 314159)
    slots = pack_slots(plan["length"], 128)
    write_plan(output / "plan.npz", plan)
    expected, _ = build_pack(
        games, output, plan, slots[0], 314159, seam_game(plan, slots[0])
    )

    restored = pack_global._existing_pack_meta(output, plan, slots[0], 314159, None)

    assert restored == expected
    assert pack_global._existing_pack_meta(output, plan, slots[1], 314159, None) is None


def test_resume_rejects_a_pack_that_does_not_match_its_plan_slot(scratch: Path) -> None:
    stage = stage_corpus(scratch)
    output = scratch / "packs"
    games = staged_games(stage)
    plan = plan_corpus(games, 314159)
    slot = pack_slots(plan["length"], 128)[0]
    build_pack(games, output, plan, slot, 314159, seam_game(plan, slot))
    path = output / "pack-00000.npz"
    packed = read_packed_shard(path)
    packed["pack_index"] = np.ones_like(packed["pack_index"])
    write_packed_shard(path, packed)

    with pytest.raises(ValueError, match="declares pack indices"):
        pack_global._existing_pack_meta(output, plan, slot, 314159, None)


def test_resume_completes_the_missing_slots_without_rewriting_verified_packs(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = stage_corpus(scratch)
    output = scratch / "packs"
    games = staged_games(stage)
    plan = plan_corpus(games, 314159)
    first = pack_slots(plan["length"], 64)[0]
    write_plan(output / "plan.npz", plan)
    build_pack(games, output, plan, first, 314159, seam_game(plan, first))
    first_pack = output / "pack-00000.npz"
    before = hashlib.sha256(first_pack.read_bytes()).hexdigest()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(PACK_SCRIPT),
            "--stage",
            str(stage),
            "--output",
            str(output),
            "--seed",
            "314159",
            "--pack-samples",
            "64",
            "--workers",
            "1",
            "--resume",
        ],
    )

    pack_global.main()

    after = hashlib.sha256(first_pack.read_bytes()).hexdigest()
    manifest = (output / "manifest.json").read_text(encoding="utf-8")
    assert before == after
    assert '"verified": true' in manifest
    assert len(list(output.glob("pack-*.npz"))) == len(pack_slots(plan["length"], 64))
