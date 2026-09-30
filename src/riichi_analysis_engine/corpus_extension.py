"""Prove that an expanded corpus retains an immutable, ordered pack prefix."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .dataset import read_manifest
from .packing import read_plan


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def extension_proof(prefix: Path) -> dict[str, object]:
    manifest = read_manifest(prefix)
    if not manifest.get("audit", {}).get("verified"):
        raise ValueError("extension prefix has not passed its pack audit")
    return {
        "type": "immutable-pack-prefix-v1",
        "manifestSha256": file_sha256(prefix / "manifest.json"),
        "planSha256": file_sha256(prefix / "plan.npz"),
        "samples": manifest["samples"],
        "packSha256": [
            file_sha256(prefix / entry["pack"]) for entry in manifest["packs"]
        ],
    }


def validate_corpus_extension(
    saved: dict[str, object], current: dict[str, object], next_sample: int
) -> dict[str, object]:
    """Reject replacements, reordered prefixes, changed bytes and replayed games."""
    prefix = Path(str(saved["path"]))
    root = Path(str(current["path"]))
    old = read_manifest(prefix)
    new = read_manifest(root)
    proof = new.get("appendPrefix")
    if not isinstance(proof, dict) or proof.get("type") != "immutable-pack-prefix-v1":
        raise RuntimeError("new corpus has no immutable prefix proof")
    if file_sha256(root / "manifest.json") != current.get("manifestSha256"):
        raise RuntimeError("new corpus manifest changed during resume")
    if file_sha256(prefix / "manifest.json") != saved.get("manifestSha256"):
        raise RuntimeError("original prefix manifest changed")
    if proof != extension_proof(prefix):
        raise RuntimeError("original prefix bytes differ from the extension proof")
    if not new.get("audit", {}).get("verified"):
        raise RuntimeError("expanded corpus has not passed its pack audit")
    for key in (
        "format",
        "seed",
        "modelInputSchema",
        "observationChannels",
        "eventMemorySchema",
        "trainingTargetSchema",
    ):
        if old.get(key) != new.get(key):
            raise RuntimeError(f"expanded corpus changed {key}")
    old_entries, new_entries = old["packs"], new["packs"]
    if len(new_entries) <= len(old_entries):
        raise RuntimeError("expanded corpus does not append packs")
    for before, after in zip(old_entries, new_entries, strict=False):
        if {k: v for k, v in before.items() if k != "pack"} != {
            k: v for k, v in after.items() if k != "pack"
        } or (prefix / before["pack"]).resolve() != (root / after["pack"]).resolve():
            raise RuntimeError("expanded corpus reordered or replaced prefix packs")
    a, b = read_plan(prefix / "plan.npz"), read_plan(root / "plan.npz")
    boundary = len(a["length"])
    for field in ("source_game", "source_member", "length"):
        if not np.array_equal(a[field], b[field][:boundary]):
            raise RuntimeError("expanded corpus changed prefix plan order")
    if np.intersect1d(a["source_game"], b["source_game"][boundary:]).size:
        raise RuntimeError("expanded corpus repeats prefix source games")
    if int(a["length"].sum()) != int(old["samples"]):
        raise RuntimeError("prefix sample count disagrees with its plan")
    if int(b["length"].sum()) != int(new["samples"]):
        raise RuntimeError("expanded sample count disagrees with its plan")
    ids = np.unique(b["source_game"])
    if not np.array_equal(ids, np.arange(int(new["sourceGames"]))):
        raise RuntimeError("expanded corpus has missing source game identities")
    if not 0 <= next_sample < int(old["samples"]) < int(new["samples"]):
        raise RuntimeError("extension cursor is outside the unchanged prefix")
    return {
        "type": proof["type"],
        "atSample": next_sample,
        "parentManifestSha256": saved["manifestSha256"],
        "manifestSha256": current["manifestSha256"],
        "prefixSamples": old["samples"],
        "samples": new["samples"],
    }


def extend_tail_schedule(
    requested: dict[str, object], saved: dict[str, object], next_sample: int
) -> dict[str, object]:
    """Freeze a newly known full-pass tail without changing the prefix rates."""
    previous = dict(saved)
    previous.pop("corpusExtension", None)
    if (
        previous.get("tailDecaySamples") != 0
        or previous.get("warmupSteps")
        or previous.get("cooldownSteps")
    ):
        raise RuntimeError("corpus extension requires an undecayed plateau prefix")
    comparable = dict(requested)
    limit = int(comparable.pop("sampleLimit"))
    tail = int(comparable["tailDecaySamples"])
    comparable["tailDecaySamples"] = 0
    if comparable != previous or not 0 < tail < limit - next_sample:
        raise RuntimeError(
            "corpus extension changed rates or places the cursor in the tail"
        )
    return {
        **requested,
        "corpusExtension": {"atSample": next_sample, "previousSchedule": saved},
    }
