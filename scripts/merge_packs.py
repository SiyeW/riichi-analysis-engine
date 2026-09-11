"""Merge the segments of a corpus that was built in pieces.

A full 2025 corpus does not fit on the build disk twice: the staged games and
the packs together need more room than the machine has. The corpus is therefore
converted and packed in segments, and each segment's staged games are deleted as
soon as that segment is packed.

Training still needs one order over one corpus, so this command joins the
segments back together: the packs move into the pack directory under one running
index, the plans concatenate (their source-game ids were numbered globally when
they were written), and the audit runs on the merged result.

    python scripts/merge_packs.py --root data/processed/v4/packs-train-2025
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from riichi_analysis_engine.packing import (
    MANIFEST_FORMAT,
    audit_packs,
    read_plan,
    write_manifest,
    write_plan,
)


def build_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="pack directory holding the segments")
    parser.add_argument(
        "--segments",
        nargs="*",
        default=[],
        help="segment directory names in build order (default: every packed subdirectory, by name)",
    )
    return parser.parse_args()


def segment_names(root: Path) -> list[str]:
    """Every subdirectory of the root that holds a packed segment, by name."""

    return sorted(path.name for path in root.iterdir() if (path / "manifest.json").exists())


def merge(root: Path, names: list[str]) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    plans: list[dict[str, np.ndarray]] = []
    games = 0
    shared: dict[str, object] = {}
    for name in names:
        segment = root / name
        manifest = json.loads((segment / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != MANIFEST_FORMAT:
            raise ValueError(f"{segment} declares an unsupported manifest format")
        for key in ("seed", "packSamples"):
            if key in shared and manifest.get(key) != shared[key]:
                raise ValueError(f"segment {name} was packed with a different {key}")
            shared[key] = manifest.get(key)

        plan = read_plan(segment / "plan.npz")
        identifiers = np.unique(plan["source_game"])
        offset = int(manifest.get("gameOffset", 0))
        if offset != games:
            raise ValueError(f"segment {name} starts at game {offset}, expected {games}")
        if int(identifiers.min()) != games or int(identifiers.max()) != games + len(identifiers) - 1:
            raise ValueError(f"segment {name} does not number its games consecutively")
        if len(identifiers) != int(manifest["sourceGames"]) - offset:
            raise ValueError(f"segment {name} holds {len(identifiers)} games, its manifest says more")

        for entry in manifest["packs"]:
            source = segment / str(entry["pack"])
            destination = root / f"pack-{len(entries):05d}.npz"
            source.replace(destination)
            entries.append({**entry, "pack": destination.name})
        plans.append(plan)
        games += len(identifiers)

    if not entries:
        raise ValueError("no packed segments were found")
    merged = {
        name: np.concatenate([plan[name] for plan in plans])
        for name in sorted(plans[0])
        if name != "game_offset"
    }
    # The merged corpus starts at its first game, whatever the first segment
    # was numbered from.
    merged["game_offset"] = np.asarray(0, dtype=np.uint32)
    write_plan(root / "plan.npz", merged)

    manifest = {
        "format": MANIFEST_FORMAT,
        "stage": " + ".join(names),
        "seed": shared.get("seed"),
        "packSamples": shared.get("packSamples"),
        "samples": int(merged["length"].sum()),
        "chunks": len(merged["length"]),
        "sourceGames": games,
        "segments": names,
        "packs": entries,
    }
    write_manifest(root / "manifest.json", manifest)
    report = audit_packs(root, manifest, merged)
    manifest["audit"] = report
    write_manifest(root / "manifest.json", manifest)
    return report


def main() -> None:
    arguments = build_arguments()
    names = arguments.segments or segment_names(arguments.root)
    if not names:
        raise SystemExit(f"{arguments.root} holds no packed segments")
    report = merge(arguments.root, names)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
