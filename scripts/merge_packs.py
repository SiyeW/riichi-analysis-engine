"""Merge the segments of a corpus that was built in pieces.

A full corpus may not fit on the build disk twice: the staged games and
the packs together need more room than the machine has. The corpus is therefore
converted and packed in segments, and each segment's staged games are deleted as
soon as that segment is packed.

Training still needs one order over one corpus, so this command joins the
segments through one manifest and one concatenated plan.  The packs stay in
their segment directories: validating every input before writing the merged
metadata makes a failed merge non-destructive and keeps restarts cheap.

    python scripts/merge_packs.py --root data/processed/v4/packs-train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from riichi_analysis_engine.packing import (
    LEGACY_MANIFEST_FORMATS,
    MANIFEST_FORMAT,
    SUPPORTED_MANIFEST_FORMATS,
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
    parser.add_argument(
        "--expected-source-games",
        type=int,
        default=0,
        help="require this many source games before writing the merged metadata",
    )
    return parser.parse_args()


def segment_names(root: Path) -> list[str]:
    """Every subdirectory of the root that holds a packed segment, by name."""

    return sorted(path.name for path in root.iterdir() if (path / "manifest.json").exists())


def merge(
    root: Path, names: list[str], expected_source_games: int = 0
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    plans: list[dict[str, np.ndarray]] = []
    seen_games: set[int] = set()
    shared: dict[str, object] = {}
    for name in names:
        segment = root / name
        manifest = json.loads((segment / "manifest.json").read_text(encoding="utf-8"))
        manifest_format = manifest.get("format")
        if manifest_format not in SUPPORTED_MANIFEST_FORMATS:
            raise ValueError(f"{segment} declares an unsupported manifest format")
        for key in (
            "seed",
            "packSamples",
            "modelInputSchema",
            "observationChannels",
            "eventMemorySchema",
            "trainingTargetSchema",
        ):
            if key in shared and manifest.get(key) != shared[key]:
                raise ValueError(f"segment {name} was packed with a different {key}")
            shared[key] = manifest.get(key)

        plan = read_plan(segment / "plan.npz")
        identifiers = np.unique(plan["source_game"])
        offset = int(manifest.get("gameOffset", 0))
        declared_games = int(manifest["sourceGames"])
        if manifest_format in LEGACY_MANIFEST_FORMATS:
            # Version 1 stored the exclusive upper bound rather than the count.
            declared_games -= offset
        if len(identifiers) != declared_games:
            raise ValueError(f"segment {name} holds {len(identifiers)} games, its manifest says more")
        overlap = seen_games.intersection(int(value) for value in identifiers)
        if overlap:
            raise ValueError(f"segment {name} repeats source game {min(overlap)}")
        seen_games.update(int(value) for value in identifiers)

        for entry in manifest["packs"]:
            source = segment / str(entry["pack"])
            if not source.is_file():
                raise FileNotFoundError(source)
            entries.append({**entry, "pack": source.relative_to(root).as_posix()})
        plans.append(plan)

    if not entries:
        raise ValueError("no packed segments were found")
    ordered_games = sorted(seen_games)
    expected_games = list(range(len(ordered_games)))
    if ordered_games != expected_games:
        missing = next(
            (game for game, actual in enumerate(ordered_games) if game != actual),
            len(ordered_games),
        )
        raise ValueError(f"the merged corpus is missing source game {missing}")
    if expected_source_games and len(ordered_games) != expected_source_games:
        raise ValueError(
            f"the merged corpus holds {len(ordered_games)} source games, "
            f"expected {expected_source_games}"
        )
    merged = {
        name: np.concatenate([plan[name] for plan in plans])
        for name in ("source_game", "source_member", "length")
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
        "sourceGames": len(seen_games),
        "segments": names,
        "packs": entries,
        "modelInputSchema": shared.get("modelInputSchema"),
        "observationChannels": shared.get("observationChannels"),
        "eventMemorySchema": shared.get("eventMemorySchema"),
        "trainingTargetSchema": shared.get("trainingTargetSchema"),
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
    report = merge(arguments.root, names, arguments.expected_source_games)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
