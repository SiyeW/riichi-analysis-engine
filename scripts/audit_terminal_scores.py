"""Audit terminal-score conservation in a globally mixed pack directory."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from riichi_analysis_engine.dataset import audit_terminal_score_arrays, read_manifest
from riichi_analysis_engine.storage import read_packed_shard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_manifest(args.packs)
    score_sums: Counter[str] = Counter()
    deficits: Counter[str] = Counter()
    labeled_samples = 0
    deficient_samples = 0
    legacy_packs = 0
    affected_games: set[int] = set()

    for entry in manifest["packs"]:
        arrays = read_packed_shard(args.packs / str(entry["pack"]))
        summary = audit_terminal_score_arrays(arrays)
        labeled_samples += int(summary["labeledSamples"])
        deficient_samples += int(summary["deficientSamples"])
        legacy_packs += int(bool(summary["legacyAssumedTotal"]))
        score_sums.update(summary["rawScoreSums"])
        deficits.update(summary["deficitPoints"])

        if int(summary["deficientSamples"]):
            scores = arrays["match_score"]
            total = arrays.get("system_total")
            expected = 100_000 if total is None else np.asarray(total)
            sums = scores.sum(axis=-1)
            mask = (sums > 0) & (sums < expected)
            affected_games.update(int(value) for value in arrays["source_game"][mask])

    report = {
        "format": "riichi-analysis-terminal-score-audit-v1",
        "packs": len(manifest["packs"]),
        "legacyPacks": legacy_packs,
        "labeledSamples": labeled_samples,
        "deficientSamples": deficient_samples,
        "affectedSourceGames": len(affected_games),
        "rawScoreSums": dict(sorted(score_sums.items(), key=lambda item: int(item[0]))),
        "deficitPoints": dict(sorted(deficits.items(), key=lambda item: int(item[0]))),
        "loaderRepairExpected": deficient_samples > 0,
        "verified": True,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
