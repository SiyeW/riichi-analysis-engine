"""Optional evaluation-only group sums for paired, clustered comparisons."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

GROUP_METRICS = frozenset(
    {
        "policyNll",
        "policyAccuracy",
        "shantenNll",
        "furitenBrier",
        "dealInEligibleConditionalPositiveNll",
        "dealInEligibleConditionalNegativeNll",
        "hiddenCountNll",
        "doraNll",
        "scoreNll",
        "doraTailMae",
    }
)


class ValidationGroups:
    def __init__(self) -> None:
        self.groups: dict[int, dict[str, dict[str, float | int]]] = {}

    def add(
        self,
        name: str,
        values: torch.Tensor,
        games: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        if name not in GROUP_METRICS:
            return
        ids = games.detach().cpu().numpy().reshape(-1)
        if not len(ids):
            if values.numel():
                raise ValueError("validation values have no source-game groups")
            return
        if values.numel() % len(ids):
            raise ValueError("validation values do not preserve the row axis")
        ids = np.repeat(ids, values.numel() // len(ids))
        numbers = values.detach().float().cpu().numpy().reshape(-1)
        if mask is not None:
            selected = mask.detach().bool().cpu().numpy().reshape(-1)
            ids = ids[selected]
            numbers = numbers[selected]
        if not np.isfinite(numbers).all():
            raise ValueError(f"nonfinite grouped validation metric: {name}")
        unique, inverse = np.unique(ids, return_inverse=True)
        sums = np.bincount(inverse, weights=numbers.astype(np.float64))
        counts = np.bincount(inverse)
        for game, total, count in zip(unique, sums, counts, strict=True):
            row = self.groups.setdefault(int(game), {}).setdefault(
                name, {"sum": 0.0, "count": 0}
            )
            row["sum"] += float(total)
            row["count"] += int(count)

    def write(
        self, path: Path, validation_samples: int, manifest_sha256: str | None = None
    ) -> None:
        payload = {
            "format": "validation-game-sums-v1",
            "validationSamples": validation_samples,
            "manifestSha256": manifest_sha256,
            "groups": [
                {"sourceGame": game, "metrics": metrics}
                for game, metrics in sorted(self.groups.items())
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
