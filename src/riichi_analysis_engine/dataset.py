"""Read a globally mixed pack directory in the order it was written.

Packing decides the training order once, and this loader follows it: one pass
over the packs in manifest order, which is exactly one pass over the corpus
with neighbouring samples coming from different games. Nothing here shuffles,
buffers or reorders samples, because doing any of that would only undo work
the packer already did and would make the training order depend on the loader
configuration.

Memory does not grow with the pack size. A pack holds tens of thousands of
samples in its packed form, but observations are only expanded for the batch
being yielded, so a larger pack costs file size rather than memory.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .packing import SUPPORTED_MANIFEST_FORMATS
from .storage import read_packed_shard, slice_packed, unpack_shard_arrays

# Provenance the packer records alongside every sample. It stays out of the
# batches: the model sees observations and targets, nothing else.
METADATA_FIELDS = frozenset(
    {
        "perspective",
        "event_index",
        "source_game",
        "pack_index",
        "kyoku_index",
        "system_total",
    }
)


def read_manifest(root: str | Path) -> dict[str, object]:
    """Read and check the manifest that defines the training order."""

    path = Path(root) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{root} holds no manifest.json, so it is not a pack directory"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") not in SUPPORTED_MANIFEST_FORMATS:
        raise ValueError(f"{path} declares an unsupported manifest format")
    packs = manifest.get("packs")
    if not isinstance(packs, list) or not packs:
        raise ValueError(f"{path} lists no packs")
    return manifest


def _settle_legacy_terminal_scores(arrays: dict[str, np.ndarray]) -> None:
    """Repair old packs whose terminal MJAI event left riichi sticks unassigned.

    New conversion writes ``system_total`` explicitly. Historical Tenhou packs
    predate that field and always use four 25,000-point accounts, so 100,000 is
    their compatibility total. The correction is deterministic from the final
    scores and the stored absolute perspective; it does not inspect a target
    that would be unavailable at inference time.
    """

    scores = arrays.get("match_score")
    if scores is None or scores.ndim != 2 or scores.shape[1] != 4:
        return
    totals = arrays.get("system_total")
    if totals is None:
        totals = np.full(len(scores), 100_000, dtype=np.int32)
    totals = np.asarray(totals, dtype=np.int64).reshape(-1)
    if len(totals) != len(scores):
        raise ValueError("system-total labels do not match terminal-score labels")
    sums = scores.astype(np.int64, copy=False).sum(axis=-1)
    deficits = totals - sums
    repair = (sums > 0) & (deficits > 0)
    if (deficits[sums > 0] < 0).any() or (deficits[repair] % 1_000 != 0).any():
        raise ValueError("terminal scores are inconsistent with the system total")
    if not repair.any():
        return
    perspectives = arrays.get("perspective")
    if perspectives is None or len(perspectives) != len(scores):
        raise ValueError("terminal-score repair requires each sample's perspective")
    for row in np.flatnonzero(repair):
        maximum = int(scores[row].max())
        tied = np.flatnonzero(scores[row] == maximum)
        winner = min(
            tied, key=lambda relative: (int(perspectives[row]) + int(relative)) % 4
        )
        scores[row, winner] += deficits[row]


def audit_terminal_score_arrays(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    """Summarize terminal-score conservation without expanding observations."""

    scores = arrays.get("match_score")
    if scores is None or scores.ndim != 2 or scores.shape[1] != 4:
        raise ValueError("pack has no four-player terminal-score labels")
    totals = arrays.get("system_total")
    legacy = totals is None
    if totals is None:
        totals = np.full(len(scores), 100_000, dtype=np.int64)
    totals = np.asarray(totals, dtype=np.int64).reshape(-1)
    if len(totals) != len(scores):
        raise ValueError("system-total labels do not match terminal-score labels")

    sums = scores.astype(np.int64, copy=False).sum(axis=-1)
    labeled = sums > 0
    deficits = totals - sums
    invalid = labeled & ((deficits < 0) | (deficits % 1_000 != 0))
    if invalid.any():
        raise ValueError("terminal scores are inconsistent with the system total")
    deficient = labeled & (deficits > 0)
    sources = arrays.get("source_game")
    affected_games = (
        len(np.unique(sources[deficient]))
        if sources is not None and len(sources) == len(scores)
        else None
    )

    def distribution(values: np.ndarray) -> dict[str, int]:
        unique, counts = np.unique(values, return_counts=True)
        return {
            str(int(value)): int(count)
            for value, count in zip(unique, counts, strict=True)
        }

    return {
        "samples": len(scores),
        "labeledSamples": int(labeled.sum()),
        "deficientSamples": int(deficient.sum()),
        "affectedSourceGames": affected_games,
        "legacyAssumedTotal": legacy,
        "rawScoreSums": distribution(sums[labeled]),
        "deficitPoints": distribution(deficits[deficient]),
    }


class PackDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Iterate a pack directory once, in manifest order."""

    def __init__(
        self,
        root: str | Path,
        *,
        batch_size: int,
        max_samples: int = 0,
        start_sample: int = 0,
    ) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if max_samples < 0:
            raise ValueError("maximum samples must be non-negative")
        if start_sample < 0:
            raise ValueError("starting sample must be non-negative")
        self.root = Path(root)
        self.batch_size = batch_size
        self.max_samples = max_samples
        # The cursor is an absolute offset in the authoritative sample stream,
        # not a batch count. It remains exact when a prior budget ended with a
        # short batch.
        self.start_sample = start_sample
        self.manifest = read_manifest(self.root)
        self.entries = [entry for entry in self.manifest["packs"]]
        self.packs = [self.root / str(entry["pack"]) for entry in self.entries]
        missing = [path for path in self.packs if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{missing[0]} is listed in the manifest but missing"
            )
        self.samples = int(self.manifest["samples"])
        if self.start_sample > self.samples:
            raise ValueError("starting sample lies beyond the end of the corpus")

    def selected_packs(self, worker_id: int = 0, worker_count: int = 1) -> list[Path]:
        """The packs one worker reads. The workers together cover every pack once."""

        if not 0 <= worker_id < worker_count:
            raise ValueError("worker id must be inside the worker count")
        return self.packs[worker_id::worker_count]

    @staticmethod
    def _sample_count(packed: dict[str, np.ndarray]) -> int:
        return len(packed["obs_offsets"]) - 1

    def _dense(
        self, packed: dict[str, np.ndarray], start: int, stop: int
    ) -> dict[str, np.ndarray]:
        arrays = unpack_shard_arrays(slice_packed(packed, start, stop))
        _settle_legacy_terminal_scores(arrays)
        return {
            name: value for name, value in arrays.items() if name not in METADATA_FIELDS
        }

    @staticmethod
    def _merge(
        parts: list[dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        return {
            name: np.concatenate([part[name] for part in parts], axis=0)
            for name in parts[0]
        }

    @staticmethod
    def _torch(arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {
            name: torch.from_numpy(np.ascontiguousarray(value))
            for name, value in arrays.items()
        }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        if worker is None:
            packs = self.selected_packs()
        else:
            if self.start_sample:
                raise ValueError(
                    "starting from a sample cursor requires a single worker"
                )
            packs = self.selected_packs(worker.id, worker.num_workers)
        accepted = 0
        skip = self.start_sample
        # A pack rarely divides into whole batches, so the samples left over at
        # the end of one pack are carried into the next one instead of dropped.
        pending: dict[str, np.ndarray] | None = None
        for entry, path in zip(self.entries, packs, strict=True):
            count = int(entry["samples"])
            if skip >= count:
                skip -= count
                continue
            packed = read_packed_shard(path)
            if self._sample_count(packed) != count:
                raise ValueError(
                    f"{path.name} holds a different sample count than the manifest"
                )
            start = skip
            skip = 0
            if pending is not None:
                take = min(self.batch_size - len(next(iter(pending.values()))), count)
                if take:
                    pending = self._merge([pending, self._dense(packed, 0, take)])
                    start = take
                if len(next(iter(pending.values()))) == self.batch_size:
                    if self.max_samples:
                        remaining = self.max_samples - accepted
                        if remaining <= 0:
                            return
                        if remaining < self.batch_size:
                            yield self._torch(
                                {
                                    name: value[:remaining]
                                    for name, value in pending.items()
                                }
                            )
                            return
                    accepted += len(next(iter(pending.values())))
                    yield self._torch(pending)
                    pending = None
            whole = (count - start) // self.batch_size * self.batch_size
            for offset in range(start, start + whole, self.batch_size):
                if self.max_samples:
                    remaining = self.max_samples - accepted
                    if remaining <= 0:
                        return
                    if remaining < self.batch_size:
                        yield self._torch(
                            self._dense(packed, offset, offset + remaining)
                        )
                        return
                accepted += self.batch_size
                yield self._torch(self._dense(packed, offset, offset + self.batch_size))
            if start + whole < count:
                pending = self._dense(packed, start + whole, count)
            if self.max_samples and accepted >= self.max_samples:
                return
        if pending is not None:
            if self.max_samples:
                remaining = self.max_samples - accepted
                if remaining <= 0:
                    return
                pending = {name: value[:remaining] for name, value in pending.items()}
            yield self._torch(pending)
