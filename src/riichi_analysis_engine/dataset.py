from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .storage import load_shard


class ShardDataset(IterableDataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        root: str | Path,
        *,
        shuffle: bool,
        seed: int = 20252026,
        max_samples: int = 0,
        batch_size: int = 256,
        shuffle_buffer_samples: int = 1024,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.shuffle = shuffle
        self.seed = seed
        self.max_samples = max_samples
        self.batch_size = batch_size
        if shuffle_buffer_samples < batch_size:
            raise ValueError("shuffle buffer must hold at least one batch")
        self.shuffle_buffer_samples = shuffle_buffer_samples
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _shards(self) -> list[Path]:
        shards = sorted(self.root.glob("*.npz"))
        if not shards:
            raise FileNotFoundError(f"no shards under {self.root}")
        worker = get_worker_info()
        if worker is not None:
            shards = shards[worker.id :: worker.num_workers]
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(shards)
        return shards

    @staticmethod
    def _selected(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
        return {
            name: value[indices]
            for name, value in arrays.items()
            if name not in {"perspective", "event_index"}
        }

    @staticmethod
    def _concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {
            name: np.concatenate([part[name] for part in parts], axis=0)
            for name in parts[0]
        }

    def _batch_stream(
        self, arrays: dict[str, np.ndarray], pending: dict[str, np.ndarray] | None
    ) -> tuple[Iterator[dict[str, torch.Tensor]], dict[str, np.ndarray] | None]:
        """Return full batches and a tail kept for the next mixed chunk."""

        if pending is not None:
            arrays = self._concatenate([pending, arrays])
        sample_count = len(next(iter(arrays.values())))
        stop = sample_count - sample_count % self.batch_size

        def batches() -> Iterator[dict[str, torch.Tensor]]:
            for start in range(0, stop, self.batch_size):
                yield {
                    name: torch.from_numpy(np.ascontiguousarray(value[start : start + self.batch_size]))
                    for name, value in arrays.items()
                }

        tail = (
            {name: np.ascontiguousarray(value[stop:]) for name, value in arrays.items()}
            if stop < sample_count
            else None
        )
        return batches(), tail

    def _iter_sequential(self) -> Iterator[dict[str, torch.Tensor]]:
        accepted = 0
        pending: dict[str, np.ndarray] | None = None
        for path in self._shards():
            arrays = load_shard(path)
            length = len(arrays["policy"])
            indices = np.arange(length)
            if self.max_samples:
                remaining = self.max_samples - accepted
                if remaining <= 0:
                    break
                indices = indices[:remaining]
            accepted += len(indices)
            batches, pending = self._batch_stream(self._selected(arrays, indices), pending)
            yield from batches
        if pending is not None:
            yield {
                name: torch.from_numpy(np.ascontiguousarray(value))
                for name, value in pending.items()
            }

    def _iter_bounded_cross_game_mixed(
        self, rng: np.random.Generator
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Mix bounded cross-game windows before assembling optimizer batches."""

        accepted = 0
        buffer: dict[str, np.ndarray] | None = None
        pending: dict[str, np.ndarray] | None = None
        for path in self._shards():
            arrays = load_shard(path)
            indices = rng.permutation(len(arrays["policy"]))
            if self.max_samples:
                remaining = self.max_samples - accepted
                if remaining <= 0:
                    break
                indices = indices[:remaining]
            accepted += len(indices)
            selected = self._selected(arrays, indices)
            cursor = 0
            while cursor < len(indices):
                take = min(self.shuffle_buffer_samples, len(indices) - cursor)
                incoming = {
                    name: np.ascontiguousarray(value[cursor : cursor + take])
                    for name, value in selected.items()
                }
                cursor += take
                if buffer is None:
                    buffer = incoming
                    continue
                combined = self._concatenate([buffer, incoming])
                order = rng.permutation(len(combined["policy"]))
                keep = min(self.shuffle_buffer_samples, len(order))
                buffer = {
                    name: np.ascontiguousarray(value[order[:keep]])
                    for name, value in combined.items()
                }
                outgoing = {
                    name: np.ascontiguousarray(value[order[keep:]])
                    for name, value in combined.items()
                }
                batches, pending = self._batch_stream(outgoing, pending)
                yield from batches
        if buffer is not None:
            order = rng.permutation(len(buffer["policy"]))
            outgoing = {
                name: np.ascontiguousarray(value[order]) for name, value in buffer.items()
            }
            batches, pending = self._batch_stream(outgoing, pending)
            yield from batches
        if pending is not None:
            yield {
                name: torch.from_numpy(np.ascontiguousarray(value))
                for name, value in pending.items()
            }

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.shuffle:
            yield from self._iter_bounded_cross_game_mixed(rng)
        else:
            yield from self._iter_sequential()
