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
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.shuffle = shuffle
        self.seed = seed
        self.max_samples = max_samples
        self.batch_size = batch_size
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

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        accepted = 0
        rng = np.random.default_rng(self.seed + self.epoch)
        pending: dict[str, np.ndarray] | None = None
        for path in self._shards():
            arrays = load_shard(path)
            length = len(arrays["policy"])
            indices = rng.permutation(length) if self.shuffle else np.arange(length)
            if self.max_samples:
                remaining = self.max_samples - accepted
                if remaining <= 0:
                    break
                indices = indices[:remaining]
            accepted += len(indices)
            selected = {
                name: value[indices]
                for name, value in arrays.items()
                if name not in {"perspective", "event_index"}
            }
            cursor = 0
            if pending is not None:
                pending_length = len(next(iter(pending.values())))
                take = min(self.batch_size - pending_length, len(indices))
                pending = {
                    name: np.concatenate((value, selected[name][:take]), axis=0)
                    for name, value in pending.items()
                }
                cursor = take
                if len(next(iter(pending.values()))) == self.batch_size:
                    yield {
                        name: torch.from_numpy(np.ascontiguousarray(value))
                        for name, value in pending.items()
                    }
                    pending = None

            while cursor + self.batch_size <= len(indices):
                stop = cursor + self.batch_size
                yield {
                    name: torch.from_numpy(np.ascontiguousarray(value[cursor:stop]))
                    for name, value in selected.items()
                }
                cursor = stop
            if cursor < len(indices):
                tail = {
                    name: np.ascontiguousarray(value[cursor:])
                    for name, value in selected.items()
                }
                if pending is None:
                    pending = tail
                else:
                    pending = {
                        name: np.concatenate((pending[name], value), axis=0)
                        for name, value in tail.items()
                    }
        if pending is not None:
            yield {
                name: torch.from_numpy(np.ascontiguousarray(value))
                for name, value in pending.items()
            }
