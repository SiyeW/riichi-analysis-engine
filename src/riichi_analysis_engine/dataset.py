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
        yielded = 0
        rng = np.random.default_rng(self.seed + self.epoch)
        for path in self._shards():
            arrays = load_shard(path)
            length = len(arrays["policy"])
            indices = rng.permutation(length) if self.shuffle else np.arange(length)
            for start in range(0, length, self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                if self.max_samples:
                    remaining = self.max_samples - yielded
                    if remaining <= 0:
                        return
                    batch_indices = batch_indices[:remaining]
                sample = {
                    name: torch.from_numpy(np.ascontiguousarray(value[batch_indices]))
                    for name, value in arrays.items()
                    if name not in {"perspective", "event_index"}
                }
                yield sample
                yielded += len(batch_indices)
                if self.max_samples and yielded >= self.max_samples:
                    return
