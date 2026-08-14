import numpy as np

from riichi_analysis_engine.dataset import ShardDataset
from riichi_analysis_engine.storage import save_shard


def test_batches_continue_across_shards(tmp_path) -> None:
    for index in range(2):
        length = 3
        save_shard(
            tmp_path / f"game-{index}.npz",
            {
                "obs": np.zeros((length, 1012, 34), dtype=np.float32),
                "action_mask": np.ones((length, 46), dtype=bool),
                "policy": np.arange(index * length, (index + 1) * length, dtype=np.int8),
            },
        )
    batches = list(ShardDataset(tmp_path, shuffle=False, batch_size=4))
    assert [len(batch["policy"]) for batch in batches] == [4, 2]
    assert np.concatenate([batch["policy"].numpy() for batch in batches]).tolist() == list(
        range(6)
    )
    limited = list(
        ShardDataset(tmp_path, shuffle=False, batch_size=4, max_samples=5)
    )
    assert [len(batch["policy"]) for batch in limited] == [4, 1]
