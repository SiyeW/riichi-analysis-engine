import numpy as np

from riichi_analysis_engine.constants import OBS_CHANNELS, TILE_TYPES
from riichi_analysis_engine.dataset import ShardDataset
from riichi_analysis_engine.storage import save_shard


def test_batches_continue_across_shards(tmp_path) -> None:
    for index in range(2):
        length = 3
        save_shard(
            tmp_path / f"game-{index}.npz",
            {
                    "obs": np.zeros((length, OBS_CHANNELS, TILE_TYPES), dtype=np.float32),
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


def test_shuffle_uses_a_bounded_cross_game_buffer(tmp_path) -> None:
    for index in range(4):
        save_shard(
            tmp_path / f"game-{index}.npz",
            {
                "obs": np.zeros((4, OBS_CHANNELS, TILE_TYPES), dtype=np.float32),
                "action_mask": np.ones((4, 46), dtype=bool),
                "policy": np.full(4, index, dtype=np.int8),
            },
        )

    first = list(
        ShardDataset(
            tmp_path,
            shuffle=True,
            seed=7,
            batch_size=4,
            shuffle_buffer_samples=4,
        )
    )
    second = list(
        ShardDataset(
            tmp_path,
            shuffle=True,
            seed=7,
            batch_size=4,
            shuffle_buffer_samples=4,
        )
    )
    first_values = [batch["policy"].numpy().tolist() for batch in first]
    second_values = [batch["policy"].numpy().tolist() for batch in second]

    assert first_values == second_values
    assert sorted(value for batch in first_values for value in batch) == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
    assert any(len(set(batch)) > 1 for batch in first_values)
