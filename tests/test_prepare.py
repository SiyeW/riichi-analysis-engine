from __future__ import annotations

from riichi_analysis_engine.prepare import (
    mortal_validation_members,
    selection_hash,
    split_holdout,
    stable_source_order,
)


def test_manifest_selection_is_order_independent() -> None:
    names = ["x.mjson", "y.mjson", "z.mjson"]
    assert stable_source_order(names, 20252026) == stable_source_order(
        list(reversed(names)), 20252026
    )
    assert mortal_validation_members(names) == mortal_validation_members(
        list(reversed(names))
    )


def test_split_holdout_is_deterministic_disjoint_and_complete() -> None:
    names = [f"game-{index:04d}.mjson" for index in range(200)]
    holdout, training = split_holdout(names, 8, 20252026)

    assert len(holdout) == 8
    assert len(training) == 192
    assert not set(holdout) & set(training)
    assert sorted(holdout + training) == sorted(names)

    # Directory order must not change the split.
    assert split_holdout(list(reversed(names)), 8, 20252026) == (holdout, training)
    # A different seed must produce a different holdout.
    assert split_holdout(names, 8, 1)[0] != holdout
    # Rebuilding the same split must reproduce the same fingerprint.
    assert selection_hash(split_holdout(names, 8, 20252026)[0]) == selection_hash(holdout)


def test_split_holdout_rejects_impossible_counts() -> None:
    names = ["a.mjson", "b.mjson"]
    for count in (-1, 3):
        try:
            split_holdout(names, count, 20252026)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for holdout count {count}")


def test_split_holdout_can_take_everything() -> None:
    names = ["a.mjson", "b.mjson"]
    holdout, remaining = split_holdout(names, 2, 20252026)
    assert sorted(holdout) == sorted(names)
    assert remaining == []
