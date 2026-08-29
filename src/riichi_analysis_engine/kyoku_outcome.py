from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OutcomeClass:
    kind: str
    winners: tuple[int, ...] = ()
    target: int | None = None


OUTCOME_CLASSES = (
    OutcomeClass("draw"),
    *(OutcomeClass("tsumo", (winner,), winner) for winner in range(4)),
    *(
        OutcomeClass(
            "ron",
            tuple(player for player in range(4) if winner_mask & (1 << player)),
            target,
        )
        for target in range(4)
        for winner_mask in range(1, 16)
        if not winner_mask & (1 << target)
    ),
)
OUTCOME_COUNT = len(OUTCOME_CLASSES)
OUTCOME_INDEX = {outcome: index for index, outcome in enumerate(OUTCOME_CLASSES)}


def outcome_class_index(win: np.ndarray, targets: np.ndarray) -> int:
    winners = tuple(int(player) for player in np.flatnonzero(win))
    if not winners:
        return 0
    winner_targets = {int(targets[winner]) for winner in winners}
    if len(winner_targets) != 1:
        raise ValueError("all winners must share the same target")
    target = winner_targets.pop()
    if len(winners) == 1 and target == winners[0]:
        outcome = OutcomeClass("tsumo", winners, target)
    else:
        if target < 0 or target in winners:
            raise ValueError("ron target must be a non-winning player")
        outcome = OutcomeClass("ron", winners, target)
    return OUTCOME_INDEX[outcome]
