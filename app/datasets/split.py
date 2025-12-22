from __future__ import annotations

import math
import random
from typing import Dict, Iterable, Sequence, Tuple

SplitRatios = Tuple[float, float, float]


def assign_splits(items: Sequence[str], ratios: SplitRatios, seed: int = 42) -> Dict[str, str]:
    if len(ratios) != 3 or not math.isclose(sum(ratios), 1.0, rel_tol=1e-3):
        raise ValueError("As razões de divisão devem ter 3 valores somando 1.0 (ex.: 0.8, 0.1, 0.1)")

    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])

    assignments: Dict[str, str] = {}
    for idx, item in enumerate(shuffled):
        if idx < train_end:
            assignments[item] = "train"
        elif idx < val_end:
            assignments[item] = "val"
        else:
            assignments[item] = "test"
    return assignments


def apply_splits(items: Iterable[str], assignments: Dict[str, str]) -> Dict[str, str]:
    return {item: assignments.get(item, "train") for item in items}
