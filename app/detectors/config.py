from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 2
    lr: float = 0.005
    device: Optional[str] = None
    num_workers: int = 2
    seed: int = 42
    weight_decay: float = 1e-4
    lr_step_size: int = 3
    lr_gamma: float = 0.1
