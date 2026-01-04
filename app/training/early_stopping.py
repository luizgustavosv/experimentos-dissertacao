from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainLossEMAStopper:
    patience: int
    min_delta: float
    min_epochs: int
    ema_alpha: float

    def __post_init__(self) -> None:
        if not (0 < self.ema_alpha <= 1):
            raise ValueError("ema_alpha deve estar em (0, 1]")
        if self.patience < 1:
            raise ValueError("patience deve ser >= 1")
        if self.min_epochs < 0:
            raise ValueError("min_epochs deve ser >= 0")
        self.ema_loss: Optional[float] = None
        self.best_ema: Optional[float] = None
        self.bad_epochs: int = 0

    def update(self, epoch_idx: int, avg_train_loss: float) -> Tuple[bool, float, float, int]:
        if math.isnan(avg_train_loss) or math.isinf(avg_train_loss):
            raise ValueError("avg_train_loss inválido para early stopping")

        if self.ema_loss is None:
            self.ema_loss = avg_train_loss
        else:
            self.ema_loss = self.ema_alpha * avg_train_loss + (1 - self.ema_alpha) * self.ema_loss

        if self.best_ema is None or (self.best_ema - self.ema_loss) >= self.min_delta:
            self.best_ema = self.ema_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        should_stop = epoch_idx >= self.min_epochs and self.bad_epochs >= self.patience
        return should_stop, self.ema_loss, self.best_ema, self.bad_epochs
