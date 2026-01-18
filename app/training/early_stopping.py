from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple


@dataclass
class EarlyStoppingConfig:
    enabled: bool
    patience: int
    min_delta: float = 0.0
    min_epochs: int = 0
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    mode: Literal["min", "max"] = "min"
    min_epochs: int = 0

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience deve ser >= 1")
        if self.min_epochs < 0:
            raise ValueError("min_epochs deve ser >= 0")
        if self.mode not in {"min", "max"}:
            raise ValueError("mode deve ser 'min' ou 'max'")
        self.best_value: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.num_bad_epochs: int = 0

    def step(self, epoch_idx: int, current_value: float) -> Tuple[bool, bool]:
        if math.isnan(current_value) or math.isinf(current_value):
            raise ValueError("current_value inválido para early stopping")

        improved = False
        if self.best_value is None:
            improved = True
        elif self.mode == "min":
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current_value
            self.best_epoch = epoch_idx
            self.num_bad_epochs = 0
        else:
            if epoch_idx >= self.min_epochs:
                self.num_bad_epochs += 1

        should_stop = epoch_idx >= self.min_epochs and self.num_bad_epochs >= self.patience
        return should_stop, improved

    def state_dict(self) -> dict[str, Optional[float] | Optional[int] | int | float | str]:
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
            "min_epochs": self.min_epochs,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "num_bad_epochs": self.num_bad_epochs,
        }

    def load_state_dict(self, state: dict[str, Optional[float] | Optional[int] | int | float | str]) -> None:
        self.best_value = state.get("best_value") if state.get("best_value") is not None else None
        self.best_epoch = state.get("best_epoch") if state.get("best_epoch") is not None else None
        self.num_bad_epochs = int(state.get("num_bad_epochs", 0))


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
