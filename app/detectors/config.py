from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 2
    lr: float = 0.001
    device: Optional[str] = None
    num_workers: int = 2
    imgsz: int = 640
    seed: int = 42
    weight_decay: float = 1e-4
    lr_step_size: int = 3
    lr_gamma: float = 0.1
    verbose: bool = False
    log_every: int = 10
    debug_dataloader: bool = False
    log_dir: Path = Path("logs")
    yolo_save_dir: Optional[Path] = None
    log_every_seconds: int = 10
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = 2
    drop_last: bool = False
    smoke_test_val_loss: bool = False
    smoke_test_samples: int = 8
    audit_datasets: bool = True
    dataset_num_classes: Optional[int] = None
    num_classes: Optional[int] = None
    val_mode: Literal["loss", "metrics"] = "loss"
    val_ratio: float = 0.1
