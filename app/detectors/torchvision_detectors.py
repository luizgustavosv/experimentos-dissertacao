from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch

from app.detectors.base import DetectionAlgorithm, DetectorContext, Logger
from app.detectors.config import TrainConfig
from app.detectors.torchvision_train import train_torchvision_detector
from app.detectors.utils import ensure_weights_size, resolve_device, validate_coco_dataset
from app.metrics import Metrics


class TorchvisionDetector(DetectionAlgorithm):
    def __init__(self, context: DetectorContext, build_model: Callable[[int], torch.nn.Module], config: Optional[TrainConfig] = None):
        super().__init__(context)
        self.build_model = build_model
        self.config = config or TrainConfig()

    def train(
        self,
        dataset_dir: Path,
        pretrained_weights: Optional[Path],
        weights_out: Path,
        epochs: int,
        early_stop: bool,
        logger: Optional[Logger] = None,
    ) -> Metrics:
        train_ann, val_ann = validate_coco_dataset(dataset_dir)
        device_str = resolve_device(self.config.device)
        num_classes = self._infer_num_classes(train_ann)
        if logger:
            logger(f"[TRAIN] {self.context.name} em {device_str} com {num_classes} classes")
            logger(f"[DATA] Anotações train: {train_ann}")
            logger(f"[DATA] Anotações val: {val_ann}")

        model = self.build_model(num_classes)
        metrics = train_torchvision_detector(
            model,
            dataset_dir,
            train_ann,
            val_ann,
            weights_out,
            TrainConfig(
                epochs=epochs or self.config.epochs,
                batch_size=self.config.batch_size,
                lr=self.config.lr,
                device=self.config.device,
                num_workers=self.config.num_workers,
                seed=self.config.seed,
                weight_decay=self.config.weight_decay,
                lr_step_size=self.config.lr_step_size,
                lr_gamma=self.config.lr_gamma,
            ),
            logger=logger,
        )

        ensure_weights_size(weights_out)
        return metrics

    def _infer_num_classes(self, ann_path: Path) -> int:
        import json

        data = json.loads(ann_path.read_text(encoding="utf-8"))
        return len(data.get("categories", [])) + 1  # +1 para background

    def infer(self, images_dir: Path, weights_path: Path, report_out: Path, logger: Optional[Logger] = None):
        raise NotImplementedError("Inferência não implementada neste escopo de treino.")

    def validate(self, images_dir: Path, report_out: Path, plots_dir: Path, logger: Optional[Logger] = None):
        raise NotImplementedError("Validação não implementada neste escopo de treino.")

    def normalize_dataset(self, dataset_type: str, dataset_dir: Path, normalized_dir: Path, logger: Optional[Logger] = None):
        raise NotImplementedError("Normalização delegada ao módulo existente.")

