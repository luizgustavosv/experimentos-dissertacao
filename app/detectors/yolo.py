from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from app.detectors.base import DetectionAlgorithm, DetectorContext, Logger
from app.detectors.config import TrainConfig
from app.detectors.utils import copy_ultralytics_checkpoint, resolve_device, seed_everything, validate_yolo_dataset
from app.metrics import Metrics


class YoloDetector(DetectionAlgorithm):
    def __init__(self, context: DetectorContext, config: Optional[TrainConfig] = None) -> None:
        super().__init__(context)
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
        from ultralytics import YOLO  # import tardio para evitar dependências pesadas em import
        yaml_path = validate_yolo_dataset(dataset_dir)
        seed_everything(self.config.seed)
        device_str = resolve_device(self.config.device)
        num_epochs = epochs or self.config.epochs
        if logger:
            logger(f"[TRAIN] {self.context.name} iniciando treino com {num_epochs} épocas em {device_str}")
            logger(f"[DATA] YAML: {yaml_path}")

        base_weights = pretrained_weights or "yolov8n.pt"
        model = YOLO(base_weights)
        results = model.train(
            data=yaml_path,
            epochs=num_epochs,
            imgsz=640,
            batch=self.config.batch_size,
            device=device_str,
            workers=self.config.num_workers,
            lr0=self.config.lr,
            seed=self.config.seed,
        )

        run_dir = Path(results.save_dir)
        weights_path = copy_ultralytics_checkpoint(run_dir, weights_out)
        if logger:
            logger(f"[TRAIN] Checkpoint copiado de {run_dir} para {weights_path}")

        return Metrics(
            precision=0.0,
            recall=0.0,
            map50=0.0,
            map50_95=0.0,
            loss_final=results.results_dict.get("loss", None) if hasattr(results, "results_dict") else None,
            epochs=num_epochs,
            train_images=len(list((dataset_dir / "images" / "train").glob("*"))),
            device=device_str,
            weights_path=weights_path,
            map_computed=False,
        )

    def infer(self, images_dir: Path, weights_path: Path, report_out: Path, logger: Optional[Logger] = None):
        raise NotImplementedError("Inferência via Ultralytics não implementada neste escopo de treino.")

    def validate(self, images_dir: Path, report_out: Path, plots_dir: Path, logger: Optional[Logger] = None):
        raise NotImplementedError("Validação via Ultralytics não implementada neste escopo de treino.")

    def normalize_dataset(self, dataset_type: str, dataset_dir: Path, normalized_dir: Path, logger: Optional[Logger] = None):
        raise NotImplementedError("Normalização delegada ao módulo de datasets existente.")
