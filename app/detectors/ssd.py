from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.datasets.heridal_to_voc import normalize_heridal_to_voc
from app.datasets.visdrone_to_voc import find_visdrone_splits, normalize_visdrone_to_voc
from app.detectors.base import DetectorContext, Logger
from app.detectors.config import TrainConfig
from app.detectors.dataset_voc import PascalVOCDataset
from app.detectors.torchvision_detectors import TorchvisionDetector
from app.detectors.torchvision_train import train_torchvision_detector
from app.detectors.torchvision_models import build_ssd
from app.detectors.utils import resolve_device, validate_voc_dataset


class SSDDetector(TorchvisionDetector):
    def __init__(self, context: DetectorContext):
        super().__init__(context, build_ssd)

    def train(
        self,
        dataset_dir: Path,
        pretrained_weights: Optional[Path],
        weights_out: Path,
        epochs: int,
        early_stop: bool,
        logger: Optional[Logger] = None,
    ):
        from torchvision import transforms

        dataset_root, class_names, train_ids, val_ids = validate_voc_dataset(dataset_dir)
        class_to_idx = {name: idx + 1 for idx, name in enumerate(class_names)}
        device_str = resolve_device(self.config.device)
        num_classes = len(class_names) + 1  # +1 para background

        if logger:
            logger(f"[TRAIN] {self.context.name} em {device_str} com {num_classes} classes (Pascal VOC)")
            logger(f"[DATA] Raiz do dataset VOC: {dataset_root}")
            logger(f"[DATA] Splits: train={len(train_ids)}, val={len(val_ids)}")
            logger(f"[DATA] Classes: {', '.join(class_names)}")

        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = PascalVOCDataset(dataset_root, train_ids, class_to_idx, transforms=transform)
        val_dataset = PascalVOCDataset(dataset_root, val_ids, class_to_idx, transforms=transform)

        model = self.build_model(num_classes)
        metrics = train_torchvision_detector(
            model,
            dataset_root,
            train_ann=None,
            val_ann=None,
            weights_out=weights_out,
            config=TrainConfig(
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
            val_ratio=0.0,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
        return metrics

    def normalize_dataset(self, dataset_type: str, dataset_dir: Path, normalized_dir: Path, logger: Optional[Logger] = None):
        dataset_type = dataset_type.lower()
        if dataset_type == "heridal":
            normalized_path = normalize_heridal_to_voc(dataset_dir, normalized_dir, logger=logger)
        else:
            visdrone_splits = find_visdrone_splits(dataset_dir)
            has_visdrone_structure = any(visdrone_splits.values())
            if dataset_type == "visdrone" or has_visdrone_structure:
                normalized_path = normalize_visdrone_to_voc(dataset_dir, normalized_dir, logger=logger)
            else:
                raise ValueError("Normalização SSD implementada apenas para os datasets HERIDAL ou VisDrone.")
        if logger:
            logger(f"[SSD][NORM] Dataset pronto em {normalized_path}")
        return normalized_path
