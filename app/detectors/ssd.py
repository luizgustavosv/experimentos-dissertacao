from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.datasets.heridal_to_voc import normalize_heridal_to_voc
from app.detectors.base import DetectorContext, Logger
from app.detectors.torchvision_detectors import TorchvisionDetector
from app.detectors.torchvision_models import build_ssd


class SSDDetector(TorchvisionDetector):
    def __init__(self, context: DetectorContext):
        super().__init__(context, build_ssd)

    def normalize_dataset(self, dataset_type: str, dataset_dir: Path, normalized_dir: Path, logger: Optional[Logger] = None):
        if dataset_type.lower() != "heridal":
            raise ValueError("Normalização SSD implementada apenas para o dataset HERIDAL.")
        normalized_path = normalize_heridal_to_voc(dataset_dir, normalized_dir, logger=logger)
        if logger:
            logger(f"[SSD][NORM] Dataset pronto em {normalized_path}")
        return normalized_path
