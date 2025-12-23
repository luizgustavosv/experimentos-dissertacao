from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.datasets.heridal_to_voc import normalize_heridal_to_voc
from app.datasets.visdrone_to_voc import find_visdrone_splits, normalize_visdrone_to_voc
from app.detectors.base import DetectorContext, Logger
from app.detectors.torchvision_detectors import TorchvisionDetector
from app.detectors.torchvision_models import build_ssd


class SSDDetector(TorchvisionDetector):
    def __init__(self, context: DetectorContext):
        super().__init__(context, build_ssd)

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
