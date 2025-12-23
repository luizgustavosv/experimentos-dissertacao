from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from app.detectors import load_detectors
from app.detectors.base import DetectionAlgorithm, Logger
from app.metrics import InferencePerformance, Metrics


@dataclass
class OperationResult:
    metrics: Optional[Metrics] = None
    inference_performance: Optional[InferencePerformance] = None
    message: str = ""


class ExperimentController:
    def __init__(self) -> None:
        self.detectors: Dict[str, DetectionAlgorithm] = load_detectors()

    def execute_train(
        self,
        algorithm_key: str,
        dataset_dir: Path,
        pretrained_weights: Optional[Path],
        weights_out: Path,
        epochs: int,
        early_stop: bool,
        logger: Optional[Logger] = None,
    ) -> OperationResult:
        detector = self._get_detector(algorithm_key)
        metrics = detector.train(dataset_dir, pretrained_weights, weights_out, epochs, early_stop, logger)
        return OperationResult(metrics=metrics, message="Treinamento concluído.")

    def execute_infer(
        self,
        algorithm_key: str,
        images_dir: Path,
        weights_path: Optional[Path],
        report_out: Path,
        pedestrian_only: bool = False,
        logger: Optional[Logger] = None,
    ) -> OperationResult:
        detector = self._get_detector(algorithm_key)
        performance = detector.infer(images_dir, weights_path, report_out, pedestrian_only=pedestrian_only, logger=logger)
        return OperationResult(inference_performance=performance, message="Inferência concluída.")

    def execute_validate(
        self,
        algorithm_key: str,
        images_dir: Path,
        report_out: Path,
        plots_dir: Path,
        pedestrian_only: bool = False,
        logger: Optional[Logger] = None,
    ) -> OperationResult:
        detector = self._get_detector(algorithm_key)
        metrics = detector.validate(images_dir, report_out, plots_dir, pedestrian_only=pedestrian_only, logger=logger)
        return OperationResult(metrics=metrics, message="Validação concluída.")

    def execute_normalize(
        self,
        algorithm_key: str,
        dataset_type: str,
        dataset_dir: Path,
        normalized_dir: Path,
        logger: Optional[Logger] = None,
    ) -> OperationResult:
        detector = self._get_detector(algorithm_key)
        detector.normalize_dataset(dataset_type, dataset_dir, normalized_dir, logger)
        return OperationResult(metrics=None, message="Normalização concluída.")

    def _get_detector(self, algorithm_key: str) -> DetectionAlgorithm:
        if algorithm_key not in self.detectors:
            raise KeyError(f"Algoritmo desconhecido: {algorithm_key}")
        return self.detectors[algorithm_key]
