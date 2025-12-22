from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

from app.metrics import InferencePerformance, Metrics

Logger = Callable[[str], None]


@dataclass
class DetectorContext:
    name: str
    architecture: str
    recommended_repo: str
    target_format: str


class DetectionAlgorithm:
    def __init__(self, context: DetectorContext):
        self.context = context

    def train(
        self,
        dataset_dir: Path,
        pretrained_weights: Optional[Path],
        weights_out: Path,
        epochs: int,
        early_stop: bool,
        logger: Optional[Logger] = None,
    ) -> Metrics:
        raise NotImplementedError

    def infer(self, images_dir: Path, report_out: Path, logger: Optional[Logger] = None) -> InferencePerformance:
        raise NotImplementedError

    def validate(self, images_dir: Path, report_out: Path, plots_dir: Path, logger: Optional[Logger] = None) -> Metrics:
        raise NotImplementedError

    def normalize_dataset(self, dataset_type: str, dataset_dir: Path, normalized_dir: Path, logger: Optional[Logger] = None):
        raise NotImplementedError

    def dataset_target(self) -> str:
        return self.context.target_format

    def _log(self, message: str, logger: Optional[Logger]) -> None:
        if logger:
            logger(message)
