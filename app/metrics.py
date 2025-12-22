from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass
class Metrics:
    precision: float
    recall: float
    map50: float
    map50_95: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "map50": self.map50,
            "map50_95": self.map50_95,
        }

    @classmethod
    def from_values(cls, values: Iterable[float]) -> "Metrics":
        precision, recall, map50, map50_95 = values
        return cls(precision=float(precision), recall=float(recall), map50=float(map50), map50_95=float(map50_95))

    def as_percentage(self) -> "Metrics":
        return Metrics(
            precision=self.precision * 100,
            recall=self.recall * 100,
            map50=self.map50 * 100,
            map50_95=self.map50_95 * 100,
        )


@dataclass
class InferencePerformance:
    images_per_second: float
    milliseconds_per_image: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "images_per_second": self.images_per_second,
            "milliseconds_per_image": self.milliseconds_per_image,
        }
