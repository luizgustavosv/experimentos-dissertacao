from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional


@dataclass
class Metrics:
    precision: float
    recall: float
    map50: float
    map50_95: float
    loss_final: Optional[float] = None
    epochs: Optional[int] = None
    train_images: Optional[int] = None
    device: Optional[str] = None
    weights_path: Optional[Path] = None
    map_computed: bool = True
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        payload: Dict[str, float] = {
            "precision": self.precision,
            "recall": self.recall,
            "map50": self.map50,
            "map50_95": self.map50_95,
        }
        if self.loss_final is not None:
            payload["loss_final"] = float(self.loss_final)
        if self.epochs is not None:
            payload["epochs"] = float(self.epochs)
        if self.train_images is not None:
            payload["train_images"] = float(self.train_images)
        if self.device is not None:
            payload["device"] = str(self.device)
        if self.weights_path is not None:
            payload["weights_path"] = str(self.weights_path)
        payload.update({k: float(v) for k, v in self.extra.items()})
        payload["map_computed"] = bool(self.map_computed)
        return payload

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
            loss_final=self.loss_final,
            epochs=self.epochs,
            train_images=self.train_images,
            device=self.device,
            weights_path=self.weights_path,
            map_computed=self.map_computed,
            extra=self.extra,
        )


@dataclass
class InferencePerformance:
    images_per_second: float
    milliseconds_per_image: float
    total_images: Optional[int] = None
    total_detections: Optional[int] = None
    total_inference_seconds: Optional[float] = None
    hardware: Optional[str] = None

    def to_dict(self) -> Dict[str, float]:
        payload: Dict[str, float] = {
            "images_per_second": self.images_per_second,
            "milliseconds_per_image": self.milliseconds_per_image,
        }
        if self.total_images is not None:
            payload["total_images"] = float(self.total_images)
        if self.total_detections is not None:
            payload["total_detections"] = float(self.total_detections)
        if self.total_inference_seconds is not None:
            payload["total_inference_seconds"] = float(self.total_inference_seconds)
        if self.hardware is not None:
            payload["hardware"] = str(self.hardware)
        return payload
