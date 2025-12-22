from __future__ import annotations

from typing import Dict

from app.detectors.base import DetectionAlgorithm, DetectorContext
from app.detectors.stubs import StubDetector


def load_detectors() -> Dict[str, DetectionAlgorithm]:
    """Retorna detectores configurados com informações de repositórios públicos."""
    architectures = {
        "YOLO": DetectorContext(
            name="YOLOv12n",
            architecture="YOLO",
            recommended_repo="https://github.com/ultralytics/ultralytics",
        ),
        "SSD": DetectorContext(
            name="SSD-300",
            architecture="SSD",
            recommended_repo="https://github.com/pytorch/vision/tree/main/references/detection",
        ),
        "Faster R-CNN": DetectorContext(
            name="Faster R-CNN",
            architecture="Faster R-CNN",
            recommended_repo="https://github.com/pytorch/vision",
        ),
        "RetinaNet": DetectorContext(
            name="RetinaNet",
            architecture="RetinaNet",
            recommended_repo="https://github.com/pytorch/vision",
        ),
    }

    return {key: StubDetector(ctx) for key, ctx in architectures.items()}
