from __future__ import annotations

from typing import Dict

from app.detectors.base import DetectionAlgorithm, DetectorContext
from app.detectors.faster_rcnn import FasterRCNNDetector
from app.detectors.retinanet import RetinaNetDetector
from app.detectors.ssd import SSDDetector
from app.detectors.yolo import YoloDetector


def load_detectors() -> Dict[str, DetectionAlgorithm]:
    """Retorna detectores configurados com informações de repositórios públicos."""
    architectures = {
        "YOLO": DetectorContext(
            name="YOLOv12n",
            architecture="YOLO",
            recommended_repo="https://github.com/ultralytics/ultralytics",
            target_format="yolo",
        ),
        "SSD": DetectorContext(
            name="SSD-300",
            architecture="SSD",
            recommended_repo="https://github.com/pytorch/vision/tree/main/references/detection",
            target_format="voc",
        ),
        "Faster R-CNN": DetectorContext(
            name="Faster R-CNN",
            architecture="Faster R-CNN",
            recommended_repo="https://github.com/pytorch/vision",
            target_format="coco",
        ),
        "RetinaNet": DetectorContext(
            name="RetinaNet",
            architecture="RetinaNet",
            recommended_repo="https://github.com/pytorch/vision",
            target_format="coco",
        ),
    }

    return {
        "YOLO": YoloDetector(architectures["YOLO"]),
        "SSD": SSDDetector(architectures["SSD"]),
        "Faster R-CNN": FasterRCNNDetector(architectures["Faster R-CNN"]),
        "RetinaNet": RetinaNetDetector(architectures["RetinaNet"]),
    }
