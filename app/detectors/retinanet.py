from __future__ import annotations

from app.detectors.base import DetectorContext
from app.detectors.faster_rcnn import FasterRCNNDetector
from app.detectors.torchvision_detectors import TorchvisionDetector
from app.detectors.torchvision_models import build_retinanet


class RetinaNetDetector(FasterRCNNDetector):
    """Compartilha a normalização COCO com o Faster R-CNN, mas com arquitetura RetinaNet."""

    def __init__(self, context: DetectorContext):
        TorchvisionDetector.__init__(self, context, build_retinanet)
