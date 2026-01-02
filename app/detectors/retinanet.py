from __future__ import annotations

from app.detectors.base import DetectorContext
from app.detectors.faster_rcnn import FasterRCNNDetector
from app.detectors.torchvision_detectors import TorchvisionDetector
from app.detectors.torchvision_models import build_retinanet


class RetinaNetDetector(FasterRCNNDetector):
    """Compartilha a normalização COCO com o Faster R-CNN, mas com arquitetura RetinaNet."""

    def __init__(self, context: DetectorContext):
        TorchvisionDetector.__init__(self, context, build_retinanet)

    def _prepare_model(self, num_classes: int, pretrained_weights, logger):  # type: ignore[override]
        # Usa o construtor configurado para RetinaNet em vez da implementação do Faster R-CNN.
        return TorchvisionDetector._prepare_model(self, num_classes, pretrained_weights, logger)
