from __future__ import annotations

import torchvision
from torchvision.models import VGG16_Weights
from torchvision.models.detection import SSD300_VGG16_Weights
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights


def build_faster_rcnn(num_classes: int):
    # Evitar carregar cabeças pré-treinadas do COCO; usa apenas o backbone pré-treinado
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=FasterRCNN_ResNet50_FPN_Weights.COCO_V1.backbone_weights,
        num_classes=num_classes,
    )


def build_retinanet(num_classes: int):
    return torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT", num_classes=num_classes)


def build_ssd(num_classes: int):
    return torchvision.models.detection.ssd300_vgg16(
        weights=None, weights_backbone=VGG16_Weights.DEFAULT, num_classes=num_classes
    )
