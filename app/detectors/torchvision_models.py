from __future__ import annotations

import torchvision
from torchvision.models import ResNet50_Weights, VGG16_Weights
from torchvision.models.detection import SSD300_VGG16_Weights


def build_faster_rcnn(num_classes: int):
    # Evitar carregar cabeças pré-treinadas do COCO; usa apenas o backbone pré-treinado
    backbone_weights = ResNet50_Weights.IMAGENET1K_V2
    try:
        return torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=num_classes,
        )
    except TypeError:
        return torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None,
            num_classes=num_classes,
        )


def build_retinanet(num_classes: int):
    return torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT", num_classes=num_classes)


def build_ssd(num_classes: int):
    return torchvision.models.detection.ssd300_vgg16(
        weights=None, weights_backbone=VGG16_Weights.DEFAULT, num_classes=num_classes
    )
