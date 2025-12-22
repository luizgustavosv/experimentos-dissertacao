from __future__ import annotations

import torchvision


def build_faster_rcnn(num_classes: int):
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT", num_classes=num_classes)


def build_retinanet(num_classes: int):
    return torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT", num_classes=num_classes)


def build_ssd(num_classes: int):
    return torchvision.models.detection.ssd300_vgg16(weights="DEFAULT", num_classes=num_classes)

