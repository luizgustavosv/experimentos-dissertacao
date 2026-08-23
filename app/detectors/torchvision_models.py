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
            box_detections_per_img=300,
        )
    except TypeError:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
        model.roi_heads.detections_per_img = 300
        return model


def build_retinanet(num_classes: int):
    from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights, retinanet_resnet50_fpn
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead

    # Carrega apenas backbone/FPN pré-treinados no COCO e recria o head para o dataset alvo.
    weights = RetinaNet_ResNet50_FPN_Weights.COCO_V1
    try:
        model = retinanet_resnet50_fpn(weights=weights, detections_per_img=300)
    except TypeError:
        model = retinanet_resnet50_fpn(weights=weights)
        model.detections_per_img = 300

    classification_head = model.head.classification_head
    num_anchors = classification_head.num_anchors
    # Conv2dNormActivation -> acesso ao Conv2d interno pelo índice 0
    in_channels = classification_head.conv[0][0].in_channels

    model.head.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)
    model.num_classes = num_classes
    return model


def build_ssd(num_classes: int):
    try:
        return torchvision.models.detection.ssd300_vgg16(
            weights=None, weights_backbone=VGG16_Weights.DEFAULT, num_classes=num_classes, detections_per_img=300
        )
    except TypeError:
        model = torchvision.models.detection.ssd300_vgg16(
            weights=None, weights_backbone=VGG16_Weights.DEFAULT, num_classes=num_classes
        )
        model.detections_per_img = 300
        return model
