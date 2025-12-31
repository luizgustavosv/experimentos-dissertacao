from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from app.detectors.utils import read_json


class CocoDetectionDataset(Dataset):
    """Dataset mínimo para treino de detecção usando anotações COCO."""

    def __init__(self, images_dir: Path, ann_path: Path, transforms=None) -> None:
        self.images_dir = images_dir.expanduser().resolve()
        self.annotations = read_json(ann_path)
        self.transforms = transforms

        self.id_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        for ann in self.annotations.get("annotations", []):
            self.id_to_anns.setdefault(ann["image_id"], []).append(ann)

        self.images = self.annotations.get("images", [])
        self.class_id_to_name = {cat["id"]: cat["name"] for cat in self.annotations.get("categories", [])}
        self.num_classes = len(self.annotations.get("categories", [])) + 1  # +1 background

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_info = self.images[idx]
        img_path = self.images_dir / img_info["file_name"].split("/")[-1]
        image = Image.open(img_path).convert("RGB")

        anns = self.id_to_anns.get(img_info["id"], [])
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in anns:
            xmin, ymin, width, height = ann["bbox"]
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(int(ann["category_id"]))
            areas.append(float(ann.get("area", width * height)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        target: Dict[str, Any] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_info["id"]]),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
            "img_path": str(img_path),
            "orig_size": torch.tensor([image.height, image.width], dtype=torch.int64),
        }

        if not boxes:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        if self.transforms:
            try:
                image, target = self.transforms(image, target)
            except TypeError:
                image = self.transforms(image)

        target = self._clip_boxes_to_image(image, target)

        return image, target

    @staticmethod
    def _clip_boxes_to_image(image: torch.Tensor | Image.Image, target: Dict[str, Any]) -> Dict[str, Any]:
        boxes = target.get("boxes")
        if boxes is None or not torch.is_tensor(boxes):
            return target

        if isinstance(image, Image.Image):  # pragma: no cover - o fluxo principal usa tensor
            height, width = image.height, image.width
        else:
            height, width = int(image.shape[-2]), int(image.shape[-1])

        clamped = boxes.clone()
        clamped[:, 0::2] = clamped[:, 0::2].clamp(0, width)
        clamped[:, 1::2] = clamped[:, 1::2].clamp(0, height)
        target["boxes"] = clamped
        return target


class DetectionTransformCompose:
    def __init__(self, transforms: Sequence):
        self.transforms = list(transforms)

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class DetectionToTensor:
    def __call__(self, image, target):
        return F.to_tensor(image), target


class DetectionResize:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image, target):
        orig_w, orig_h = image.size
        if orig_h == self.size and orig_w == self.size:
            return image, target

        image = F.resize(image, [self.size, self.size])
        boxes = target.get("boxes")
        if torch.is_tensor(boxes):
            scale_x = self.size / float(orig_w)
            scale_y = self.size / float(orig_h)
            scaled = boxes.clone()
            scaled[:, 0] = boxes[:, 0] * scale_x
            scaled[:, 2] = boxes[:, 2] * scale_x
            scaled[:, 1] = boxes[:, 1] * scale_y
            scaled[:, 3] = boxes[:, 3] * scale_y
            target["boxes"] = scaled
        return image, target

