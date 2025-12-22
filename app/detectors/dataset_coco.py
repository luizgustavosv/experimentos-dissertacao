from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

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
        }

        if not boxes:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        if self.transforms:
            image = self.transforms(image)

        return image, target

