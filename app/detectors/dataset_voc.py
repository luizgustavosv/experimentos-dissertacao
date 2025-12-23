from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

_VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class PascalVOCDataset(Dataset):
    """Dataset mínimo para treino de detecção usando anotações Pascal VOC."""

    def __init__(self, dataset_root: Path, image_ids: Sequence[str], class_to_idx: Mapping[str, int], transforms=None) -> None:
        self.dataset_root = dataset_root.expanduser().resolve()
        self.images_dir = self.dataset_root / "JPEGImages"
        self.annotations_dir = self.dataset_root / "Annotations"
        self.image_ids = [img_id.strip() for img_id in image_ids if img_id.strip()]
        self.class_to_idx = dict(class_to_idx)
        self.transforms = transforms

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_id = self.image_ids[idx]
        image_path = self._resolve_image_path(image_id)
        annotation_path = self.annotations_dir / f"{image_id}.xml"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Anotação VOC não encontrada: {annotation_path}")

        image = Image.open(image_path).convert("RGB")
        boxes, labels, areas = self._parse_annotation(annotation_path)

        target: Dict[str, Any] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }

        if not boxes:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def _resolve_image_path(self, image_id: str) -> Path:
        candidates = [
            self.images_dir / f"{image_id}{ext}" for ext in _VALID_IMAGE_EXTENSIONS if (self.images_dir / f"{image_id}{ext}").exists()
        ]
        if not candidates:
            matches = list(self.images_dir.glob(f"{image_id}.*"))
            if not matches:
                raise FileNotFoundError(f"Imagem não encontrada para o id {image_id} em {self.images_dir}")
            candidates = matches
        return candidates[0]

    def _parse_annotation(self, annotation_path: Path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        boxes = []
        labels = []
        areas = []

        for obj in root.findall("object"):
            name = obj.findtext("name")
            if name is None:
                continue
            if name not in self.class_to_idx:
                raise ValueError(f"Classe '{name}' não encontrada no mapeamento de classes.")
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            try:
                xmin = float(bndbox.findtext("xmin"))
                ymin = float(bndbox.findtext("ymin"))
                xmax = float(bndbox.findtext("xmax"))
                ymax = float(bndbox.findtext("ymax"))
            except (TypeError, ValueError):
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[name])
            areas.append(max(0.0, (xmax - xmin) * (ymax - ymin)))

        return boxes, labels, areas
