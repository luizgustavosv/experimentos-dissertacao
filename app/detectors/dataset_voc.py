from __future__ import annotations

import logging
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
        self.class_to_idx = {name: int(idx) for name, idx in dict(class_to_idx).items()}
        if self.class_to_idx.get("human") != 1:
            raise ValueError("Mapeamento de classes inválido: 'human' deve estar mapeada para o índice 1 (background=0)")
        if any(idx <= 0 for idx in self.class_to_idx.values()):
            raise ValueError("Mapeamento de classes inválido: labels devem ser positivos (background=0 é reservado)")
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
        width, height = image.size
        boxes, labels, _ = self._parse_annotation(annotation_path)

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        if boxes_tensor.numel() > 0 and width > 0 and height > 0:
            boxes_tensor = boxes_tensor.clone()
            boxes_tensor[:, 0::2] = boxes_tensor[:, 0::2].clamp(0, width - 1)
            boxes_tensor[:, 1::2] = boxes_tensor[:, 1::2].clamp(0, height - 1)

        valid = torch.ones(boxes_tensor.shape[0], dtype=torch.bool)
        if boxes_tensor.numel() > 0:
            finite_mask = torch.isfinite(boxes_tensor).all(dim=1)
            valid &= finite_mask
            widths = boxes_tensor[:, 2] - boxes_tensor[:, 0]
            heights = boxes_tensor[:, 3] - boxes_tensor[:, 1]
            spatial_valid = (widths > 0) & (heights > 0)
            valid &= spatial_valid

        label_mask = torch.isfinite(labels_tensor.float()) & (labels_tensor > 0)
        valid &= label_mask

        if valid.any():
            boxes_tensor = boxes_tensor[valid]
            labels_tensor = labels_tensor[valid]
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        areas_tensor = torch.zeros((boxes_tensor.shape[0],), dtype=torch.float32)
        if boxes_tensor.numel() > 0:
            widths = boxes_tensor[:, 2] - boxes_tensor[:, 0]
            heights = boxes_tensor[:, 3] - boxes_tensor[:, 1]
            areas_tensor = widths * heights

        if boxes_tensor.shape[0] == 0 and boxes:
            logging.warning("Todas as boxes foram descartadas por serem inválidas. id=%s path=%s", image_id, image_path)

        target: Dict[str, Any] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
            "area": areas_tensor,
            "iscrowd": torch.zeros(boxes_tensor.shape[0], dtype=torch.int64),
            "img_path": str(image_path),
        }

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
