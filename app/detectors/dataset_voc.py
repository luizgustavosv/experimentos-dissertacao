from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

_VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class PascalVOCDataset(Dataset):
    """Dataset mínimo para treino de detecção usando anotações Pascal VOC."""

    def __init__(
        self,
        dataset_root: Path,
        image_ids: Sequence[str],
        class_to_idx: Mapping[str, int],
        transforms=None,
        *,
        logger: Optional[logging.Logger] = None,
        debug: bool = False,
        split_metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        num_classes: Optional[int] = None,
    ) -> None:
        self.dataset_root = dataset_root.expanduser().resolve()
        self.images_dir = self.dataset_root / "JPEGImages"
        self.annotations_dir = self.dataset_root / "Annotations"
        self.image_ids = [img_id.strip() for img_id in image_ids if img_id.strip()]
        self.class_to_idx = {name: int(idx) for name, idx in dict(class_to_idx).items()}
        self.transforms = transforms
        self.logger = logger
        self.debug = debug
        self.num_classes = num_classes or (max(self.class_to_idx.values()) + 1 if self.class_to_idx else None)
        self.split_metadata = list(split_metadata) if split_metadata else []

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        stage_base = f"[STAGE=dataset][idx={idx}]"
        meta = self.split_metadata[idx] if idx < len(self.split_metadata) else {}
        line_no = meta.get("line_no") if isinstance(meta, dict) else None
        meta_hint = f" line_no={line_no}" if line_no is not None else ""
        image_id = self.image_ids[idx]
        try:
            image_path = self._resolve_image_path(image_id)
        except Exception as exc:  # pragma: no cover - proteção extra
            raise FileNotFoundError(
                f"{stage_base}{meta_hint} Falha ao resolver imagem id={image_id}: {exc}"
            ) from exc

        annotation_path = self.annotations_dir / f"{image_id}.xml"
        if not annotation_path.exists():
            raise FileNotFoundError(
                f"{stage_base}{meta_hint} Anotação VOC não encontrada: {annotation_path}"
            )

        if self.debug and self.logger:
            self.logger.debug(
                "%s raw_id=%s img_resolved=%s ann_resolved=%s", stage_base, image_id, image_path, annotation_path
            )

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:  # pragma: no cover - robustez de IO
            raise RuntimeError(
                f"{stage_base}{meta_hint} erro ao ler imagem {image_path}: {exc}"
            ) from exc

        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"{stage_base}{meta_hint} imagem com dimensões inválidas: {width}x{height} em {image_path}")
        if self.debug and self.logger:
            self.logger.debug(
                "%s imagem lida shape=%sx%s dtype=%s", stage_base, height, width, getattr(image, "dtype", "PIL")
            )

        try:
            parsed = self._parse_annotation(annotation_path, width, height, meta_hint)
        except Exception as exc:
            raise RuntimeError(
                f"{stage_base}{meta_hint} falha ao parsear anotação idx={idx} ann_path={annotation_path} img_path={image_path}: {exc}"
            ) from exc

        if isinstance(parsed, (tuple, list)):
            if len(parsed) >= 3:
                boxes_list, labels_list = parsed[0], parsed[1]
                clamp_count = parsed[2]
                extras = list(parsed[3:]) if len(parsed) > 3 else []
                if len(parsed) >= 4 and not isinstance(clamp_count, (int, float)) and isinstance(parsed[3], (int, float)):
                    extras.insert(0, clamp_count)
                    clamp_count = parsed[3]
            else:
                raise ValueError(
                    f"{stage_base}{meta_hint} parse_annotation retornou menos de 3 itens (idx={idx} ann_path={annotation_path} img_path={image_path})"
                )
        else:
            raise TypeError(
                f"{stage_base}{meta_hint} parse_annotation deve retornar tuple/list (idx={idx} ann_path={annotation_path} img_path={image_path})"
            )

        if extras and self.logger and self.logger.isEnabledFor(logging.DEBUG):
            try:
                extras_info = []
                for extra in extras:
                    if isinstance(extra, dict):
                        extras_info.append(f"dict_keys={list(extra.keys())}")
                    else:
                        extras_info.append(type(extra).__name__)
                self.logger.debug(
                    "%s extras adicionais em parse_annotation: num_extras=%d detalhes=%s", stage_base, len(extras), extras_info
                )
            except Exception:
                self.logger.debug("%s extras adicionais em parse_annotation: num_extras=%d", stage_base, len(extras))

        try:
            clamp_count = int(clamp_count) if clamp_count is not None else 0
        except Exception:
            if self.logger and self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("%s clamp_count não numérico; forçando para 0 (valor=%s)", stage_base, clamp_count)
            clamp_count = 0

        if len(boxes_list) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.as_tensor(boxes_list, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels_list, dtype=torch.int64)

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

        if boxes_tensor.shape[0] == 0 and len(boxes_list) > 0:
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
            if self.debug and self.logger:
                self.logger.debug(
                    "%s antes_transform shape=%s boxes=%d", stage_base, tuple(image.size)[::-1], boxes_tensor.shape[0]
                )
            image = self.transforms(image)
            if self.debug and self.logger:
                self.logger.debug(
                    "%s depois_transform shape=%s boxes=%d", stage_base, tuple(image.shape), target["boxes"].shape[0]
                )

        if self.num_classes and (target["labels"] >= self.num_classes).any():
            raise ValueError(
                f"{stage_base}{meta_hint} classe fora do intervalo (num_classes={self.num_classes}) em {annotation_path}"
            )

        if clamp_count > 0 and self.logger:
            self.logger.warning(
                "%s boxes foram clampadas %d vez(es) para ficar dentro da imagem %s", stage_base, clamp_count, image_path
            )

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

    def _parse_annotation(self, annotation_path: Path, width: int, height: int, meta_hint: str = ""):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        boxes = []
        labels = []
        areas = []
        clamp_count = 0

        if self.debug and self.logger:
            self.logger.debug(
                "[STAGE=annotation%s] parsing=%s format=VOC", meta_hint, annotation_path
            )

        for obj in root.findall("object"):
            name = obj.findtext("name")
            if name is None:
                continue

            class_name = name.lower().strip()
            if class_name not in ("pedestrian", "human"):
                continue

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

            if not all(math.isfinite(v) for v in (xmin, ymin, xmax, ymax)):
                raise ValueError(
                    f"[STAGE=annotation{meta_hint}] coordenadas não finitas em {annotation_path}: {(xmin, ymin, xmax, ymax)}"
                )

            if xmax <= xmin or ymax <= ymin:
                raise ValueError(
                    f"[STAGE=annotation{meta_hint}] box inválida (ordem) em {annotation_path}: {(xmin, ymin, xmax, ymax)}"
                )

            if xmax > width or ymax > height or xmin < 0 or ymin < 0:
                clamp_count += 1
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # background=0, person=1
            areas.append(max(0.0, (xmax - xmin) * (ymax - ymin)))

        if self.debug and self.logger:
            self.logger.debug(
                "[STAGE=annotation%s] num_boxes=%d classes=%s minmax=%s", meta_hint, len(boxes),
                sorted(set(labels)) if labels else [],
                (
                    (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
                    if boxes
                    else "<empty>"
                ),
            )

        return boxes, labels, areas, clamp_count
