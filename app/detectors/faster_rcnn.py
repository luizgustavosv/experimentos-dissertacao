from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from app.detectors.base import DetectorContext, Logger
from app.detectors.torchvision_detectors import TorchvisionDetector
from app.detectors.torchvision_models import build_faster_rcnn


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class ImageAnnotations:
    filename: str
    width: Optional[int] = None
    height: Optional[int] = None
    boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)


class FasterRCNNDetector(TorchvisionDetector):
    def __init__(self, context: DetectorContext):
        super().__init__(context, build_faster_rcnn)

    def normalize_dataset(self, dataset_type: str, dataset_dir: Path, normalized_dir: Path, logger: Optional[Logger] = None):
        if dataset_type.lower() != "heridal":
            raise ValueError("Normalização Faster R-CNN implementada apenas para o dataset HERIDAL.")

        dataset_dir = dataset_dir.expanduser().resolve()
        normalized_dir = normalized_dir.expanduser().resolve()
        train_dir = dataset_dir / "train"

        if not train_dir.exists():
            raise FileNotFoundError(f"Pasta 'train' não encontrada em {dataset_dir}")

        csv_path = train_dir / "annotations.csv"
        if not csv_path.exists():
            legacy_csv = train_dir / "_annotations.csv"
            if legacy_csv.exists():
                csv_path = legacy_csv
                self._log(f"[FRCNN][NORM] Usando arquivo legado de anotações: {legacy_csv.name}", logger)
            else:
                raise FileNotFoundError(f"annotations.csv não encontrado em {train_dir}")

        images = self._list_images(train_dir)
        deduped_images = self._deduplicate_images(images, logger)
        if not deduped_images:
            raise ValueError(f"Nenhuma imagem válida encontrada em {train_dir}")

        annotations, class_order = self._parse_annotations(csv_path, deduped_images, logger)
        total_images = len(deduped_images)
        images_with_ann = sum(1 for key in deduped_images if annotations.get(key))
        images_without_ann = total_images - images_with_ann

        rng = random.Random(42)
        shuffled = sorted(deduped_images.values(), key=lambda p: p.name.lower())
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * 0.8)
        train_files = shuffled[:split_idx]
        val_files = shuffled[split_idx:]

        images_root = normalized_dir / "images"
        annotations_root = normalized_dir / "annotations"
        train_images_dir = images_root / "train"
        val_images_dir = images_root / "val"
        for path in [train_images_dir, val_images_dir, annotations_root]:
            path.mkdir(parents=True, exist_ok=True)

        categories = self._build_categories(class_order, logger)

        stats = {"valid_boxes": 0, "discarded_boxes": 0}
        train_payload = self._build_coco_split(
            train_files,
            train_images_dir,
            annotations,
            categories,
            "train",
            stats,
            logger,
        )
        val_payload = self._build_coco_split(
            val_files,
            val_images_dir,
            annotations,
            categories,
            "val",
            stats,
            logger,
        )

        train_json = annotations_root / "instances_train.json"
        val_json = annotations_root / "instances_val.json"
        train_json.write_text(json.dumps(train_payload, indent=2), encoding="utf-8")
        val_json.write_text(json.dumps(val_payload, indent=2), encoding="utf-8")

        self._log(f"[FRCNN][NORM] Imagens no disco: {total_images}", logger)
        self._log(f"[FRCNN][NORM] Imagens com anotação: {images_with_ann}", logger)
        self._log(f"[FRCNN][NORM] Imagens sem anotação: {images_without_ann}", logger)
        self._log(f"[FRCNN][NORM] BBoxes válidos: {stats['valid_boxes']}", logger)
        self._log(f"[FRCNN][NORM] BBoxes descartados: {stats['discarded_boxes']}", logger)
        self._log(f"[FRCNN][NORM] Total de anotações escritas: {stats['valid_boxes']}", logger)
        self._log(f"[FRCNN][NORM] JSON train: {train_json}", logger)
        self._log(f"[FRCNN][NORM] JSON val: {val_json}", logger)

        report = {
            "source_dataset": str(dataset_dir),
            "total_images": total_images,
            "train_images": len(train_files),
            "val_images": len(val_files),
            "images_with_annotations": images_with_ann,
            "images_without_annotations": images_without_ann,
            "bboxes": stats,
            "categories": categories,
            "outputs": {
                "train_annotations": str(train_json),
                "val_annotations": str(val_json),
                "train_images_dir": str(train_images_dir),
                "val_images_dir": str(val_images_dir),
            },
            "split": {"train_ratio": 0.8, "val_ratio": 0.2, "seed": 42},
        }
        (normalized_dir / "normalization_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def _deduplicate_images(self, images: Sequence[Path], logger: Optional[Logger]) -> Dict[str, Path]:
        deduped: Dict[str, Path] = {}
        for path in images:
            key = path.name.lower()
            if key in deduped:
                self._log(f"[FRCNN][NORM] Ignorando duplicata (case-insensitive): {path.name}", logger)
                continue
            deduped[key] = path
        return deduped

    def _parse_annotations(
        self, csv_path: Path, images: Dict[str, Path], logger: Optional[Logger]
    ) -> Tuple[Dict[str, ImageAnnotations], List[str]]:
        annotations: Dict[str, ImageAnnotations] = {}
        class_order: List[str] = ["human"]

        with csv_path.open("r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                filename = row.get("filename", "").strip()
                if not filename:
                    continue
                key = filename.lower()
                if key not in images:
                    self._log(f"[FRCNN][NORM] Aviso: imagem no CSV não encontrada no disco: {filename}", logger)
                    continue

                xmin = self._safe_int(row.get("xmin"))
                ymin = self._safe_int(row.get("ymin"))
                xmax = self._safe_int(row.get("xmax"))
                ymax = self._safe_int(row.get("ymax"))
                if None in (xmin, ymin, xmax, ymax):
                    self._log(f"[FRCNN][NORM] Anotação ignorada por coordenadas ausentes: {row}", logger)
                    continue

                width = self._safe_int(row.get("width"))
                height = self._safe_int(row.get("height"))
                label = (row.get("class") or "human").strip() or "human"
                if label not in class_order:
                    class_order.append(label)

                img_ann = annotations.setdefault(
                    key,
                    ImageAnnotations(
                        filename=images[key].name,
                        width=width if width and width > 0 else None,
                        height=height if height and height > 0 else None,
                    ),
                )
                if width and width > 0:
                    img_ann.width = img_ann.width or width
                if height and height > 0:
                    img_ann.height = img_ann.height or height
                img_ann.boxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))
                img_ann.labels.append(label)

        return annotations, class_order

    def _build_categories(self, class_order: List[str], logger: Optional[Logger]) -> Dict[str, int]:
        categories: Dict[str, int] = {}
        next_id = 1
        for name in class_order:
            if name in categories:
                continue
            categories[name] = next_id
            next_id += 1
        mapping = ", ".join(f"{name}->{cid}" for name, cid in categories.items())
        self._log(f"[FRCNN][NORM] Mapeamento de classes: {mapping}", logger)
        return categories

    def _build_coco_split(
        self,
        files: Sequence[Path],
        destination: Path,
        annotations: Dict[str, ImageAnnotations],
        categories: Dict[str, int],
        split_name: str,
        stats: Dict[str, int],
        logger: Optional[Logger],
    ) -> Dict[str, List[Dict]]:
        images_payload: List[Dict] = []
        annotations_payload: List[Dict] = []
        image_id = 1
        annotation_id = 1

        for img_path in files:
            target_path = destination / img_path.name
            shutil.copy2(img_path, target_path)

            key = img_path.name.lower()
            ann_info = annotations.get(key)
            width, height = self._resolve_dimensions(img_path, ann_info)

            images_payload.append(
                {"id": image_id, "file_name": img_path.name, "width": width, "height": height}
            )

            if ann_info:
                for bbox, label in zip(ann_info.boxes, ann_info.labels):
                    coco_bbox = self._clamp_bbox(bbox, width, height)
                    if coco_bbox is None:
                        stats["discarded_boxes"] += 1
                        continue
                    category_id = categories.get(label)
                    if category_id is None:
                        self._log(
                            f"[FRCNN][NORM] Classe desconhecida '{label}' ignorada na imagem {img_path.name}",
                            logger,
                        )
                        stats["discarded_boxes"] += 1
                        continue
                    annotations_payload.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": list(coco_bbox),
                            "area": float(coco_bbox[2] * coco_bbox[3]),
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1
                    stats["valid_boxes"] += 1

            image_id += 1

        return {
            "images": images_payload,
            "annotations": annotations_payload,
            "categories": [{"id": cid, "name": name} for name, cid in sorted(categories.items(), key=lambda item: item[1])],
        }

    @staticmethod
    def _safe_int(value: Optional[str]) -> Optional[int]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        try:
            return int(float(value))
        except ValueError:
            return None

    def _resolve_dimensions(self, path: Path, ann_info: Optional[ImageAnnotations]) -> Tuple[int, int]:
        if ann_info and ann_info.width and ann_info.height:
            return ann_info.width, ann_info.height
        with Image.open(path) as img:
            return img.size

    @staticmethod
    def _clamp_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
        xmin, ymin, xmax, ymax = bbox
        xmin = max(0, min(xmin, width))
        ymin = max(0, min(ymin, height))
        xmax = max(0, min(xmax, width))
        ymax = max(0, min(ymax, height))
        box_width = xmax - xmin
        box_height = ymax - ymin
        if box_width <= 0 or box_height <= 0:
            return None
        return xmin, ymin, box_width, box_height

    @staticmethod
    def _list_images(root: Path) -> List[Path]:
        return [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
