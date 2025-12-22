from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from app.datasets.ir import AnnotationRecord, DatasetIR, ImageRecord
from app.datasets.utils import copy_image, ensure_dir
from app.detectors.base import Logger


def _coco_image_dict(img: ImageRecord, split: str) -> Dict:
    return {"id": img.id, "file_name": f"{split}/{img.filename}", "width": img.width, "height": img.height}


def _coco_annotation_dict(ann: AnnotationRecord) -> Dict:
    width = ann.xmax - ann.xmin
    height = ann.ymax - ann.ymin
    return {
        "id": ann.id,
        "image_id": ann.image_id,
        "category_id": ann.class_id + 1,
        "bbox": [ann.xmin, ann.ymin, width, height],
        "area": ann.area,
        "iscrowd": 0,
        "segmentation": [],
    }


def export_coco(dataset: DatasetIR, output_dir: Path, is_labelled: bool, logger: Optional[Logger] = None) -> Path:
    output_dir = output_dir.expanduser().resolve()
    images_root = ensure_dir(output_dir / "images")
    ann_by_img = dataset.annotations_by_image()
    ann_by_split = dataset.annotations_by_split()

    for split, imgs in dataset.images_by_split().items():
        split_dir = ensure_dir(images_root / split)
        for img in imgs:
            copy_image(img.path, split_dir / img.filename)
            if logger:
                logger(f"[COCO] Imagem copiada: {img.filename} ({split})")

        annotations = ann_by_split.get(split, [])
        if not is_labelled and not annotations:
            continue

        data = {
            "images": [_coco_image_dict(img, split) for img in imgs],
            "annotations": [_coco_annotation_dict(ann) for ann in annotations],
            "categories": [{"id": idx + 1, "name": name} for idx, name in enumerate(dataset.classes)],
        }
        ann_path = output_dir / f"{split}.json"
        ann_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        if logger:
            logger(f"[COCO] Anotações salvas em {ann_path}")

    return output_dir
