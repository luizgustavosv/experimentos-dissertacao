from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml

from app.datasets.ir import AnnotationRecord, DatasetIR, ImageRecord
from app.datasets.progress import NormalizationProgressBar
from app.datasets.utils import copy_image, ensure_dir
from app.detectors.base import Logger


def _annotations_by_image(dataset: DatasetIR) -> Dict[int, Iterable[AnnotationRecord]]:
    return dataset.annotations_by_image()


def _normalize_bbox(ann: AnnotationRecord, img: ImageRecord) -> str:
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    img_w, img_h = img.width, img.height
    cx = _clamp(((ann.xmin + ann.xmax) / 2) / img_w)
    cy = _clamp(((ann.ymin + ann.ymax) / 2) / img_h)
    w = _clamp((ann.xmax - ann.xmin) / img_w)
    h = _clamp((ann.ymax - ann.ymin) / img_h)
    return f"{ann.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def export_yolo(dataset: DatasetIR, output_dir: Path, is_labelled: bool, logger: Optional[Logger] = None) -> Path:
    output_dir = output_dir.expanduser().resolve()
    images_root = ensure_dir(output_dir / "images")
    labels_root = ensure_dir(output_dir / "labels")
    for split in ("train", "val", "test"):
        ensure_dir(images_root / split)
        ensure_dir(labels_root / split)

    progress = NormalizationProgressBar(total=len(dataset.images), logger=logger)

    ann_by_img = _annotations_by_image(dataset)
    split_paths: Dict[str, str] = {}
    for img in dataset.images:
        split_img_dir = ensure_dir(images_root / img.split)
        split_lbl_dir = ensure_dir(labels_root / img.split)
        copy_image(img.path, split_img_dir / img.filename)
        split_paths[img.split] = f"images/{img.split}"

        label_lines = []
        if is_labelled:
            anns = ann_by_img.get(img.id, [])
            label_lines = [_normalize_bbox(ann, img) for ann in anns]
        (split_lbl_dir / f"{Path(img.filename).stem}.txt").write_text("\n".join(label_lines), encoding="utf-8")
        if logger:
            logger(f"[YOLO] Exportado rótulo para {img.filename} ({img.split})")
        progress.advance()

    progress.finish()

    train_images = images_root / "train"
    train_labels = labels_root / "train"
    if not train_images.exists() or not train_labels.exists():
        raise ValueError(
            f"Dataset YOLO normalizado inválido: pastas obrigatórias não encontradas ({train_images}, {train_labels})"
        )

    dataset_yaml = {
        "path": output_dir.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(dataset.classes),
        "names": {idx: name for idx, name in enumerate(dataset.classes)},
    }
    yaml_path = output_dir / "dataset.yaml"
    yaml.safe_dump(dataset_yaml, yaml_path.open("w", encoding="utf-8"), sort_keys=False)
    if logger:
        logger(f"[YOLO] dataset.yaml salvo em {yaml_path}")
    return output_dir
