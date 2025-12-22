from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

from app.datasets.ir import AnnotationRecord, DatasetIR, ImageRecord
from app.datasets.progress import NormalizationProgressBar
from app.datasets.utils import copy_image, ensure_dir
from app.detectors.base import Logger


def _annotations_by_image(dataset: DatasetIR) -> Dict[int, Iterable[AnnotationRecord]]:
    return dataset.annotations_by_image()


def _normalize_bbox(ann: AnnotationRecord, img: ImageRecord) -> str:
    img_w, img_h = img.width, img.height
    cx = ((ann.xmin + ann.xmax) / 2) / img_w
    cy = ((ann.ymin + ann.ymax) / 2) / img_h
    w = (ann.xmax - ann.xmin) / img_w
    h = (ann.ymax - ann.ymin) / img_h
    return f"{ann.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def export_yolo(dataset: DatasetIR, output_dir: Path, is_labelled: bool, logger: Optional[Logger] = None) -> Path:
    output_dir = output_dir.expanduser().resolve()
    images_root = ensure_dir(output_dir / "images")
    labels_root = ensure_dir(output_dir / "labels")

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

    dataset_yaml = {
        "path": ".",
        "names": {idx: name for idx, name in enumerate(dataset.classes)},
    }
    if "train" in split_paths:
        dataset_yaml["train"] = split_paths["train"]
    if "val" in split_paths:
        dataset_yaml["val"] = split_paths["val"]
    if "test" in split_paths:
        dataset_yaml["test"] = split_paths["test"]

    additional = {k: v for k, v in split_paths.items() if k not in {"train", "val", "test"}}
    if additional:
        dataset_yaml["additional_splits"] = additional

    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(json.dumps(dataset_yaml, indent=2), encoding="utf-8")
    if logger:
        logger(f"[YOLO] dataset.yaml salvo em {yaml_path}")
    return output_dir
