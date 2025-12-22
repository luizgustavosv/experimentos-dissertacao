from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.datasets.ir import AnnotationRecord, DatasetIR, ImageRecord
from app.datasets.utils import clip_bbox, is_image_file, load_image_size, load_json
from app.detectors.base import Logger


def _detect_splits(root: Path) -> List[Tuple[str, Path]]:
    candidates = []
    for split in ("train", "val", "test-dev", "test-challenge"):
        split_dir = root / split
        if split_dir.exists():
            candidates.append((split, split_dir))
    if candidates:
        return candidates
    if (root / "images").exists():
        return [(root.name, root)]
    return []


def _load_human_categories(config_path: Path, override: Optional[List[int]]) -> List[int]:
    if override is not None:
        return sorted(set(override))
    if not config_path.exists():
        raise FileNotFoundError(f"Arquivo de categorias não encontrado em {config_path}")
    return sorted(set(load_json(config_path)))


def _read_annotation_file(path: Path) -> List[List[int]]:
    if not path.exists():
        return []
    lines: List[List[int]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                values = [int(float(v)) for v in line.split(",")]
                if len(values) != 8:
                    continue
                lines.append(values)
            except ValueError:
                continue
    return lines


def _iter_image_files(images_dir: Path) -> List[Path]:
    return [p for p in sorted(images_dir.iterdir()) if p.is_file() and is_image_file(p)]


def _make_annotation(
    ann_id: int,
    image_id: int,
    class_id: int,
    class_name: str,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
) -> AnnotationRecord:
    area = max(0, (xmax - xmin) * (ymax - ymin))
    return AnnotationRecord(
        id=ann_id,
        image_id=image_id,
        class_id=class_id,
        class_name=class_name,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        area=area,
    )


def read_visdrone(
    dataset_dir: Path,
    human_categories: List[int],
    logger: Optional[Logger] = None,
) -> Tuple[DatasetIR, Dict[str, int], List[str], bool]:
    dataset_dir = dataset_dir.expanduser().resolve()
    splits = _detect_splits(dataset_dir)
    if not splits:
        raise FileNotFoundError(f"Não foram encontradas pastas de split em {dataset_dir}")

    images: List[ImageRecord] = []
    annotations: List[AnnotationRecord] = []
    discarded: Dict[str, int] = {
        "invalid_bbox_size": 0,
        "ignored_by_score": 0,
        "non_human_category": 0,
        "clipped_empty": 0,
        "missing_annotation_file": 0,
    }
    warnings: List[str] = []
    is_labelled = False
    human_class = "human"

    image_id = 0
    ann_id = 0
    for split_name, split_dir in splits:
        images_dir = split_dir / "images"
        annotations_dir = split_dir / "annotations"
        if not images_dir.exists():
            raise FileNotFoundError(f"Diretório de imagens não encontrado: {images_dir}")
        image_files = _iter_image_files(images_dir)
        for image_path in image_files:
            image_id += 1
            width, height = load_image_size(image_path)
            img_record = ImageRecord(
                id=image_id,
                filename=image_path.name,
                path=image_path,
                width=width,
                height=height,
                split=split_name,
            )
            images.append(img_record)

            annotation_path = annotations_dir / f"{image_path.stem}.txt"
            if not annotations_dir.exists():
                warnings.append(f"Split {split_name} não possui anotações (assumindo conjunto não rotulado).")
                continue
            if not annotation_path.exists():
                discarded["missing_annotation_file"] += 1
                warnings.append(f"Arquivo de anotação ausente para {image_path.name} em {split_name}")
                continue

            is_labelled = True
            rows = _read_annotation_file(annotation_path)
            ann_for_image = 0
            for row in rows:
                xmin, ymin, w, h, score_or_ignored, category, _, _ = row
                if w <= 0 or h <= 0:
                    discarded["invalid_bbox_size"] += 1
                    continue
                if score_or_ignored == 0:
                    discarded["ignored_by_score"] += 1
                    continue
                if category not in human_categories:
                    discarded["non_human_category"] += 1
                    continue

                xmax = xmin + w
                ymax = ymin + h
                xmin_c, ymin_c, xmax_c, ymax_c = clip_bbox(xmin, ymin, xmax, ymax, width, height)
                if xmax_c <= xmin_c or ymax_c <= ymin_c:
                    discarded["clipped_empty"] += 1
                    continue

                ann_id += 1
                ann_for_image += 1
                annotations.append(
                    _make_annotation(
                        ann_id,
                        image_id,
                        class_id=0,
                        class_name=human_class,
                        xmin=xmin_c,
                        ymin=ymin_c,
                        xmax=xmax_c,
                        ymax=ymax_c,
                    )
                )
            if ann_for_image == 0 and annotations_dir.exists():
                warnings.append(f"{image_path.name} ficou sem anotações após filtros em {split_name}")

    dataset = DatasetIR(classes=[human_class], images=images, annotations=annotations)
    return dataset, discarded, warnings, is_labelled
