from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.datasets.ir import AnnotationRecord, DatasetIR, ImageRecord
from app.datasets.split import SplitRatios, assign_splits
from app.datasets.utils import is_image_file
from app.detectors.base import Logger


def _parse_row(row: Dict[str, str]) -> Tuple[str, int, int, int, int, int, int]:
    filename = row["filename"]
    width = int(row["width"])
    height = int(row["height"])
    xmin = int(row["xmin"])
    ymin = int(row["ymin"])
    xmax = int(row["xmax"])
    ymax = int(row["ymax"])
    return filename, width, height, xmin, ymin, xmax, ymax


def read_heridal(
    dataset_dir: Path,
    split_ratios: SplitRatios = (0.8, 0.1, 0.1),
    seed: int = 42,
    logger: Optional[Logger] = None,
) -> Tuple[DatasetIR, Dict[str, int], List[str], bool]:
    dataset_dir = dataset_dir.expanduser().resolve()
    train_dir = dataset_dir / "train"
    annotations_path = train_dir / "_annotations.csv"
    if not annotations_path.exists():
        raise FileNotFoundError(f"_annotations.csv não encontrado em {annotations_path}")

    images: Dict[str, ImageRecord] = {}
    annotations: List[AnnotationRecord] = []
    discarded: Dict[str, int] = {"non_human_class": 0}
    warnings: List[str] = []
    human_class = "human"
    class_id = 0

    with annotations_path.open("r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get("class") != human_class:
                discarded["non_human_class"] += 1
                continue
            filename, width, height, xmin, ymin, xmax, ymax = _parse_row(row)
            image_path = train_dir / filename
            if not image_path.exists() or not is_image_file(image_path):
                warnings.append(f"Imagem ausente ou inválida referenciada no CSV: {filename}")
                continue
            if filename not in images:
                image_id = len(images) + 1
                images[filename] = ImageRecord(
                    id=image_id,
                    filename=filename,
                    path=image_path,
                    width=width,
                    height=height,
                    split="train",
                )
            image_id = images[filename].id
            area = max(0, (xmax - xmin) * (ymax - ymin))
            ann_id = len(annotations) + 1
            annotations.append(
                AnnotationRecord(
                    id=ann_id,
                    image_id=image_id,
                    class_id=class_id,
                    class_name=human_class,
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    area=area,
                )
            )

    split_assignments = assign_splits(sorted(images.keys()), split_ratios, seed=seed)
    for img in images.values():
        img.split = split_assignments.get(img.filename, "train")

    dataset = DatasetIR(classes=[human_class], images=list(images.values()), annotations=annotations)
    return dataset, discarded, warnings, True
