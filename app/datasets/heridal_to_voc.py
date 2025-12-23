from __future__ import annotations

import csv
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import xml.etree.ElementTree as ET

from app.detectors.base import Logger


HUMAN_CLASS_NAME = "human"


def parse_heridal_csv(
    dataset_dir: Path, normalized_dir: Path, logger: Optional[Logger] = None
) -> Tuple[Dict[str, dict], int, int, List[str]]:
    dataset_dir = dataset_dir.expanduser().resolve()
    normalized_dir = normalized_dir.expanduser().resolve()
    train_dir = dataset_dir / "train"
    annotations_path = train_dir / "annotations.csv"
    if logger:
        logger(f"[SSD][NORM] Dataset: {dataset_dir}")
        logger(f"[SSD][NORM] Saída: {normalized_dir}")
        logger(f"[SSD][NORM] CSV: {annotations_path}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"annotations.csv não encontrado em {annotations_path}")

    images: Dict[str, dict] = {}
    total_bboxes = 0
    discarded_bboxes = 0
    warnings: List[str] = []

    with annotations_path.open("r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = row.get("filename", "").strip()
            if not filename:
                discarded_bboxes += 1
                warnings.append("Linha no CSV sem filename; anotação descartada.")
                continue
            image_path = train_dir / filename
            if not image_path.exists():
                discarded_bboxes += 1
                msg = f"Imagem ausente referenciada no CSV: {filename}"
                warnings.append(msg)
                if logger:
                    logger(f"[SSD][NORM][WARN] {msg}")
                continue

            width = int(row["width"])
            height = int(row["height"])
            xmin = int(row["xmin"])
            ymin = int(row["ymin"])
            xmax = int(row["xmax"])
            ymax = int(row["ymax"])

            xmin = max(0, min(xmin, width - 1))
            ymin = max(0, min(ymin, height - 1))
            xmax = max(0, min(xmax, width - 1))
            ymax = max(0, min(ymax, height - 1))

            if xmin >= xmax or ymin >= ymax:
                discarded_bboxes += 1
                msg = f"BBox inválido para {filename}: ({xmin}, {ymin}, {xmax}, {ymax}); descartado."
                warnings.append(msg)
                if logger:
                    logger(f"[SSD][NORM][WARN] {msg}")
                continue

            image_data = images.setdefault(
                filename,
                {
                    "path": image_path,
                    "width": width,
                    "height": height,
                    "bboxes": [],
                },
            )
            image_data["bboxes"].append((xmin, ymin, xmax, ymax))
            total_bboxes += 1

    if logger:
        logger(f"[SSD][NORM] Imagens únicas: {len(images)}")
        logger(f"[SSD][NORM] BBoxes válidos: {total_bboxes}")
        logger(f"[SSD][NORM] BBoxes descartados: {discarded_bboxes}")

    return images, total_bboxes, discarded_bboxes, warnings


def make_split(
    filenames: Iterable[str], train_ratio: float = 0.8, val_ratio: float = 0.2, test_ratio: float = 0.0, seed: int = 42
) -> Dict[str, str]:
    names = list(filenames)
    rng = random.Random(seed)
    rng.shuffle(names)

    total = len(names)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = int(total * test_ratio)
    assigned = train_count + val_count + test_count
    remainder = total - assigned
    train_count += remainder  # envia qualquer sobra para treino

    splits: Dict[str, str] = {}
    for idx, name in enumerate(names):
        if idx < train_count:
            splits[name] = "train"
        elif idx < train_count + val_count:
            splits[name] = "val"
        else:
            splits[name] = "test"
    return splits


def write_voc_xml(
    image_path: Path,
    width: int,
    height: int,
    bboxes: List[Tuple[int, int, int, int]],
    output_path: Path,
) -> None:
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = image_path.parent.name
    ET.SubElement(annotation, "filename").text = image_path.name
    ET.SubElement(annotation, "path").text = str(image_path.resolve())

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"

    for xmin, ymin, xmax, ymax in bboxes:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = HUMAN_CLASS_NAME
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    tree = ET.ElementTree(annotation)
    tree.write(output_path, encoding="utf-8")


def write_imagesets(images_per_split: Dict[str, List[str]], imagesets_dir: Path) -> None:
    imagesets_dir.mkdir(parents=True, exist_ok=True)
    for split_name, filenames in images_per_split.items():
        split_path = imagesets_dir / f"{split_name}.txt"
        with split_path.open("w", encoding="utf-8") as fh:
            for fname in sorted(filenames):
                fh.write(f"{Path(fname).stem}\n")


def normalize_heridal_to_voc(dataset_dir: Path, normalized_dir: Path, logger: Optional[Logger] = None) -> Path:
    dataset_dir = dataset_dir.expanduser().resolve()
    normalized_dir = normalized_dir.expanduser().resolve()

    images, total_bboxes, discarded_bboxes, warnings = parse_heridal_csv(
        dataset_dir, normalized_dir, logger=logger
    )
    if not images:
        raise ValueError("Nenhuma imagem válida encontrada no CSV do HERIDAL.")

    splits = make_split(images.keys())
    if logger:
        train_count = sum(1 for split in splits.values() if split == "train")
        val_count = sum(1 for split in splits.values() if split == "val")
        test_count = sum(1 for split in splits.values() if split == "test")
        logger(f"[SSD][NORM] Split -> train: {train_count}, val: {val_count}, test: {test_count}")

    image_root = normalized_dir / "images"
    ann_root = normalized_dir / "annotations_voc"
    imagesets_root = normalized_dir / "ImageSets" / "Main"

    for split in ["train", "val", "test"]:
        (image_root / split).mkdir(parents=True, exist_ok=True)
        (ann_root / split).mkdir(parents=True, exist_ok=True)
    imagesets_root.mkdir(parents=True, exist_ok=True)

    images_per_split: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    for filename, data in images.items():
        split = splits.get(filename, "train")
        dest_image = image_root / split / filename
        shutil.copy2(data["path"], dest_image)
        write_voc_xml(
            dest_image,
            data["width"],
            data["height"],
            data["bboxes"],
            ann_root / split / f"{Path(filename).stem}.xml",
        )
        images_per_split.setdefault(split, []).append(filename)

    write_imagesets(images_per_split, imagesets_root)

    labels_path = normalized_dir / "labels.txt"
    labels_path.write_text(f"{HUMAN_CLASS_NAME}\n", encoding="utf-8")

    metadata = {
        "origem": "HERIDAL",
        "formato": "VOC",
        "classes": [HUMAN_CLASS_NAME],
        "timestamp": datetime.now().isoformat(),
        "bboxes_total": total_bboxes,
        "bboxes_descartados": discarded_bboxes,
        "avisos": warnings,
    }
    (normalized_dir / "dataset.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    if logger:
        logger(f"[SSD][NORM] Normalização concluída em {normalized_dir}")

    return normalized_dir
