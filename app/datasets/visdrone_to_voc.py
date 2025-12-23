from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.datasets.utils import ensure_dir, is_image_file, load_image_size
from app.detectors.base import Logger


VISDRONE_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]
VALID_CLASS_IDS = set(range(1, len(VISDRONE_CLASSES) + 1))


def _classify_split_from_name(name: str) -> Optional[str]:
    lower = name.lower()
    if "train" in lower:
        return "train"
    if "val" in lower:
        return "val"
    if "test" in lower:
        return "test"
    return None


def _iter_image_files(images_dir: Path) -> List[Path]:
    return [p for p in sorted(images_dir.iterdir()) if p.is_file() and is_image_file(p)]


def find_visdrone_splits(dataset_dir: Path) -> Dict[str, Optional[Tuple[Path, Path]]]:
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Diretório do dataset não encontrado: {dataset_dir}")
    splits: Dict[str, Optional[Tuple[Path, Path]]] = {"train": None, "val": None, "test": None}

    def register_split(split_name: str, base_dir: Path) -> None:
        images_dir = base_dir / "images"
        annotations_dir = base_dir / "annotations"
        if not images_dir.exists():
            return
        splits[split_name] = (images_dir, annotations_dir)

    for child in dataset_dir.iterdir():
        if not child.is_dir():
            continue
        split_name = _classify_split_from_name(child.name)
        if child.name.startswith("VisDrone2019-DET-") and split_name:
            register_split(split_name, child)
        elif split_name:
            register_split(split_name, child)

    if splits["train"] is None and splits["val"] is None and (dataset_dir / "images").exists():
        register_split("train", dataset_dir)

    return splits


def parse_visdrone_annotation(
    txt_path: Path, img_w: int, img_h: int
) -> Tuple[List[Tuple[str, int, int, int, int]], int, int]:
    results: List[Tuple[str, int, int, int, int]] = []
    ignored = 0
    discarded = 0
    if not txt_path.exists():
        return results, ignored, discarded

    with txt_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                discarded += 1
                continue
            try:
                x = int(float(parts[0]))
                y = int(float(parts[1]))
                w = int(float(parts[2]))
                h = int(float(parts[3]))
                score = float(parts[4])
                class_id = int(float(parts[5]))
            except ValueError:
                discarded += 1
                continue

            if score == 0:
                ignored += 1
                continue
            if w <= 0 or h <= 0:
                discarded += 1
                continue
            if class_id not in VALID_CLASS_IDS:
                discarded += 1
                continue

            xmin = max(0, x)
            ymin = max(0, y)
            xmax = min(x + w, img_w)
            ymax = min(y + h, img_h)
            if xmax <= xmin or ymax <= ymin:
                discarded += 1
                continue

            cls_name = VISDRONE_CLASSES[class_id - 1]
            results.append((cls_name, xmin, ymin, xmax, ymax))

    return results, ignored, discarded


def _write_pascal_voc_xml(
    filename: str,
    width: int,
    height: int,
    objects: List[Tuple[str, int, int, int, int]],
    output_path: Path,
) -> None:
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "VOC2007"
    ET.SubElement(annotation, "filename").text = filename

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"

    for cls_name, xmin, ymin, xmax, ymax in objects:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = cls_name
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


def _build_split_lists(
    items: List[Tuple[str, str]],  # (filename, split)
    train_exists: bool,
    val_exists: bool,
    seed: int = 42,
) -> Dict[str, List[str]]:
    if train_exists and val_exists:
        result: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        for fname, split in items:
            if split in result:
                result.setdefault(split, []).append(fname)
        return result

    rng = random.Random(seed)
    names = [fname for fname, _ in items]
    rng.shuffle(names)
    total = len(names)
    split_idx = int(total * 0.8)
    return {"train": names[:split_idx], "val": names[split_idx:], "test": []}


def normalize_visdrone_to_voc(dataset_dir: Path, normalized_dir: Path, logger: Optional[Logger] = None) -> Path:
    dataset_dir = dataset_dir.expanduser().resolve()
    normalized_dir = normalized_dir.expanduser().resolve()

    splits = find_visdrone_splits(dataset_dir)
    if logger:
        logger(
            "[SSD][NORM] VisDrone splits detectados: "
            + ", ".join(f"{k}={v[0].parent if v else None}" for k, v in splits.items())
        )

    voc_root = normalized_dir / "VOC2007"
    images_dir = ensure_dir(voc_root / "JPEGImages")
    annotations_dir = ensure_dir(voc_root / "Annotations")
    imagesets_dir = ensure_dir(voc_root / "ImageSets" / "Main")

    stats = {
        "images_found": {"train": 0, "val": 0, "test": 0},
        "images_copied": 0,
        "xml_generated": 0,
        "bboxes_valid": 0,
        "bboxes_discarded": 0,
        "lines_ignored": 0,
        "missing_annotations": 0,
        "duplicates_skipped": 0,
    }

    gathered: List[Tuple[Path, str, Path]] = []  # (image_path, split, annotation_dir)
    seen_lower: Dict[str, Path] = {}

    for split_name in ("train", "val", "test"):
        split_paths = splits.get(split_name)
        if split_paths is None:
            continue
        img_dir, ann_dir = split_paths
        image_files = _iter_image_files(img_dir)
        stats["images_found"][split_name] = len(image_files)
        for img_path in image_files:
            key = img_path.name.lower()
            if key in seen_lower:
                stats["duplicates_skipped"] += 1
                continue
            seen_lower[key] = img_path
            gathered.append((img_path, split_name, ann_dir))

    if not gathered:
        raise ValueError("Nenhuma imagem VisDrone encontrada para normalização.")

    train_exists = splits.get("train") is not None
    val_exists = splits.get("val") is not None
    filenames_for_split = _build_split_lists([(img.name, split) for img, split, _ in gathered], train_exists, val_exists)
    train_set = set(filenames_for_split.get("train", []))
    val_set = set(filenames_for_split.get("val", []))
    test_set = set(filenames_for_split.get("test", []))

    for img_path, _, ann_dir in gathered:
        if img_path.name in train_set:
            assigned_split = "train"
        elif img_path.name in val_set:
            assigned_split = "val"
        elif img_path.name in test_set:
            assigned_split = "test"
        else:
            assigned_split = "val"
        dest_image_path = images_dir / img_path.name
        shutil.copy2(img_path, dest_image_path)
        stats["images_copied"] += 1

        width, height = load_image_size(img_path)
        ann_path = ann_dir / f"{img_path.stem}.txt"
        objects, ignored, discarded = parse_visdrone_annotation(ann_path, width, height)
        stats["lines_ignored"] += ignored
        stats["bboxes_discarded"] += discarded
        if not ann_path.exists():
            stats["missing_annotations"] += 1
        stats["bboxes_valid"] += len(objects)

        _write_pascal_voc_xml(img_path.name, width, height, objects, annotations_dir / f"{img_path.stem}.xml")
        stats["xml_generated"] += 1

        split_file = imagesets_dir / f"{assigned_split}.txt"
        with split_file.open("a", encoding="utf-8") as fh:
            fh.write(f"{img_path.stem}\n")

    labels_path = voc_root / "labels.txt"
    labels_path.write_text("\n".join(VISDRONE_CLASSES) + "\n", encoding="utf-8")

    if logger:
        logger(
            "[SSD][NORM] Contagem de imagens por split de origem: "
            + ", ".join(f"{k}={v}" for k, v in stats["images_found"].items())
        )
        logger(f"[SSD][NORM] Imagens copiadas: {stats['images_copied']}")
        logger(f"[SSD][NORM] XML gerados: {stats['xml_generated']}")
        logger(f"[SSD][NORM] BBoxes válidos: {stats['bboxes_valid']}")
        logger(f"[SSD][NORM] BBoxes descartados: {stats['bboxes_discarded']}")
        logger(f"[SSD][NORM] Linhas ignoradas (score==0/ignored region): {stats['lines_ignored']}")
        logger(f"[SSD][NORM] Anotações ausentes: {stats['missing_annotations']}")
        if stats["duplicates_skipped"]:
            logger(f"[SSD][NORM][WARN] Imagens duplicadas ignoradas (case-insensitive): {stats['duplicates_skipped']}")
        logger(f"[SSD][NORM] Dataset VOC pronto em {voc_root}")

    return voc_root
