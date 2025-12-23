from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET

from app.detectors.base import Logger


HUMAN_CLASS_NAME = "human"


def _list_train_images(train_dir: Path) -> Set[str]:
    patterns = ["*.jpg", "*.JPG", "*.jpeg", "*.png"]
    disk_images: Set[str] = set()
    for pattern in patterns:
        for path in train_dir.glob(pattern):
            disk_images.add(path.name)
    return disk_images


def parse_heridal_csv(
    dataset_dir: Path, normalized_dir: Path, logger: Optional[Logger] = None
) -> Tuple[Dict[str, dict], int, int, List[str], dict]:
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

    disk_images = _list_train_images(train_dir)
    total_images_on_disk = len(disk_images)
    if logger:
        logger(f"[SSD][NORM] Imagens no disco (train): {total_images_on_disk}")

    images: Dict[str, dict] = {}
    total_bboxes = 0
    discarded_bboxes = 0
    warnings: List[str] = []
    unique_in_csv: Set[str] = set()
    missing_on_disk: Set[str] = set()

    with annotations_path.open("r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            raw_filename = row.get("filename", "")
            filename = raw_filename.strip()
            if not filename:
                discarded_bboxes += 1
                warnings.append("Linha no CSV sem filename; anotação descartada.")
                continue

            filename = filename.replace("\\", "/")
            basename = Path(filename).name
            unique_in_csv.add(basename)
            expected_path = train_dir / basename
            if basename not in disk_images:
                discarded_bboxes += 1
                missing_on_disk.add(basename)
                msg = f"Imagem ausente referenciada no CSV: {basename}"
                warnings.append(msg)
                if logger:
                    logger(f"[SSD][NORM][WARN] {msg}")
                continue

            try:
                width = int(row["width"])
                height = int(row["height"])
                xmin = int(row["xmin"])
                ymin = int(row["ymin"])
                xmax = int(row["xmax"])
                ymax = int(row["ymax"])
            except (TypeError, ValueError, KeyError) as exc:
                discarded_bboxes += 1
                msg = f"Falha ao ler campos numéricos para {basename}: {exc}"
                warnings.append(msg)
                if logger:
                    logger(f"[SSD][NORM][WARN] {msg}")
                images.setdefault(
                    basename,
                    {
                        "path": expected_path,
                        "width": None,
                        "height": None,
                        "bboxes": [],
                    },
                )
                continue

            image_data = images.setdefault(
                basename,
                {
                    "path": expected_path,
                    "width": width,
                    "height": height,
                    "bboxes": [],
                },
            )
            if image_data["width"] is None:
                image_data["width"] = width
            if image_data["height"] is None:
                image_data["height"] = height

            xmin = max(0, min(xmin, width - 1))
            ymin = max(0, min(ymin, height - 1))
            xmax = max(0, min(xmax, width - 1))
            ymax = max(0, min(ymax, height - 1))

            if xmin >= xmax or ymin >= ymax:
                discarded_bboxes += 1
                msg = f"BBox inválido para {basename}: ({xmin}, {ymin}, {xmax}, {ymax}); descartado."
                warnings.append(msg)
                if logger:
                    logger(f"[SSD][NORM][WARN] {msg}")
                continue

            image_data["bboxes"].append((xmin, ymin, xmax, ymax))
            total_bboxes += 1

    skipped_no_valid_bbox = {name for name, data in images.items() if not data["bboxes"]}

    if logger:
        logger(f"[SSD][NORM] Imagens únicas no CSV: {len(unique_in_csv)}")
        logger(f"[SSD][NORM] Imagens encontradas no disco: {len(images)}")
        logger(f"[SSD][NORM] Imagens ausentes no disco: {len(missing_on_disk)}")
        logger(f"[SSD][NORM] Imagens com zero bboxes válidas: {len(skipped_no_valid_bbox)}")
        logger(f"[SSD][NORM] BBoxes válidos: {total_bboxes}")
        logger(f"[SSD][NORM] BBoxes descartados: {discarded_bboxes}")

    audit_data = {
        "total_images_on_disk": total_images_on_disk,
        "unique_in_csv": unique_in_csv,
        "missing_on_disk": missing_on_disk,
        "skipped_no_valid_bbox": skipped_no_valid_bbox,
    }

    return images, total_bboxes, discarded_bboxes, warnings, audit_data


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

    (
        images,
        total_bboxes,
        discarded_bboxes,
        warnings,
        audit_data,
    ) = parse_heridal_csv(dataset_dir, normalized_dir, logger=logger)
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
    collisions: Dict[str, Set[str]] = {}
    map_stem_to_basename: Dict[str, str] = {}
    used_output_stems: Set[str] = set()

    for basename, data in images.items():
        split = splits.get(basename, "train")
        stem = Path(basename).stem
        output_stem = stem
        if stem in map_stem_to_basename and map_stem_to_basename[stem] != basename:
            collisions.setdefault(stem, set()).update({map_stem_to_basename[stem], basename})
            hash_suffix = hashlib.md5(basename.encode("utf-8")).hexdigest()[:8]
            output_stem = f"{stem}_{hash_suffix}"
            if logger:
                logger(f"[SSD][NORM][WARN] Colisão de stem '{stem}' entre '{map_stem_to_basename[stem]}' e '{basename}'. Renomeando para '{output_stem}'.")
        map_stem_to_basename.setdefault(stem, basename)
        if output_stem in used_output_stems:
            extra_hash = hashlib.md5(f"{basename}-{output_stem}".encode("utf-8")).hexdigest()[:8]
            output_stem = f"{output_stem}_{extra_hash}"
            if logger:
                logger(f"[SSD][NORM][WARN] Stem resultante duplicado; ajustado para '{output_stem}'.")
        used_output_stems.add(output_stem)

        output_filename = f"{output_stem}{Path(basename).suffix}"
        dest_image = image_root / split / output_filename
        shutil.copy2(data["path"], dest_image)
        width = data.get("width") or 0
        height = data.get("height") or 0
        write_voc_xml(
            dest_image,
            width,
            height,
            data["bboxes"],
            ann_root / split / f"{output_stem}.xml",
        )
        if logger and not data["bboxes"]:
            logger(f"[SSD][NORM][WARN] XML gerado sem bboxes válidas para {basename}.")
        images_per_split.setdefault(split, []).append(output_filename)

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

    skipped_no_valid_bbox = audit_data.get("skipped_no_valid_bbox", set())
    missing_on_disk = audit_data.get("missing_on_disk", set())
    unique_in_csv = audit_data.get("unique_in_csv", set())
    total_images_on_disk = audit_data.get("total_images_on_disk", 0)
    summary_log = [
        f"[SSD][NORM] total_images_on_disk: {total_images_on_disk}",
        f"[SSD][NORM] count_unique_in_csv: {len(unique_in_csv)}",
        f"[SSD][NORM] count_found_on_disk: {len(images)}",
        f"[SSD][NORM] count_missing_on_disk: {len(missing_on_disk)}",
        f"[SSD][NORM] count_images_with_zero_valid_boxes: {len(skipped_no_valid_bbox)}",
        f"[SSD][NORM] count_collisions: {len(collisions)}",
    ]
    if logger:
        for msg in summary_log:
            logger(msg)

    normalization_report = {
        "summary": {
            "total_images_on_disk": total_images_on_disk,
            "count_unique_in_csv": len(unique_in_csv),
            "count_found_on_disk": len(images),
            "count_missing_on_disk": len(missing_on_disk),
            "count_images_with_zero_valid_boxes": len(skipped_no_valid_bbox),
            "count_collisions": len(collisions),
            "bboxes_total": total_bboxes,
            "bboxes_descartados": discarded_bboxes,
        },
        "missing_on_disk_sample": sorted(list(missing_on_disk))[:200],
        "collisions_sample": [
            {"stem": stem, "basenames": sorted(list(names))}
            for stem, names in list(collisions.items())[:200]
        ],
    }
    report_path = normalized_dir / "normalization_report.json"
    report_path.write_text(json.dumps(normalization_report, ensure_ascii=False, indent=2), encoding="utf-8")

    if logger:
        logger(f"[SSD][NORM] Normalização concluída em {normalized_dir}")

    return normalized_dir
