from __future__ import annotations

import csv
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET

from app.detectors.base import Logger


HUMAN_CLASS_NAME = "human"


def _list_train_images(train_dir: Path) -> Tuple[List[Path], int, int]:
    allowed_suffixes = {".jpg", ".jpeg", ".png"}
    raw_images: List[Path] = []
    for path in train_dir.iterdir():
        if path.is_file() and path.suffix.lower() in allowed_suffixes:
            raw_images.append(path)

    unique_by_key: Dict[str, Path] = {}
    for path in raw_images:
        key = path.name.lower()
        unique_by_key.setdefault(key, path)

    unique_images = sorted(unique_by_key.values(), key=lambda p: p.name.lower())
    return unique_images, len(raw_images), len(unique_images)


def _probe_image_size(image_path: Path) -> Tuple[int, int]:
    try:
        from PIL import Image  # type: ignore

        with Image.open(image_path) as img:
            width, height = img.size
            return width, height
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover - robust fallback
        raise RuntimeError(f"Falha ao ler dimensões da imagem {image_path}: {exc}")

    try:  # pragma: no cover - fallback apenas se PIL indisponível
        import cv2  # type: ignore

        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Falha ao ler a imagem {image_path} com OpenCV.")
        height, width = img.shape[:2]
        return width, height
    except ImportError:
        raise RuntimeError(
            "Pillow (PIL) não está instalado e não foi possível usar OpenCV para obter dimensões; instale Pillow."
        )

    raise RuntimeError(f"Não foi possível determinar dimensões para {image_path}.")


def _resolve_image_size(
    basename: str,
    sizes_by_basename: Dict[str, Tuple[int, int]],
    image_path: Path,
    warnings: List[str],
    logger: Optional[Logger],
) -> Tuple[int, int]:
    size_from_csv = sizes_by_basename.get(basename)
    measured_width, measured_height = None, None
    try:
        measured_width, measured_height = _probe_image_size(image_path)
    except Exception as exc:
        warnings.append(str(exc))
        if logger:
            logger(f"[SSD][NORM][WARN] {exc}")

    if size_from_csv and all(dim > 0 for dim in size_from_csv):
        if measured_width and measured_height and (measured_width, measured_height) != size_from_csv:
            msg = (
                f"Tamanho divergente para {basename}: CSV={size_from_csv}, imagem={measured_width}x{measured_height}; "
                f"usando dimensões medidas."
            )
            warnings.append(msg)
            if logger:
                logger(f"[SSD][NORM][WARN] {msg}")
        if measured_width and measured_height:
            return measured_width, measured_height
        return size_from_csv

    if measured_width and measured_height:
        return measured_width, measured_height

    raise RuntimeError(f"Não foi possível determinar dimensões para {image_path}.")


def parse_heridal_csv(
    dataset_dir: Path, normalized_dir: Path, disk_basenames_lower: Set[str], logger: Optional[Logger] = None
) -> Tuple[Dict[str, List[Tuple[int, int, int, int]]], Dict[str, Tuple[int, int]], int, int, List[str], dict]:
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

    annotations_by_basename: Dict[str, List[Tuple[int, int, int, int]]] = {}
    sizes_by_basename: Dict[str, Tuple[int, int]] = {}
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
            basename_lower = basename.lower()
            unique_in_csv.add(basename)
            annotations_by_basename.setdefault(basename, [])

            if basename_lower not in disk_basenames_lower:
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
                continue

            if basename not in sizes_by_basename:
                sizes_by_basename[basename] = (width, height)

            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(xmax, width - 1)
            ymax = min(ymax, height - 1)

            if xmin >= xmax or ymin >= ymax or xmax <= 0 or ymax <= 0:
                discarded_bboxes += 1
                msg = f"BBox inválido para {basename}: ({xmin}, {ymin}, {xmax}, {ymax}); descartado."
                warnings.append(msg)
                if logger:
                    logger(f"[SSD][NORM][WARN] {msg}")
                continue

            annotations_by_basename[basename].append((xmin, ymin, xmax, ymax))
            total_bboxes += 1

    if logger:
        logger(f"[SSD][NORM] Imagens únicas no CSV: {len(annotations_by_basename)}")
        logger(f"[SSD][NORM] Imagens ausentes no disco: {len(missing_on_disk)}")
        logger(f"[SSD][NORM] BBoxes válidos: {total_bboxes}")
        logger(f"[SSD][NORM] BBoxes descartados: {discarded_bboxes}")

    audit_data = {
        "unique_in_csv": unique_in_csv,
        "missing_on_disk": missing_on_disk,
    }

    return annotations_by_basename, sizes_by_basename, total_bboxes, discarded_bboxes, warnings, audit_data


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


def _resolve_output_dir(normalized_dir: Path, logger: Optional[Logger]) -> Path:
    normalized_dir = normalized_dir.expanduser().resolve()
    generation_markers = ["JPEGImages", "Annotations", "ImageSets", "normalization_report.json"]
    has_existing_outputs = normalized_dir.exists() and any((normalized_dir / marker).exists() for marker in generation_markers)
    if has_existing_outputs:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target_normalized_dir = normalized_dir / f"run-{timestamp}"
        if logger:
            logger(
                f"[SSD][NORM][WARN] Saída existente detectada em {normalized_dir}; "
                f"escrevendo em subpasta segura {target_normalized_dir}"
            )
    else:
        target_normalized_dir = normalized_dir
    target_normalized_dir.mkdir(parents=True, exist_ok=True)
    if logger:
        logger(f"[SSD][NORM] Diretório final de saída: {target_normalized_dir}")
    return target_normalized_dir


def _validate_integrity(
    images_per_split: Dict[str, List[str]],
    stem_to_filename: Dict[str, str],
    jpeg_dir: Path,
    ann_dir: Path,
) -> Tuple[bool, List[dict]]:
    failures: List[dict] = []
    for split_name, stems in images_per_split.items():
        for stem in stems:
            filename = stem_to_filename.get(stem)
            image_path = jpeg_dir / filename if filename else None
            xml_path = ann_dir / f"{stem}.xml"
            missing_items = []
            if filename is None:
                missing_items.append("filename_lookup")
            else:
                if not image_path.exists():
                    missing_items.append("image")
                if not xml_path.exists():
                    missing_items.append("xml")
            if missing_items:
                failures.append(
                    {
                        "split": split_name,
                        "id": stem,
                        "filename": filename,
                        "missing": missing_items,
                        "image_path": str(image_path) if image_path else None,
                        "xml_path": str(xml_path),
                    }
                )
            if len(failures) >= 50:
                break
        if len(failures) >= 50:
            break
    return len(failures) == 0, failures


def normalize_heridal_to_voc(dataset_dir: Path, normalized_dir: Path, logger: Optional[Logger] = None) -> Path:
    dataset_dir = dataset_dir.expanduser().resolve()
    target_normalized_dir = _resolve_output_dir(normalized_dir, logger)

    train_images_dir = dataset_dir / "train"
    if not train_images_dir.exists():
        raise FileNotFoundError(f"Diretório de imagens do HERIDAL não encontrado: {train_images_dir}")
    disk_images, disk_images_raw_count, disk_images_unique_count = _list_train_images(train_images_dir)
    disk_basenames = [path.name for path in disk_images]
    disk_basename_by_lower = {path.name.lower(): path.name for path in disk_images}
    disk_basenames_lower = set(disk_basename_by_lower.keys())
    total_images_on_disk = len(disk_basenames)
    duplicates_removed = disk_images_raw_count - disk_images_unique_count
    if logger:
        logger(f"[SSD][NORM] Imagens no disco (train) - bruto: {disk_images_raw_count}")
        logger(f"[SSD][NORM] Imagens no disco (train) - únicas: {disk_images_unique_count}")
        if duplicates_removed:
            logger(f"[SSD][NORM][WARN] Duplicatas removidas: {duplicates_removed}")
        logger(f"[SSD][NORM] Imagens no disco (train): {total_images_on_disk}")

    (
        annotations_by_basename,
        sizes_by_basename,
        total_bboxes,
        discarded_bboxes,
        warnings,
        audit_data,
    ) = parse_heridal_csv(dataset_dir, target_normalized_dir, disk_basenames_lower, logger=logger)
    if logger:
        logger(f"[SSD][NORM] Imagens únicas no CSV: {len(annotations_by_basename)}")

    remapped_annotations: Dict[str, List[Tuple[int, int, int, int]]] = {}
    for name, boxes in annotations_by_basename.items():
        key = disk_basename_by_lower.get(name.lower(), name)
        remapped_annotations.setdefault(key, []).extend(boxes)
    annotations_by_basename = remapped_annotations

    remapped_sizes: Dict[str, Tuple[int, int]] = {}
    for name, size in sizes_by_basename.items():
        key = disk_basename_by_lower.get(name.lower(), name)
        remapped_sizes.setdefault(key, size)
    sizes_by_basename = remapped_sizes

    if not disk_basenames:
        raise ValueError("Nenhuma imagem encontrada em dataset_dir/train.")

    unique_in_csv = audit_data.get("unique_in_csv", set())
    missing_on_disk = audit_data.get("missing_on_disk", set())

    normalized_annotations: Dict[str, List[Tuple[int, int, int, int]]] = {}
    for name, boxes in annotations_by_basename.items():
        resolved_name = disk_basename_by_lower.get(name.lower())
        if resolved_name:
            normalized_annotations.setdefault(resolved_name, []).extend(boxes)

    normalized_sizes: Dict[str, Tuple[int, int]] = {}
    for name, size in sizes_by_basename.items():
        resolved_name = disk_basename_by_lower.get(name.lower())
        if resolved_name:
            normalized_sizes.setdefault(resolved_name, size)

    stem_to_filename: Dict[str, str] = {}
    seen_lower_stems: Dict[str, str] = {}
    stem_collisions: Dict[str, Set[str]] = {}
    usable_items: List[Tuple[str, str, List[Tuple[int, int, int, int]], int, int]] = []
    skipped_no_annotations: List[str] = []
    post_validation_discarded = 0
    for basename in sorted(normalized_annotations.keys()):
        stem = Path(basename).stem
        lower_stem = stem.lower()
        if lower_stem in seen_lower_stems and seen_lower_stems[lower_stem] != basename:
            stem_collisions.setdefault(stem, set()).update({seen_lower_stems[lower_stem], basename})
            warnings.append(
                f"Colisão de stem '{stem}' entre '{seen_lower_stems[lower_stem]}' e '{basename}'. "
                "Apenas a primeira ocorrência será usada para evitar IDs duplicados."
            )
            if logger:
                logger(f"[SSD][NORM][WARN] {warnings[-1]}")
            continue
        seen_lower_stems[lower_stem] = basename

        src_image = train_images_dir / basename
        width, height = _resolve_image_size(basename, normalized_sizes, src_image, warnings, logger)
        original_bboxes = normalized_annotations.get(basename, [])
        valid_bboxes: List[Tuple[int, int, int, int]] = []
        for xmin, ymin, xmax, ymax in original_bboxes:
            xmin = max(0, min(xmin, width - 1))
            ymin = max(0, min(ymin, height - 1))
            xmax = max(0, min(xmax, width - 1))
            ymax = max(0, min(ymax, height - 1))
            if xmin >= xmax or ymin >= ymax:
                post_validation_discarded += 1
                warnings.append(
                    f"BBox descartado após validação final em {basename}: ({xmin}, {ymin}, {xmax}, {ymax})."
                )
                if logger:
                    logger(f"[SSD][NORM][WARN] {warnings[-1]}")
                continue
            valid_bboxes.append((xmin, ymin, xmax, ymax))

        if not valid_bboxes:
            skipped_no_annotations.append(basename)
            continue

        stem_to_filename[stem] = basename
        usable_items.append((stem, basename, valid_bboxes, width, height))

    if not usable_items:
        raise ValueError("Nenhuma imagem válida foi encontrada no CSV que também exista no disco.")

    rng = random.Random(42)
    shuffled = usable_items[:]
    rng.shuffle(shuffled)

    total_usable_images = len(shuffled)
    train_count = int(total_usable_images * 0.8)
    val_count = total_usable_images - train_count
    train_items = shuffled[:train_count]
    val_items = shuffled[train_count:]

    images_per_split: Dict[str, List[str]] = {
        "train": [item[0] for item in train_items],
        "val": [item[0] for item in val_items],
    }

    if set(images_per_split["train"]).intersection(images_per_split["val"]):
        raise ValueError("Conjuntos de treino e validação apresentam interseção após split.")
    all_ids = images_per_split["train"] + images_per_split["val"]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("IDs duplicados detectados nos splits gerados.")

    jpeg_dir = target_normalized_dir / "JPEGImages"
    ann_dir = target_normalized_dir / "Annotations"
    imagesets_dir = target_normalized_dir / "ImageSets" / "Main"
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    imagesets_dir.mkdir(parents=True, exist_ok=True)

    images_with_annotations = len(usable_items)
    images_without_annotations = len(skipped_no_annotations)
    total_xml_written = 0
    sample_examples: List[dict] = []

    for stem, basename, valid_bboxes, width, height in usable_items:
        src_image = train_images_dir / basename
        dest_image = jpeg_dir / basename
        shutil.copy2(src_image, dest_image)
        xml_path = ann_dir / f"{stem}.xml"
        write_voc_xml(dest_image, width, height, valid_bboxes, xml_path)
        total_xml_written += 1
        if len(sample_examples) < 10:
            sample_examples.append(
                {"id": stem, "filename": basename, "xml_path": str(xml_path), "bboxes": len(valid_bboxes)}
            )

    write_imagesets(images_per_split, imagesets_dir)

    labels_path = target_normalized_dir / "labels.txt"
    labels_path.write_text(f"{HUMAN_CLASS_NAME}\n", encoding="utf-8")

    normalization_report = {
        "summary": {
            "imagens_no_disco": total_images_on_disk,
            "imagens_unicas_no_csv": len(unique_in_csv),
            "imagens_usadas_no_split": total_usable_images,
            "bboxes_total": total_bboxes,
            "bboxes_descartadas": discarded_bboxes + post_validation_discarded,
            "bboxes_descartadas_pre_validacao": discarded_bboxes,
            "bboxes_descartadas_pos_validacao": post_validation_discarded,
            "train": len(images_per_split["train"]),
            "val": len(images_per_split["val"]),
            "test": 0,
            "images_with_annotations": images_with_annotations,
            "images_without_annotations": images_without_annotations,
            "total_xml_written": total_xml_written,
        },
        "skipped_no_annotations": {
            "count": len(skipped_no_annotations),
            "examples": skipped_no_annotations[:200],
        },
        "datasets": {
            "JPEGImages": str(jpeg_dir),
            "Annotations": str(ann_dir),
            "ImageSets": str(imagesets_dir),
        },
        "warnings": warnings,
        "missing_on_disk_sample": sorted(list(missing_on_disk))[:50],
        "stem_collisions_sample": [
            {"stem": stem, "basenames": sorted(list(names))} for stem, names in stem_collisions.items()
        ],
        "samples": sample_examples,
    }

    integrity_ok, integrity_failures = _validate_integrity(images_per_split, stem_to_filename, jpeg_dir, ann_dir)
    normalization_report["integrity"] = {"ok": integrity_ok, "failures": integrity_failures}
    report_path = target_normalized_dir / "normalization_report.json"
    report_path.write_text(json.dumps(normalization_report, ensure_ascii=False, indent=2), encoding="utf-8")

    if logger:
        logger(f"[SSD][NORM] imagens_no_disco: {total_images_on_disk}")
        logger(f"[SSD][NORM] imagens_unicas_no_csv: {len(unique_in_csv)}")
        logger(f"[SSD][NORM] imagens_com_bbox: {images_with_annotations}")
        logger(f"[SSD][NORM] imagens_sem_bbox: {images_without_annotations}")
        logger(f"[SSD][NORM] imagens_usadas_no_split: {total_usable_images}")
        logger(f"[SSD][NORM] bboxes_total: {total_bboxes}")
        logger(f"[SSD][NORM] bboxes_descartadas: {discarded_bboxes + post_validation_discarded}")
        logger(f"[SSD][NORM] split -> train: {len(images_per_split['train'])}, val: {len(images_per_split['val'])}")
        if not integrity_ok:
            logger(f"[SSD][NORM][ERROR] Falha de integridade detectada; consulte {report_path}")
        else:
            logger(f"[SSD][NORM] Normalização concluída em {target_normalized_dir}")

    if not integrity_ok:
        raise RuntimeError("Falha de integridade após normalização; consulte normalization_report.json para detalhes.")

    return target_normalized_dir
