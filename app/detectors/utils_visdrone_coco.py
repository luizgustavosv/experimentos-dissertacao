from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from app.detectors.base import Logger


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VISDRONE_CATEGORIES: Sequence[Tuple[int, str]] = (
    (1, "pedestrian"),
    (2, "people"),
    (3, "bicycle"),
    (4, "car"),
    (5, "van"),
    (6, "truck"),
    (7, "tricycle"),
    (8, "awning-tricycle"),
    (9, "bus"),
    (10, "motor"),
)


@dataclass
class CocoSplitArtifacts:
    split: str
    images_json: Path
    images_dir: Path
    images_copied: int = 0
    images_with_annotations: int = 0
    missing_annotations: int = 0
    ignored_regions: int = 0
    bboxes_valid: int = 0
    bboxes_discarded: int = 0


def _infer_split_name(name: str) -> Optional[str]:
    lower = name.lower()
    if "train" in lower:
        return "train"
    if "val" in lower:
        return "val"
    if "test-dev" in lower or "test_dev" in lower or "test" in lower:
        return "test"
    return None


def _iter_images(images_dir: Path) -> Iterable[Path]:
    return (p for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def find_visdrone_splits(dataset_dir: Path) -> Dict[str, Optional[Tuple[Path, Path]]]:
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Diretório do dataset não encontrado: {dataset_dir}")

    splits: Dict[str, Optional[Tuple[Path, Path]]] = {"train": None, "val": None, "test": None}
    candidates: List[Path] = []

    if (dataset_dir / "images").exists() and "challenge" not in dataset_dir.name.lower():
        candidates.append(dataset_dir)
    for child in dataset_dir.iterdir():
        if child.is_dir() and "challenge" not in child.name.lower() and (
            child.name.startswith("VisDrone2019-DET-") or _infer_split_name(child.name)
        ):
            candidates.append(child)

    if not candidates and dataset_dir.name.startswith("VisDrone2019-DET-") and "challenge" not in dataset_dir.name.lower():
        candidates.append(dataset_dir)

    if not candidates:
        return splits

    errors: List[str] = []

    def register(split_name: str, base_dir: Path) -> None:
        images_dir = base_dir / "images"
        annotations_dir = base_dir / "annotations"
        missing: List[Path] = []
        if not images_dir.exists():
            missing.append(images_dir)
        if split_name in {"train", "val"} and not annotations_dir.exists():
            missing.append(annotations_dir)
        if missing:
            errors.append(f"{split_name}: faltando {', '.join(str(p) for p in missing)}")
            return
        splits[split_name] = (images_dir, annotations_dir)

    for candidate in candidates:
        split_name = _infer_split_name(candidate.name) or "train"
        register(split_name, candidate)

    if errors:
        raise FileNotFoundError(
            "Estrutura VisDrone inválida. Procure por pastas com images/ e annotations/. "
            f"Problemas encontrados: {', '.join(errors)}"
        )

    return splits


def parse_visdrone_txt(txt_path: Path, img_w: int, img_h: int) -> Tuple[List[Tuple[int, int, int, int, int]], int, int]:
    """Retorna (class_id, xmin, ymin, w, h), linhas ignoradas, bboxes descartados."""
    boxes: List[Tuple[int, int, int, int, int]] = []
    ignored = 0
    discarded = 0
    if not txt_path.exists():
        return boxes, ignored, discarded

    with txt_path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 6:
                discarded += 1
                continue
            try:
                x = int(float(row[0]))
                y = int(float(row[1]))
                w = int(float(row[2]))
                h = int(float(row[3]))
                score = str(row[4]).strip()
                class_id = int(float(row[5]))
            except (ValueError, TypeError):
                discarded += 1
                continue

            if score == "0":
                ignored += 1
                continue

            xmin = max(0, x)
            ymin = max(0, y)
            xmax = min(x + w, img_w)
            ymax = min(y + h, img_h)
            width = xmax - xmin
            height = ymax - ymin
            if width <= 0 or height <= 0:
                discarded += 1
                continue
            if class_id < 1 or class_id > len(VISDRONE_CATEGORIES):
                discarded += 1
                continue

            boxes.append((class_id, xmin, ymin, width, height))

    return boxes, ignored, discarded


def copy_images(img_dir: Path, out_images_dir: Path, selected: Optional[set[str]] = None) -> List[Path]:
    out_images_dir.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    for img_path in _iter_images(img_dir):
        if selected is not None and img_path.name not in selected:
            continue
        target_path = out_images_dir / img_path.name
        shutil.copy2(img_path, target_path)
        copied.append(img_path)
    return copied


def build_coco_for_split(
    split_name: str,
    img_dir: Path,
    ann_dir: Optional[Path],
    out_images_dir: Path,
    out_json_path: Path,
    logger: Optional[Logger],
    selected: Optional[set[str]] = None,
) -> CocoSplitArtifacts:
    artifacts = CocoSplitArtifacts(split=split_name, images_json=out_json_path, images_dir=out_images_dir)
    images_payload: List[Dict] = []
    annotations_payload: List[Dict] = []
    annotation_id = 1

    copied_images = copy_images(img_dir, out_images_dir, selected)
    artifacts.images_copied = len(copied_images)

    for image_id, img_path in enumerate(copied_images, start=1):
        with Image.open(img_path) as img:
            width, height = img.size

        coco_file_name = f"images/{split_name}/{img_path.name}"
        images_payload.append({"id": image_id, "file_name": coco_file_name, "width": width, "height": height})

        if ann_dir:
            ann_path = ann_dir / f"{img_path.stem}.txt"
            if ann_path.exists():
                parsed, ignored, discarded = parse_visdrone_txt(ann_path, width, height)
                artifacts.ignored_regions += ignored
                artifacts.bboxes_discarded += discarded
                if parsed:
                    artifacts.images_with_annotations += 1
                for class_id, xmin, ymin, w, h in parsed:
                    annotations_payload.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id,
                            "bbox": [xmin, ymin, w, h],
                            "area": float(w * h),
                            "iscrowd": 0,
                        }
                    )
                    artifacts.bboxes_valid += 1
                    annotation_id += 1
            else:
                artifacts.missing_annotations += 1

    categories = [{"id": cid, "name": name} for cid, name in VISDRONE_CATEGORIES]
    coco_payload = {"images": images_payload, "annotations": annotations_payload, "categories": categories}
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(coco_payload, indent=2), encoding="utf-8")
    if logger:
        logger(
            f"[VisDrone][COCO] Split {split_name}: imgs={artifacts.images_copied}, "
            f"com anotações={artifacts.images_with_annotations}, ignoradas={artifacts.ignored_regions}, "
            f"bboxes válidos={artifacts.bboxes_valid}, descartados={artifacts.bboxes_discarded}, "
            f"anotações ausentes={artifacts.missing_annotations}"
        )

    return artifacts


def normalize_visdrone_to_coco(dataset_dir: Path, normalized_dir: Path, logger: Optional[Logger] = None) -> Dict[str, CocoSplitArtifacts]:
    dataset_dir = dataset_dir.expanduser().resolve()
    normalized_dir = normalized_dir.expanduser().resolve()
    images_root = normalized_dir / "images"
    annotations_root = normalized_dir / "annotations"
    annotations_root.mkdir(parents=True, exist_ok=True)

    splits = find_visdrone_splits(dataset_dir)
    available = {name: paths for name, paths in splits.items() if paths}
    if logger:
        logger(
            "[VisDrone][COCO] Splits detectados: "
            + ", ".join(f"{k}={v[0].parent if v else None}" for k, v in splits.items())
        )
    if not available:
        raise ValueError("Nenhum split VisDrone encontrado (esperado VisDrone2019-DET-*/images e annotations).")

    artifacts: Dict[str, CocoSplitArtifacts] = {}

    train_split = available.get("train")
    val_split = available.get("val")
    if train_split and val_split:
        artifacts["train"] = build_coco_for_split(
            "train",
            train_split[0],
            train_split[1],
            images_root / "train",
            annotations_root / "instances_train.json",
            logger,
        )
        artifacts["val"] = build_coco_for_split(
            "val",
            val_split[0],
            val_split[1],
            images_root / "val",
            annotations_root / "instances_val.json",
            logger,
        )
    else:
        source_split = train_split or val_split
        if not source_split:
            fallback = available.get("test")
            if fallback and fallback[1].exists():
                source_split = fallback
        if not source_split or not source_split[1].exists():
            raise ValueError("Para criar divisão 80/20 é necessário pelo menos um split com images/ e annotations/.")
        img_dir, ann_dir = source_split
        image_names = [p.name for p in _iter_images(img_dir)]
        rng = random.Random(42)
        rng.shuffle(image_names)
        split_idx = int(len(image_names) * 0.8)
        train_names = set(image_names[:split_idx])
        val_names = set(image_names[split_idx:])
        artifacts["train"] = build_coco_for_split(
            "train",
            img_dir,
            ann_dir,
            images_root / "train",
            annotations_root / "instances_train.json",
            logger,
            selected=train_names,
        )
        artifacts["val"] = build_coco_for_split(
            "val",
            img_dir,
            ann_dir,
            images_root / "val",
            annotations_root / "instances_val.json",
            logger,
            selected=val_names,
        )

    if "test" in available:
        img_dir, ann_dir = available["test"]
        artifacts["test"] = build_coco_for_split(
            "test",
            img_dir,
            ann_dir if ann_dir and ann_dir.exists() else None,
            images_root / "test",
            annotations_root / "instances_test.json",
            logger,
        )

    report = {
        "source_dataset": str(dataset_dir),
        "categories": [{"id": cid, "name": name} for cid, name in VISDRONE_CATEGORIES],
        "splits": {
            name: {
                "images_copied": art.images_copied,
                "images_with_annotations": art.images_with_annotations,
                "ignored_regions": art.ignored_regions,
                "bboxes_valid": art.bboxes_valid,
                "bboxes_discarded": art.bboxes_discarded,
                "missing_annotations": art.missing_annotations,
                "json": str(art.images_json),
                "images_dir": str(art.images_dir),
            }
            for name, art in artifacts.items()
        },
    }
    (normalized_dir / "normalization_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if logger:
        logger(f"[VisDrone][COCO] Relatório salvo em {normalized_dir / 'normalization_report.json'}")
        logger("[VisDrone][COCO] Use dataset_root=normalized_dir, images=train/val/test e JSON em annotations/")
    return artifacts
