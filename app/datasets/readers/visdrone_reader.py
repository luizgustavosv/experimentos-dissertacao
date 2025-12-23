from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.datasets.ir import AnnotationRecord, DatasetIR, ImageRecord
from app.datasets.utils import clip_bbox, is_image_file, load_image_size
from app.detectors.base import Logger


VISDRONE_CLASS_NAMES = [
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
VALID_VISDRONE_CATEGORIES = set(range(1, len(VISDRONE_CLASS_NAMES) + 1))


def _classify_split(name: str) -> Optional[str]:
    lower = name.lower()
    if "train" in lower:
        return "train"
    if "val" in lower:
        return "val"
    if "test-dev" in lower or "test-challenge" in lower or "test" in lower:
        return "test"
    return None


def _iter_image_files(images_dir: Path) -> List[Path]:
    return [p for p in sorted(images_dir.iterdir()) if p.is_file() and is_image_file(p)]


def _has_annotation_files(annotations_dir: Path) -> bool:
    if not annotations_dir.exists():
        return False
    return any(p.is_file() and p.suffix.lower() == ".txt" for p in annotations_dir.iterdir())


def _gather_candidate_dirs(input_dir: Path) -> List[Path]:
    candidates = [input_dir]
    for child in input_dir.iterdir():
        if child.is_dir():
            candidates.append(child)
            if child.name.startswith("VisDrone2019-DET-"):
                continue
            for grandchild in child.iterdir():
                if grandchild.is_dir() and grandchild.name.startswith("VisDrone2019-DET-"):
                    candidates.append(grandchild)

    if _classify_split(input_dir.name):
        for sibling in input_dir.parent.iterdir():
            if sibling.is_dir():
                candidates.append(sibling)

    unique: List[Path] = []
    seen = set()
    for path in candidates:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def resolve_visdrone_splits(input_dir: Path) -> Dict[str, Optional[Path]]:
    input_dir = input_dir.expanduser().resolve()
    resolved: Dict[str, Optional[Path]] = {"train": None, "val": None, "test": None}
    tested: List[Tuple[Path, str, bool, bool]] = []

    for candidate in _gather_candidate_dirs(input_dir):
        split = _classify_split(candidate.name)
        if split is None:
            continue

        images_dir = candidate / "images"
        ann_dir = candidate / "annotations"
        has_images = images_dir.exists() and any(_iter_image_files(images_dir))
        has_annotations = _has_annotation_files(ann_dir)
        tested.append((candidate, split, has_images, has_annotations))

        if not has_images:
            continue
        if split in {"train", "val"} and not has_annotations:
            raise FileNotFoundError(f"split {split} sem annotations em {candidate}")
        if split == "test":
            current = resolved["test"]
            current_has_ann = current is not None and _has_annotation_files(current / "annotations")
            if resolved["test"] is None or (not current_has_ann and has_annotations):
                resolved["test"] = candidate
        elif resolved[split] is None:
            resolved[split] = candidate

    missing = [s for s in ("train", "val") if resolved[s] is None]
    if missing:
        tested_desc = "; ".join(
            f"{path} (split={split}, imagens={'ok' if has_img else 'faltando'}, "
            f"annotations={'ok' if has_ann else 'faltando'})"
            for path, split, has_img, has_ann in tested
        )
        details = tested_desc if tested_desc else "nenhum candidato detectado"
        raise FileNotFoundError(
            f"Splits obrigatórios não encontrados ({', '.join(missing)}) em {input_dir}. Avaliados: {details}"
        )
    return resolved


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
    logger: Optional[Logger] = None,
) -> Tuple[DatasetIR, Dict[str, int], List[str], bool]:
    dataset_dir = dataset_dir.expanduser().resolve()
    splits = resolve_visdrone_splits(dataset_dir)
    if logger:
        logger(
            f"VisDrone splits detectados: train={splits['train']}, val={splits['val']}, "
            f"test={splits.get('test')}"
        )

    images: List[ImageRecord] = []
    annotations: List[AnnotationRecord] = []
    discarded: Dict[str, int] = {
        "invalid_bbox_size": 0,
        "ignored_by_score": 0,
        "ignored_region": 0,
        "unknown_category": 0,
        "clipped_empty": 0,
        "missing_annotation_file": 0,
    }
    warnings: List[str] = []
    is_labelled = False
    images_per_split: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    labels_per_split: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    class_counts: Dict[int, int] = {idx: 0 for idx in range(len(VISDRONE_CLASS_NAMES))}

    image_id = 0
    ann_id = 0
    for split_name in ("train", "val", "test"):
        split_dir = splits.get(split_name)
        if split_dir is None:
            continue
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
                split="test" if split_name.startswith("test") else split_name,
            )
            images.append(img_record)
            images_per_split[img_record.split] = images_per_split.get(img_record.split, 0) + 1

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
                xmin, ymin, w, h, score, category, _, _ = row
                if w <= 0 or h <= 0:
                    discarded["invalid_bbox_size"] += 1
                    continue
                if category == 0:
                    discarded["ignored_region"] += 1
                    continue
                if category not in VALID_VISDRONE_CATEGORIES:
                    discarded["unknown_category"] += 1
                    continue
                if score == 0:
                    discarded["ignored_by_score"] += 1
                    continue

                xmax = xmin + w
                ymax = ymin + h
                xmin_c, ymin_c, xmax_c, ymax_c = clip_bbox(xmin, ymin, xmax, ymax, width, height)
                if xmax_c <= xmin_c or ymax_c <= ymin_c:
                    discarded["clipped_empty"] += 1
                    continue

                ann_id += 1
                ann_for_image += 1
                class_id = category - 1
                annotations.append(
                    _make_annotation(
                        ann_id,
                        image_id,
                        class_id=class_id,
                        class_name=VISDRONE_CLASS_NAMES[class_id],
                        xmin=xmin_c,
                        ymin=ymin_c,
                        xmax=xmax_c,
                        ymax=ymax_c,
                    )
                )
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
            if ann_for_image == 0 and annotations_dir.exists():
                warnings.append(f"{image_path.name} ficou sem anotações após filtros em {split_name}")
            labels_per_split[img_record.split] = labels_per_split.get(img_record.split, 0) + ann_for_image

    dataset = DatasetIR(classes=VISDRONE_CLASS_NAMES, images=images, annotations=annotations)
    if logger and any(class_counts.values()):
        logger(
            "VisDrone classes mantidas: "
            + ", ".join(
                f"{VISDRONE_CLASS_NAMES[idx]}={count}" for idx, count in sorted(class_counts.items()) if count > 0
            )
        )
    if logger:
        logger(
            "Contagem gerada por split (imagens/labels): "
            + ", ".join(
                f"{split}={images_per_split.get(split,0)}/{labels_per_split.get(split,0)}"
                for split in ("train", "val", "test")
            )
        )
    return dataset, discarded, warnings, is_labelled
