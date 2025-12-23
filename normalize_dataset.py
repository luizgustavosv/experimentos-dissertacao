from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from PIL import Image

# Classes conhecidas do VisDrone (para mapeamento e detecção automática do formato)
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


@dataclass
class ObjectAnnotation:
    cls: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass
class SampleResult:
    canonical_id: str
    original_name: str
    image_path: Path
    xml_path: Path
    width: int
    height: int
    objects: List[ObjectAnnotation] = field(default_factory=list)


AnnotationMap = Dict[str, List[ObjectAnnotation]]
SizeMap = Dict[str, Tuple[int, int]]


def _log(logger, message: str) -> None:
    if logger:
        logger(message)
    else:
        print(message)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _canonical_key(name: str) -> str:
    # Remove todas as extensões e normaliza para minúsculas
    candidate = name
    while True:
        stem = Path(candidate).stem
        if stem == candidate:
            break
        candidate = stem
    return candidate.lower()


def _load_state(out_dir: Path) -> dict:
    state_path = out_dir / "normalization_state.json"
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_state(out_dir: Path, state: dict) -> None:
    state_path = out_dir / "normalization_state.json"
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_report_row(out_dir: Path, split: str, original_image: str, reason: str, details: str) -> None:
    report_path = out_dir / "normalization_report.csv"
    header_needed = not report_path.exists()
    with report_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if header_needed:
            writer.writerow(["split", "original_image_name", "reason", "details"])
        writer.writerow([split, original_image, reason, details])


def _copy_to_skipped(out_dir: Path, split: str, reason: str, image_path: Optional[Path], ann_path: Optional[Path]) -> None:
    reason_dir = _ensure_dir(out_dir / "skipped" / split / reason.replace(" ", "_"))
    if image_path and image_path.exists():
        shutil.copy2(image_path, reason_dir / image_path.name)
    if ann_path and ann_path.exists():
        shutil.copy2(ann_path, reason_dir / ann_path.name)


def _detect_annotation_format(annotations_dir: Path) -> str:
    annotations_dir = annotations_dir.expanduser().resolve()
    if any(annotations_dir.glob("*.csv")):
        return "csv"
    if any(annotations_dir.glob("*.xml")):
        return "voc_xml"
    if any(annotations_dir.glob("*.txt")):
        return "txt"
    return "unknown"


def _parse_visdrone(txt_path: Path, img_w: int, img_h: int) -> Tuple[List[ObjectAnnotation], int, int]:
    results: List[ObjectAnnotation] = []
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
            if not (1 <= class_id <= len(VISDRONE_CLASSES)):
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
            results.append(ObjectAnnotation(cls=cls_name, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))

    return results, ignored, discarded


def _parse_voc_xml(xml_path: Path) -> List[ObjectAnnotation]:
    objs: List[ObjectAnnotation] = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bnd = obj.find("bndbox")
        if not name or bnd is None:
            continue
        try:
            xmin = int(float(bnd.findtext("xmin", "0")))
            ymin = int(float(bnd.findtext("ymin", "0")))
            xmax = int(float(bnd.findtext("xmax", "0")))
            ymax = int(float(bnd.findtext("ymax", "0")))
        except ValueError:
            continue
        objs.append(ObjectAnnotation(cls=name.strip(), xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))
    return objs


def _parse_csv_annotations(csv_path: Path, known_keys: Set[str]) -> Tuple[AnnotationMap, SizeMap, List[str]]:
    annotations: AnnotationMap = {}
    sizes: SizeMap = {}
    warnings: List[str] = []
    with csv_path.open("r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = row.get("filename", "") or row.get("file", "") or ""
            if not filename:
                warnings.append("Linha no CSV sem filename; ignorada.")
                continue
            base_key = _canonical_key(Path(filename).name)
            if base_key not in known_keys:
                warnings.append(f"Anotação refere-se a imagem inexistente: {filename}")
                continue
            try:
                width = int(float(row.get("width", 0)))
                height = int(float(row.get("height", 0)))
                xmin = int(float(row.get("xmin", 0)))
                ymin = int(float(row.get("ymin", 0)))
                xmax = int(float(row.get("xmax", 0)))
                ymax = int(float(row.get("ymax", 0)))
            except ValueError:
                warnings.append(f"Valores inválidos no CSV para {filename}; anotação descartada.")
                continue
            cls_name = row.get("class", "human") or "human"
            sizes.setdefault(base_key, (width, height))
            annotations.setdefault(base_key, []).append(ObjectAnnotation(cls=cls_name, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))
    return annotations, sizes, warnings


def _map_class(src_name: str, class_map: Optional[Dict[str, str]]) -> Optional[str]:
    if class_map is None:
        return "human"
    if "*" in class_map:
        default_value = class_map["*"]
    else:
        default_value = None
    if src_name in class_map:
        return class_map[src_name]
    lower_name = src_name.lower()
    if lower_name in class_map:
        return class_map[lower_name]
    return default_value


def _sanitize_bbox(xmin: int, ymin: int, xmax: int, ymax: int, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    xmin = max(0, min(xmin, width - 1))
    ymin = max(0, min(ymin, height - 1))
    xmax = max(0, min(xmax, width - 1))
    ymax = max(0, min(ymax, height - 1))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def _write_voc_xml(dest: Path, folder: str, filename: str, width: int, height: int, objects: Sequence[ObjectAnnotation]) -> None:
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = folder
    ET.SubElement(annotation, "filename").text = filename
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    for obj in objects:
        obj_el = ET.SubElement(annotation, "object")
        ET.SubElement(obj_el, "name").text = obj.cls
        ET.SubElement(obj_el, "pose").text = "Unspecified"
        ET.SubElement(obj_el, "truncated").text = "0"
        ET.SubElement(obj_el, "difficult").text = "0"
        box_el = ET.SubElement(obj_el, "bndbox")
        ET.SubElement(box_el, "xmin").text = str(obj.xmin)
        ET.SubElement(box_el, "ymin").text = str(obj.ymin)
        ET.SubElement(box_el, "xmax").text = str(obj.xmax)
        ET.SubElement(box_el, "ymax").text = str(obj.ymax)
    tree = ET.ElementTree(annotation)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dest, encoding="utf-8")


def _read_image(path: Path) -> Tuple[Image.Image, int, int]:
    with Image.open(path) as img:
        rgb_img = img.convert("RGB")
        width, height = rgb_img.size
        return rgb_img.copy(), width, height


def _save_jpeg(img: Image.Image, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format="JPEG", quality=95)


def _next_split_index(out_dir: Path, split: str) -> int:
    jpeg_dir = out_dir / "JPEGImages"
    max_idx = 0
    if jpeg_dir.exists():
        for path in jpeg_dir.glob(f"{split}_*.jpg"):
            try:
                number = int(path.stem.split("_")[-1])
                max_idx = max(max_idx, number)
            except ValueError:
                continue
    return max_idx + 1


def _purge_existing_split(out_dir: Path, split: str) -> None:
    jpeg_dir = out_dir / "JPEGImages"
    ann_dir = out_dir / "Annotations"
    for path in list(jpeg_dir.glob(f"{split}_*.jpg")) + list(ann_dir.glob(f"{split}_*.xml")):
        try:
            path.unlink()
        except FileNotFoundError:
            continue


def build_imagesets_main(out_dir: Path, ids_by_split: Dict[str, Sequence[str]]) -> None:
    imagesets = _ensure_dir(out_dir / "ImageSets" / "Main")
    for split_name in ("train", "val"):
        ids = ids_by_split.get(split_name, [])
        lines = [f"{_canonical_key(i)}\n" for i in ids]
        (imagesets / f"{split_name}.txt").write_text("".join(lines), encoding="utf-8")


def normalize_to_voc(
    src_images_dir: str,
    src_annotations_dir: str,
    out_dir: str,
    split: str,
    class_map: Optional[Dict[str, str]] = None,
    keep_skipped: bool = True,
    limit: Optional[int] = None,
    selected_original_stems: Optional[Set[str]] = None,
    logger=None,
) -> List[SampleResult]:
    split = split.lower()
    images_dir = Path(src_images_dir).expanduser().resolve()
    annotations_dir = Path(src_annotations_dir).expanduser().resolve()
    out_root = Path(out_dir).expanduser().resolve()
    jpeg_dir = _ensure_dir(out_root / "JPEGImages")
    ann_dir = _ensure_dir(out_root / "Annotations")
    _ensure_dir(out_root / "ImageSets" / "Main")

    if not images_dir.exists():
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {images_dir}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Diretório de anotações não encontrado: {annotations_dir}")

    _purge_existing_split(out_root, split)

    image_candidates = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
    keyed_images: Dict[str, Path] = {}
    duplicate_count = 0
    for img_path in sorted(image_candidates):
        key = _canonical_key(img_path.name)
        if selected_original_stems and key not in selected_original_stems:
            continue
        if key in keyed_images:
            duplicate_count += 1
            continue
        keyed_images[key] = img_path

    ann_format = _detect_annotation_format(annotations_dir)
    _log(logger, f"[NORM] split={split} formato_anotacao={ann_format} imagens={len(keyed_images)} duplicadas={duplicate_count}")

    csv_annotations: AnnotationMap = {}
    csv_sizes: SizeMap = {}
    if ann_format == "csv":
        csv_files = sorted(annotations_dir.glob("*.csv"))
        if csv_files:
            csv_annotations, csv_sizes, csv_warnings = _parse_csv_annotations(csv_files[0], set(keyed_images.keys()))
            for warn in csv_warnings:
                _append_report_row(out_root, split, "<csv>", "csv_warning", warn)

    annotation_lookup: Dict[str, Path] = {}
    if ann_format in {"voc_xml", "txt"}:
        for ann_path in annotations_dir.iterdir():
            if not ann_path.is_file():
                continue
            if ann_format == "voc_xml" and ann_path.suffix.lower() != ".xml":
                continue
            if ann_format == "txt" and ann_path.suffix.lower() != ".txt":
                continue
            key = _canonical_key(ann_path.name)
            annotation_lookup[key] = ann_path

    start_idx = _next_split_index(out_root, split)
    processed: List[SampleResult] = []
    seen_classes: Set[str] = set()
    skipped_count = 0
    examples: List[str] = []

    for idx, (key, img_path) in enumerate(sorted(keyed_images.items(), key=lambda kv: kv[0])):
        if limit is not None and len(processed) >= limit:
            break
        try:
            img, width, height = _read_image(img_path)
        except Exception as exc:
            skipped_count += 1
            _append_report_row(out_root, split, img_path.name, "image_error", str(exc))
            if keep_skipped:
                _copy_to_skipped(out_root, split, "image_error", img_path, None)
            continue

        objects: List[ObjectAnnotation] = []
        ann_path: Optional[Path] = None
        if ann_format == "csv":
            objects = csv_annotations.get(key, [])
        elif ann_format == "voc_xml":
            ann_path = annotation_lookup.get(key)
            if ann_path:
                objects = _parse_voc_xml(ann_path)
        elif ann_format == "txt":
            ann_path = annotation_lookup.get(key)
            if ann_path:
                parsed, ignored, discarded = _parse_visdrone(ann_path, width, height)
                if ignored:
                    _append_report_row(out_root, split, img_path.name, "ignored_objects", f"{ignored} linhas com score==0")
                if discarded:
                    _append_report_row(out_root, split, img_path.name, "discarded_objects", f"{discarded} linhas inválidas")
                objects = parsed

        if not objects:
            skipped_count += 1
            _append_report_row(out_root, split, img_path.name, "no_annotations", "Nenhum objeto válido encontrado")
            if keep_skipped:
                _copy_to_skipped(out_root, split, "no_annotations", img_path, ann_path)
            continue

        normalized_objects: List[ObjectAnnotation] = []
        for obj in objects:
            target_cls = _map_class(obj.cls, class_map)
            if target_cls is None:
                skipped_count += 1
                _append_report_row(out_root, split, img_path.name, "unknown_class", f"Classe ignorada: {obj.cls}")
                continue
            bbox = _sanitize_bbox(obj.xmin, obj.ymin, obj.xmax, obj.ymax, width, height)
            if bbox is None:
                skipped_count += 1
                _append_report_row(out_root, split, img_path.name, "invalid_bbox", f"BBox inválido: {obj}")
                continue
            normalized_objects.append(ObjectAnnotation(cls=target_cls, xmin=bbox[0], ymin=bbox[1], xmax=bbox[2], ymax=bbox[3]))
            seen_classes.add(target_cls)

        if not normalized_objects:
            if keep_skipped:
                _copy_to_skipped(out_root, split, "no_valid_objects", img_path, ann_path)
            _append_report_row(out_root, split, img_path.name, "no_valid_objects", "Todos os objetos foram descartados")
            skipped_count += 1
            continue

        canonical_id = f"{split}_{start_idx:06d}"
        start_idx += 1
        dest_image = jpeg_dir / f"{canonical_id}.jpg"
        dest_xml = ann_dir / f"{canonical_id}.xml"

        try:
            _save_jpeg(img, dest_image)
            _write_voc_xml(dest_xml, folder="VOC", filename=f"{canonical_id}.jpg", width=width, height=height, objects=normalized_objects)
        except Exception as exc:
            skipped_count += 1
            _append_report_row(out_root, split, img_path.name, "write_error", str(exc))
            if keep_skipped:
                _copy_to_skipped(out_root, split, "write_error", img_path, ann_path)
            continue

        sample = SampleResult(
            canonical_id=canonical_id,
            original_name=img_path.name,
            image_path=dest_image,
            xml_path=dest_xml,
            width=width,
            height=height,
            objects=normalized_objects,
        )
        processed.append(sample)
        if len(examples) < 5:
            examples.append(f"{img_path.name} -> {canonical_id} ({dest_image.name}, {dest_xml.name})")

    state = _load_state(out_root)
    splits_state = state.get("splits", {})
    splits_state[split] = [s.canonical_id for s in processed]
    state["splits"] = splits_state
    existing_classes = set(state.get("classes", []))
    state["classes"] = sorted(existing_classes.union(seen_classes or {"human"}))
    _save_state(out_root, state)

    build_imagesets_main(out_root, splits_state)
    labels_path = out_root / "labels.txt"
    labels_path.write_text("\n".join(state["classes"]) + "\n", encoding="utf-8")

    _log(logger, f"[NORM] split={split} processadas={len(processed)} puladas={skipped_count} exemplos={examples[:5]}")
    for example in examples:
        _log(logger, f"[NORM][EXAMPLE] {example}")

    return processed


def validate_voc_dataset(out_dir: str, fix: bool = True, logger=None) -> bool:
    out_root = Path(out_dir).expanduser().resolve()
    jpeg_dir = out_root / "JPEGImages"
    ann_dir = out_root / "Annotations"
    imagesets_dir = out_root / "ImageSets" / "Main"
    required = [jpeg_dir, ann_dir, imagesets_dir]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Pastas obrigatórias ausentes: {', '.join(str(p) for p in missing)}")

    valid = True
    skipped = 0

    image_ids = {_canonical_key(p.stem): p for p in jpeg_dir.glob("*.jpg")}
    xml_ids = {_canonical_key(p.stem): p for p in ann_dir.glob("*.xml")}

    for only_img in set(image_ids) - set(xml_ids):
        valid = False
        path = image_ids[only_img]
        _append_report_row(out_root, "<validate>", path.name, "orphan_image", "Sem XML correspondente")
        if fix:
            path.unlink(missing_ok=True)
            skipped += 1

    for only_xml in set(xml_ids) - set(image_ids):
        valid = False
        path = xml_ids[only_xml]
        _append_report_row(out_root, "<validate>", path.name, "orphan_xml", "Sem imagem correspondente")
        if fix:
            path.unlink(missing_ok=True)
            skipped += 1

    ids = sorted(set(image_ids).intersection(xml_ids))
    imagesets = {
        "train": (imagesets_dir / "train.txt"),
        "val": (imagesets_dir / "val.txt"),
    }
    ids_by_split: Dict[str, List[str]] = {k: [] for k in imagesets}
    for split, file_path in imagesets.items():
        if file_path.exists():
            ids_by_split[split] = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def drop_id(drop_id: str, reason: str) -> None:
        nonlocal skipped, valid
        valid = False
        _append_report_row(out_root, "<validate>", drop_id, reason, "Amostra removida na validação")
        if fix:
            (jpeg_dir / f"{drop_id}.jpg").unlink(missing_ok=True)
            (ann_dir / f"{drop_id}.xml").unlink(missing_ok=True)
            skipped += 1

    for split, split_ids in ids_by_split.items():
        filtered = []
        for sid in split_ids:
            if sid not in ids:
                valid = False
                _append_report_row(out_root, "<validate>", sid, "split_missing_files", f"{split}.txt referencia ID inexistente")
                continue
            filtered.append(sid)
        ids_by_split[split] = filtered

    for sid in ids:
        img_path = jpeg_dir / f"{sid}.jpg"
        xml_path = ann_dir / f"{sid}.xml"
        try:
            img, width, height = _read_image(img_path)
        except Exception:
            drop_id(sid, "unreadable_image")
            continue
        try:
            objects = _parse_voc_xml(xml_path)
        except Exception:
            drop_id(sid, "unreadable_xml")
            continue
        changed = False
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename_el = root.find("filename")
        if filename_el is not None and filename_el.text != f"{sid}.jpg":
            filename_el.text = f"{sid}.jpg"
            changed = True
        size_el = root.find("size")
        if size_el is not None:
            if size_el.findtext("width") != str(width):
                size_el.find("width").text = str(width)
                changed = True
            if size_el.findtext("height") != str(height):
                size_el.find("height").text = str(height)
                changed = True
        valid_objects: List[ObjectAnnotation] = []
        for obj in objects:
            bbox = _sanitize_bbox(obj.xmin, obj.ymin, obj.xmax, obj.ymax, width, height)
            if bbox is None:
                changed = True
                continue
            valid_objects.append(ObjectAnnotation(cls=obj.cls, xmin=bbox[0], ymin=bbox[1], xmax=bbox[2], ymax=bbox[3]))
        if not valid_objects:
            drop_id(sid, "no_objects_after_validation")
            continue
        if changed:
            for obj_el in list(root.findall("object")):
                root.remove(obj_el)
            for obj in valid_objects:
                obj_el = ET.SubElement(root, "object")
                ET.SubElement(obj_el, "name").text = obj.cls
                ET.SubElement(obj_el, "pose").text = "Unspecified"
                ET.SubElement(obj_el, "truncated").text = "0"
                ET.SubElement(obj_el, "difficult").text = "0"
                box_el = ET.SubElement(obj_el, "bndbox")
                ET.SubElement(box_el, "xmin").text = str(obj.xmin)
                ET.SubElement(box_el, "ymin").text = str(obj.ymin)
                ET.SubElement(box_el, "xmax").text = str(obj.xmax)
                ET.SubElement(box_el, "ymax").text = str(obj.ymax)
            tree.write(xml_path, encoding="utf-8")

    build_imagesets_main(out_root, ids_by_split)
    _log(logger, f"[VALIDATE] válido={valid} corrigidos={skipped}")
    return valid


def _load_class_map(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    cm_path = Path(path).expanduser().resolve()
    if not cm_path.exists():
        raise FileNotFoundError(f"Arquivo de mapeamento de classes não encontrado: {cm_path}")
    data = json.loads(cm_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("class_map precisa ser um JSON objeto {classe_origem: classe_destino}")
    return {str(k): str(v) for k, v in data.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Normalização robusta para Pascal VOC")
    parser.add_argument("--src-images", type=str, help="Diretório com imagens de origem")
    parser.add_argument("--src-ann", type=str, help="Diretório com anotações de origem (txt/xml/csv)")
    parser.add_argument("--out", type=str, required=True, help="Diretório de saída VOC")
    parser.add_argument("--split", type=str, default="train", help="Split (train ou val)")
    parser.add_argument("--class-map", type=str, help="JSON com mapeamento de classes")
    parser.add_argument("--keep-skipped", action="store_true", default=False, help="Manter amostras descartadas em skipped/")
    parser.add_argument("--limit", type=int, help="Limita o número de amostras processadas (debug)")
    parser.add_argument("--validate-only", action="store_true", help="Apenas valida o OUT")
    parser.add_argument("--quick-test", action="store_true", help="Processa apenas 10 amostras para validação rápida")
    args = parser.parse_args(argv)

    if args.validate_only:
        ok = validate_voc_dataset(args.out, fix=True)
        return 0 if ok else 1

    if not args.src_images or not args.src_ann:
        parser.error("--src-images e --src-ann são obrigatórios quando não estiver em modo de validação")
    class_map = _load_class_map(args.class_map)
    _log(None, "[NORM] Iniciando processamento")
    if args.quick_test and args.limit is None:
        args.limit = 10
        _log(None, "[NORM] quick-test ativado: processando apenas 10 amostras")
    processed = normalize_to_voc(
        src_images_dir=args.src_images,
        src_annotations_dir=args.src_ann,
        out_dir=args.out,
        split=args.split,
        class_map=class_map,
        keep_skipped=args.keep_skipped,
        limit=args.limit,
        logger=None,
    )
    _log(None, f"[NORM] Total processado ({args.split}): {len(processed)}")
    validate_voc_dataset(args.out, fix=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
