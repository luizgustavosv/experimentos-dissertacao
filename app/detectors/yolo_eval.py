from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import torch
import yaml
from PIL import Image
from ultralytics import YOLO

from app.avaliacao.metricas import DEFAULT_CURVE_CONF_THRESHOLD, DEFAULT_MAX_DETECTIONS, evaluate_coco_predictions
from app.detectors.base import Logger


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _emit_to(logger: Optional[Logger], log_cb: Optional[Callable[[str], None]], message: str) -> None:
    print(message, flush=True)
    if log_cb:
        log_cb(message)
    if logger:
        logger(message)


def _resolve_split_path(data: dict, data_path: Path, split: str) -> Path:
    raw = data.get(split)
    if raw is None:
        raise ValueError(f"data.yaml não contém o split '{split}'.")
    path = Path(str(raw))
    if path.is_absolute():
        return path
    base = Path(str(data.get("path", data_path.parent)))
    if not base.is_absolute():
        base = data_path.parent / base
    return (base / path).resolve()


def _list_images(path: Path) -> list[Path]:
    if path.is_file():
        base = path.parent
        return [
            (base / line.strip()).resolve() if not Path(line.strip()).is_absolute() else Path(line.strip()).resolve()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not path.is_dir():
        raise FileNotFoundError(f"Split de imagens YOLO não encontrado: {path}")
    return sorted(p.resolve() for p in path.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)


def _label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _class_names(data: dict) -> list[str]:
    names = data.get("names")
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names)]
    if isinstance(names, list):
        return [str(item) for item in names]
    raise ValueError("data.yaml deve conter a chave 'names' para avaliação unificada.")


def _build_coco_gt(images: list[Path], class_names: list[str]) -> tuple[dict, dict[str, int]]:
    coco_images = []
    coco_annotations = []
    image_id_by_path: dict[str, int] = {}
    ann_id = 1

    for image_id, image_path in enumerate(images, start=1):
        with Image.open(image_path) as img:
            width, height = img.size
        image_id_by_path[str(image_path.resolve())] = image_id
        coco_images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})

        label_path = _label_path_for_image(image_path)
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, xc, yc, bw, bh = [float(x) for x in parts[:5]]
            box_w = bw * width
            box_h = bh * height
            x = (xc * width) - box_w / 2
            y = (yc * height) - box_h / 2
            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(cls) + 1,
                    "bbox": [x, y, box_w, box_h],
                    "area": float(box_w * box_h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    categories = [{"id": idx + 1, "name": name} for idx, name in enumerate(class_names)]
    return {"images": coco_images, "annotations": coco_annotations, "categories": categories}, image_id_by_path


def _build_coco_image_index(images: list[Path], class_names: list[str]) -> dict:
    categories = [{"id": idx + 1, "name": name} for idx, name in enumerate(class_names)]
    coco_images = [{"id": image_id, "file_name": image_path.name} for image_id, image_path in enumerate(images, start=1)]
    return {"images": coco_images, "annotations": [], "categories": categories}


def evaluate_yolo(
    data_yaml: str,
    weights_path: str,
    out_dir: str,
    split: str = "val",
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cpu",
    conf: float = 0.25,
    iou: float = 0.5,
    logger: Optional[Logger] = None,
    log_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    data_path = Path(data_yaml).expanduser().resolve()
    weights = Path(weights_path).expanduser().resolve()
    output_dir = Path(out_dir).expanduser().resolve()

    if not data_path.is_file():
        raise FileNotFoundError(f"Arquivo data.yaml não encontrado: {data_path}")
    if not weights.is_file():
        raise FileNotFoundError(f"Pesos YOLO não encontrados: {weights}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not device:
        device = "cpu"
    device = str(device)
    if (device.isdigit() or device.startswith("cuda")) and not torch.cuda.is_available():
        device = "cpu"

    data = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    class_names = _class_names(data)
    split_path = _resolve_split_path(data, data_path, split)
    images = _list_images(split_path)
    if not images:
        raise ValueError(f"Nenhuma imagem encontrada para o split YOLO '{split}' em {split_path}.")

    gt_payload, image_id_by_path = _build_coco_gt(images, class_names)
    gt_path = output_dir / "gt_coco.json"
    train_gt_path: Path | None = None
    if split != "train" and data.get("train") is not None:
        train_images = _list_images(_resolve_split_path(data, data_path, "train"))
        train_gt_path = output_dir / "gt_coco_train_index.json"
        train_gt_path.write_text(
            json.dumps(_build_coco_image_index(train_images, class_names), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    predictions_path = output_dir / "predictions_coco.json"
    gt_path.write_text(json.dumps(gt_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    _emit_to(
        logger,
        log_cb,
        f"[EVAL][YOLO] Inferência unificada: data={data_path}, split={split}, imagens={len(images)}, "
        f"weights={weights}, imgsz={imgsz}, batch={batch}, device={device}, "
        f"export_conf={DEFAULT_CURVE_CONF_THRESHOLD}, operating_conf={conf}, iou={iou}, max_det={DEFAULT_MAX_DETECTIONS}",
    )

    model = YOLO(str(weights))
    results = model.predict(
        source=[str(p) for p in images],
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=DEFAULT_CURVE_CONF_THRESHOLD,
        iou=iou,
        max_det=DEFAULT_MAX_DETECTIONS,
        verbose=True,
    )

    predictions = []
    for result_item in results:
        image_id = image_id_by_path.get(str(Path(str(result_item.path)).resolve()))
        if image_id is None:
            continue
        boxes = getattr(result_item, "boxes", None)
        if boxes is None:
            continue
        for xyxy, score, cls in zip(boxes.xyxy.cpu(), boxes.conf.cpu(), boxes.cls.cpu()):
            x1, y1, x2, y2 = [float(v) for v in xyxy.tolist()]
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(cls.item()) + 1,
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(score.item()),
                }
            )
    predictions_path.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")

    result = evaluate_coco_predictions(
        gt_annotations=gt_path,
        train_annotations=train_gt_path,
        predictions_json=predictions_path,
        output_dir=output_dir,
        model_name="YOLOv12n",
        dataset_name=str(data_path),
        split=split,
        weights_path=weights,
        conf_threshold=float(conf),
        iou_threshold=float(iou),
        max_detections=DEFAULT_MAX_DETECTIONS,
        input_size=int(imgsz),
        device=device,
        logger=lambda msg: _emit_to(logger, log_cb, msg),
        extra={"data_yaml": str(data_path), "batch": batch, "class_names": class_names},
    )
    metrics = result.get("metrics", {})
    diagnostic_metrics = metrics.get("diagnostic", {}) if isinstance(metrics.get("diagnostic"), dict) else metrics
    _emit_to(
        logger,
        log_cb,
        f"[EVAL][YOLO] mAP@0.5={diagnostic_metrics.get('map50')} | mAP@0.5:0.95={diagnostic_metrics.get('map50_95')} | "
        f"precision_micro={metrics.get('precision_micro')} | recall_micro={metrics.get('recall_micro')}",
    )
    return result
