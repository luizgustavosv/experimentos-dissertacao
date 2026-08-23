from __future__ import annotations

import csv
import json
from datetime import datetime
import os
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from app.avaliacao.metricas import DEFAULT_MAX_DETECTIONS, evaluate_coco_predictions
from app.detectors.base import Logger
from app.detectors.dataset_voc import PascalVOCDataset
from app.detectors.torchvision_models import build_ssd
from app.detectors.utils import (
    extract_checkpoint_meta,
    extract_checkpoint_state,
    filter_torchvision_predictions,
    infer_ssd_num_classes,
    load_ssd_weights,
    resolve_device,
    validate_voc_dataset,
)


_PREDICTION_KEYS = ("boxes", "scores", "labels")


def ssd_collate_fn(batch: Sequence[tuple[torch.Tensor, dict]]) -> tuple[list[torch.Tensor], list[dict]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def _load_split_ids(dataset_root: Path, split: str, fallback_ids: Sequence[str]) -> List[str]:
    imagesets_dir = dataset_root / "ImageSets" / "Main"
    split_file = imagesets_dir / f"{split}.txt"
    if split_file.exists():
        return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return list(fallback_ids)


def _filter_predictions(output: Dict[str, torch.Tensor], threshold: float, max_detections: int = DEFAULT_MAX_DETECTIONS) -> Dict[str, torch.Tensor]:
    filtered, _ = filter_torchvision_predictions(output, score_threshold=threshold)
    scores = filtered["scores"]
    if scores.numel() > max_detections:
        order = torch.argsort(scores, descending=True)[:max_detections]
        return {key: filtered[key][order] for key in _PREDICTION_KEYS}
    return {key: filtered[key] for key in _PREDICTION_KEYS}


def _xyxy_to_xywh(box: torch.Tensor) -> list[float]:
    x1, y1, x2, y2 = [float(x) for x in box.detach().cpu().tolist()]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def evaluate_torchvision_ssd_voc(
    voc_root: str,
    weights_path: str,
    split: str = "val",
    device: Optional[str] = None,
    batch_size: int = 1,
    num_workers: int = 2,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    out_dir: Optional[str] = None,
    logger: Optional[Logger] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    progress_every: int = 50,
    strict_weights: bool = True,
) -> dict:
    dataset_root, class_names, train_ids, val_ids = validate_voc_dataset(Path(voc_root))
    split_normalized = split.lower().strip()
    available_ids = {
        "train": train_ids,
        "val": val_ids,
        "test": _load_split_ids(dataset_root, "test", val_ids),
    }
    if split_normalized not in available_ids:
        raise ValueError(f"Split desconhecido: {split}. Opções válidas: train, val, test.")

    device_str = resolve_device(device)
    torch_device = torch.device(device_str)

    def _emit(message: str) -> None:
        print(message, flush=True)
        if log_cb:
            log_cb(message)
        if logger:
            logger(message)

    _emit("[EVAL] Iniciando avaliação SSD")
    _emit(
        f"[EVAL] Parâmetros: voc_root={dataset_root}, weights={weights_path}, split={split_normalized}, conf_threshold={conf_threshold}, "
        f"iou_threshold={iou_threshold}, device={device_str}, num_workers={num_workers}, out_dir={out_dir or 'padrão'}"
    )

    progress_every = max(1, progress_every)

    transform = transforms.Compose([transforms.ToTensor()])
    class_to_idx = {name: idx + 1 for idx, name in enumerate(class_names)}
    dataset = PascalVOCDataset(dataset_root, available_ids[split_normalized], class_to_idx, transforms=transform)
    _emit(f"[EVAL] Total de imagens no split '{split_normalized}': {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=ssd_collate_fn)

    weights_resolved = Path(weights_path).expanduser().resolve()
    loaded = torch.load(weights_resolved, map_location=torch_device)
    checkpoint_epoch = loaded.get("epoch") if isinstance(loaded, dict) and isinstance(loaded.get("epoch"), int) else None
    meta = extract_checkpoint_meta(loaded)
    meta_dataset_num = meta.get("dataset_num_classes") or meta.get("num_classes")
    meta_model_num = meta.get("model_num_classes")
    meta_backbone = meta.get("backbone")
    meta_imgsz = meta.get("imgsz")
    state_dict, _ = extract_checkpoint_state(loaded)
    ckpt_num_classes = infer_ssd_num_classes(state_dict, logger=_emit)
    expected_model_num = (
        meta_model_num if isinstance(meta_model_num, int) else None
    ) or (
        (meta_dataset_num + 1) if isinstance(meta_dataset_num, int) else None
    ) or ckpt_num_classes or (len(class_names) + 1)

    _emit(f"[EVAL][SSD] weights_path={weights_resolved}")
    if meta:
        _emit(
            "[EVAL][SSD] meta="
            f"dataset_num_classes={meta_dataset_num} model_num_classes={meta_model_num} "
            f"backbone={meta_backbone} imgsz={meta_imgsz}"
        )
    _emit(
        "[EVAL][SSD] num_classes_model="
        f"{expected_model_num} num_classes_ckpt={ckpt_num_classes} strict={strict_weights}"
    )
    if meta_backbone and meta_backbone != "vgg16":
        _emit(f"[EVAL][SSD][WARN] backbone do checkpoint={meta_backbone} porém build_ssd usa vgg16.")

    model = build_ssd(num_classes=int(expected_model_num))
    load_ssd_weights(
        model,
        weights_resolved,
        torch_device,
        strict=strict_weights,
        loaded=loaded,
        logger=_emit,
    )
    model.to(torch_device)
    model.eval()
    if hasattr(model, "score_thresh"):
        model.score_thresh = float(conf_threshold)
    if hasattr(model, "detections_per_img"):
        model.detections_per_img = DEFAULT_MAX_DETECTIONS

    total_batches = len(dataloader)

    _emit(
        f"[EVAL] Configuração de DataLoader: batch_size={batch_size}, num_workers={num_workers}, total_de_lotes={total_batches}"
    )
    if os.name == "nt" and num_workers > 0:
        _emit("[EVAL] Ambiente Windows detectado: usando funções globais compatíveis com multiprocessing.")

    start_time = time.time()
    predictions: list[dict[str, object]] = []
    coco_images: list[dict[str, object]] = []
    coco_annotations: list[dict[str, object]] = []
    ann_id = 1
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader, start=1):
            images_device = [img.to(torch_device) for img in images]
            outputs = model(images_device)
            for image_tensor, output, target in zip(images, outputs, targets):
                image_id = int(target["image_id"].reshape(-1)[0].item())
                img_path = str(target.get("img_path", ""))
                height, width = int(image_tensor.shape[-2]), int(image_tensor.shape[-1])
                coco_images.append({"id": image_id, "file_name": Path(img_path).name if img_path else str(image_id), "width": width, "height": height})

                gt_boxes = target.get("boxes", torch.zeros((0, 4), dtype=torch.float32)).detach().cpu().float()
                gt_labels = target.get("labels", torch.zeros((0,), dtype=torch.int64)).detach().cpu().long()
                for box, label in zip(gt_boxes, gt_labels):
                    bbox = _xyxy_to_xywh(box)
                    coco_annotations.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": int(label.item()),
                            "bbox": bbox,
                            "area": float(bbox[2] * bbox[3]),
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1

                preds = _filter_predictions(output, conf_threshold, DEFAULT_MAX_DETECTIONS)
                for box, score, label in zip(preds["boxes"].detach().cpu(), preds["scores"].detach().cpu(), preds["labels"].detach().cpu()):
                    predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": int(label.item()),
                            "bbox": _xyxy_to_xywh(box),
                            "score": float(score.item()),
                        }
                    )

            processed_images = min(batch_idx * batch_size, len(dataset))
            if processed_images % progress_every == 0 or batch_idx == total_batches:
                elapsed = time.time() - start_time
                rate = processed_images / elapsed if elapsed > 0 else 0.0
                remaining = max(len(dataset) - processed_images, 0)
                eta = remaining / rate if rate > 0 else float("inf")
                _emit(
                    f"[EVAL] {processed_images}/{len(dataset)} | {rate:.2f} img/s | elapsed {elapsed:.1f}s | ETA {eta:.1f}s"
                )

    output_dir = Path(out_dir) if out_dir else Path(weights_path).expanduser().resolve().parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_path = output_dir / "gt_coco.json"
    predictions_path = output_dir / "predictions_coco.json"
    categories = [{"id": idx + 1, "name": name} for idx, name in enumerate(class_names)]
    gt_payload = {"images": coco_images, "annotations": coco_annotations, "categories": categories}
    gt_path.write_text(json.dumps(gt_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    predictions_path.write_text(json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8")

    result = evaluate_coco_predictions(
        gt_annotations=gt_path,
        predictions_json=predictions_path,
        output_dir=output_dir,
        model_name="SSD300",
        dataset_name=str(dataset_root),
        split=split_normalized,
        weights_path=weights_resolved,
        conf_threshold=float(conf_threshold),
        iou_threshold=float(iou_threshold),
        max_detections=DEFAULT_MAX_DETECTIONS,
        input_size=300,
        device=device_str,
        epoch_relative=checkpoint_epoch,
        epoch_accumulated=checkpoint_epoch,
        logger=_emit,
        extra={"classes": class_names, "checkpoint_meta": meta},
    )
    metrics = result.get("metrics", {})
    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"

    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_fields = ["created_at", "split", "map50_95", "map50", "ar100", "precision_micro", "recall_micro", "conf_threshold", "iou_threshold", "weights_path", "num_images", "device"]
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()
        row = {
            "created_at": result.get("created_at"),
            "split": result.get("split"),
            "map50_95": metrics.get("map50_95"),
            "map50": metrics.get("map50"),
            "ar100": metrics.get("ar100"),
            "precision_micro": metrics.get("precision_micro"),
            "recall_micro": metrics.get("recall_micro"),
            "conf_threshold": result.get("parameters", {}).get("conf_threshold"),
            "iou_threshold": result.get("parameters", {}).get("iou_association_threshold"),
            "weights_path": result.get("weights_path"),
            "num_images": result.get("num_images"),
            "device": result.get("parameters", {}).get("device"),
        }
        writer.writerow(row)

    _emit(f"[EVAL] Resultado salvo em {output_dir}")
    _emit(
        f"[EVAL] mAP@0.5:0.95={metrics.get('map50_95')} | mAP@0.5={metrics.get('map50')} | Precision={metrics.get('precision_micro')} | Recall={metrics.get('recall_micro')}"
    )

    return result
