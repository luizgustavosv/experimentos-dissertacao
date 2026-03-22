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
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from app.detectors.base import Logger
from app.detectors.dataset_voc import PascalVOCDataset
from app.detectors.torchvision_models import build_ssd
from app.detectors.utils import (
    filter_torchvision_predictions,
    infer_ssd_num_classes,
    load_ssd_weights,
    resolve_device,
    resolve_ssd_run_config,
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


def _filter_predictions(output: Dict[str, torch.Tensor], threshold: float) -> Dict[str, torch.Tensor]:
    filtered, _ = filter_torchvision_predictions(output, score_threshold=threshold)
    return {key: filtered[key] for key in _PREDICTION_KEYS}


def _prepare_target(target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "boxes": target.get("boxes", torch.zeros((0, 4), dtype=torch.float32)).detach().cpu().float(),
        "labels": target.get("labels", torch.zeros((0,), dtype=torch.int64)).detach().cpu().long(),
    }


def _update_pr_counters(
    preds: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    iou_threshold: float,
) -> Dict[str, int]:
    tp = fp = fn = 0
    pred_boxes = preds["boxes"].float()
    pred_scores = preds["scores"].float()
    pred_labels = preds["labels"].long()
    gt_boxes = target["boxes"].float()
    gt_labels = target["labels"].long()

    if pred_scores.numel() > 0:
        order = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]

    matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
    for box, label in zip(pred_boxes, pred_labels):
        if label.item() != 1:
            continue
        if gt_boxes.numel() == 0:
            fp += 1
            continue
        ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)
        max_iou, max_idx = (ious.max(0) if ious.numel() > 0 else (torch.tensor(0.0), torch.tensor(0)))
        if max_iou >= iou_threshold and not matched[max_idx] and gt_labels[max_idx] == 1:
            tp += 1
            matched[max_idx] = True
        else:
            fp += 1

    positives = int((gt_labels == 1).sum().item())
    matched_count = int(matched.sum().item())
    fn += max(0, positives - matched_count)
    return {"tp": tp, "fp": fp, "fn": fn}


def evaluate_torchvision_ssd_voc(
    voc_root: str,
    weights_path: str,
    split: str = "val",
    device: Optional[str] = None,
    batch_size: int = 1,
    num_workers: int = 2,
    conf_threshold: float = 0.05,
    iou_threshold: float = 0.5,
    out_dir: Optional[str] = None,
    logger: Optional[Logger] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    progress_every: int = 50,
    strict_weights: bool = True,
    strict_head: bool = True,
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
    args_info = resolve_ssd_run_config(weights_resolved, logger=_emit)
    loaded = torch.load(weights_resolved, map_location=torch_device)
    meta = loaded.get("meta") if isinstance(loaded, dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    meta_dataset_num = meta.get("dataset_num_classes") or meta.get("num_classes")
    meta_model_num = meta.get("model_num_classes")
    meta_backbone = meta.get("backbone")
    meta_imgsz = meta.get("imgsz")

    expected_model_num = args_info.get("model_num_classes") or meta_model_num
    expected_dataset_num = args_info.get("dataset_num_classes") or meta_dataset_num
    if expected_model_num is None and isinstance(expected_dataset_num, int):
        expected_model_num = expected_dataset_num + 1

    state_dict = None
    if isinstance(loaded, dict):
        state_dict = loaded.get("model_state") or loaded.get("state_dict") or loaded.get("model")
    if not isinstance(state_dict, dict) and isinstance(loaded, dict):
        state_dict = loaded if loaded and all(torch.is_tensor(v) for v in loaded.values()) else None
    if not isinstance(state_dict, dict):
        state_dict = None
    ckpt_num_classes = infer_ssd_num_classes(state_dict or {}, logger=_emit)

    if expected_model_num is None:
        expected_model_num = ckpt_num_classes or (len(class_names) + 1)

    _emit(f"[EVAL][SSD] weights_path={weights_resolved}")
    if args_info:
        _emit(
            "[EVAL][SSD] args.yaml="
            f"{args_info.get('args_path')} dataset_num_classes={args_info.get('dataset_num_classes')} "
            f"model_num_classes={args_info.get('model_num_classes')} backbone={args_info.get('backbone')} "
            f"imgsz={args_info.get('imgsz')}"
        )
    if meta:
        _emit(
            "[EVAL][SSD] meta="
            f"dataset_num_classes={meta_dataset_num} model_num_classes={meta_model_num} "
            f"backbone={meta_backbone} imgsz={meta_imgsz}"
        )
    _emit(
        "[EVAL][SSD] num_classes_model="
        f"{expected_model_num} num_classes_ckpt={ckpt_num_classes} strict={strict_weights} strict_head={strict_head}"
    )
    if args_info.get("backbone") and args_info.get("backbone") != "vgg16":
        _emit(f"[EVAL][SSD][WARN] backbone esperado={args_info.get('backbone')} porém build_ssd usa vgg16.")
    if meta_backbone and meta_backbone != "vgg16":
        _emit(f"[EVAL][SSD][WARN] backbone do checkpoint={meta_backbone} porém build_ssd usa vgg16.")

    model = build_ssd(num_classes=int(expected_model_num))
    load_ssd_weights(
        model,
        weights_resolved,
        torch_device,
        strict=strict_weights,
        strict_head=strict_head,
        expected_num_classes=int(expected_model_num),
        loaded=loaded,
        logger=_emit,
    )
    model.to(torch_device)
    model.eval()

    metric = MeanAveragePrecision(iou_type="bbox")
    counters = {"tp": 0, "fp": 0, "fn": 0}

    total_batches = len(dataloader)

    _emit(
        f"[EVAL] Configuração de DataLoader: batch_size={batch_size}, num_workers={num_workers}, total_de_lotes={total_batches}"
    )
    if os.name == "nt" and num_workers > 0:
        _emit("[EVAL] Ambiente Windows detectado: usando funções globais compatíveis com multiprocessing.")

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader, start=1):
            images_device = [img.to(torch_device) for img in images]
            outputs = model(images_device)
            for output, target in zip(outputs, targets):
                preds = _filter_predictions(output, conf_threshold)
                tgt = _prepare_target(target)
                metric.update([preds], [tgt])
                pr_counts = _update_pr_counters(preds, tgt, iou_threshold)
                for key in counters:
                    counters[key] += pr_counts[key]

            processed_images = min(batch_idx * batch_size, len(dataset))
            if processed_images % progress_every == 0 or batch_idx == total_batches:
                elapsed = time.time() - start_time
                rate = processed_images / elapsed if elapsed > 0 else 0.0
                remaining = max(len(dataset) - processed_images, 0)
                eta = remaining / rate if rate > 0 else float("inf")
                _emit(
                    f"[EVAL] {processed_images}/{len(dataset)} | {rate:.2f} img/s | elapsed {elapsed:.1f}s | ETA {eta:.1f}s"
                )

    metric_result = metric.compute()
    precision = counters["tp"] / (counters["tp"] + counters["fp"] + 1e-8) if (counters["tp"] + counters["fp"]) > 0 else 0.0
    recall = counters["tp"] / (counters["tp"] + counters["fn"] + 1e-8) if (counters["tp"] + counters["fn"]) > 0 else 0.0

    result = {
        "map": float(metric_result.get("map", torch.tensor(0.0)).item()),
        "map50": float(metric_result.get("map_50", torch.tensor(0.0)).item()),
        "mar_100": float(metric_result.get("mar_100", torch.tensor(0.0)).item()),
        "precision": float(precision),
        "recall": float(recall),
        "conf_threshold": float(conf_threshold),
        "iou_threshold": float(iou_threshold),
        "split": split_normalized,
        "weights_path": str(Path(weights_path).expanduser().resolve()),
        "timestamp": datetime.now().isoformat(),
        "device": device_str,
        "num_images": len(dataset),
        "classes": class_names,
    }

    output_dir = Path(out_dir) if out_dir else Path(weights_path).expanduser().resolve().parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"

    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_fields = ["timestamp", "split", "map", "map50", "mar_100", "precision", "recall", "conf_threshold", "iou_threshold", "weights_path", "num_images", "device"]
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerow({field: result.get(field, "") for field in csv_fields})

    _emit(f"[EVAL] Resultado salvo em {output_dir}")
    _emit(
        f"[EVAL] mAP@0.5:0.95={result['map']:.4f} | mAP@0.5={result['map50']:.4f} | Precision={result['precision']:.4f} | Recall={result['recall']:.4f}"
    )

    return result
