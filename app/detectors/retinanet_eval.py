from __future__ import annotations

import json
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import torch

from app.detectors.base import Logger
from app.detectors.config import TrainConfig
from app.detectors.torchvision_train import (
    _build_val_loader_and_classes,
    _configure_logging,
    _extract_state_dict,
    _extract_checkpoint_meta,
    _safe_stream,
    ensure_weights_size,
    resolve_device,
    run_val_loss_loop,
    run_val_coco_metrics,
)


def _retinanet_anchors_per_location() -> int:
    from torchvision.models.detection import retinanet_resnet50_fpn

    model = retinanet_resnet50_fpn(weights=None, weights_backbone=None, num_classes=91)
    anchors_per_location = model.anchor_generator.num_anchors_per_location()
    return anchors_per_location[0] if anchors_per_location else 0


def _infer_retinanet_num_classes(state_dict: dict, logger: logging.Logger) -> Optional[int]:
    bias = state_dict.get("head.classification_head.cls_logits.bias")
    if bias is None:
        logger.warning("[RETINANET][VAL] Não foi possível inferir num_classes: bias ausente no state_dict.")
        return None

    anchors_per_location = _retinanet_anchors_per_location()
    if anchors_per_location <= 0:
        logger.warning("[RETINANET][VAL] anchors_per_location inválido ao inferir num_classes.")
        return None

    if bias.numel() % anchors_per_location != 0:
        logger.warning(
            "[RETINANET][VAL] Tamanho do bias (%s) não divisível por anchors_per_location=%s.",
            bias.numel(),
            anchors_per_location,
        )
        return None

    num_classes = int(bias.numel() // anchors_per_location)
    logger.info(
        "[RETINANET][VAL] Inferido ckpt_num_classes=%d a partir de cls_logits.bias (anchors=%d)",
        num_classes,
        anchors_per_location,
    )
    return num_classes


def _strip_retinanet_head(state_dict: dict) -> tuple[dict, list[str]]:
    ignored = [key for key in state_dict if key.startswith("head.classification_head.cls_logits")]
    filtered = {key: value for key, value in state_dict.items() if key not in ignored}
    return filtered, ignored


def _load_retinanet_weights_with_head_guard(
    model: torch.nn.Module,
    state_dict: dict,
    *,
    model_num_classes: int,
    emit_info: Callable[[str], None],
    emit_warning: Callable[[str], None],
    logging_logger: logging.Logger,
) -> tuple[list[str], list[str], Optional[int], bool, list[str], bool]:
    ckpt_num_classes = _infer_retinanet_num_classes(state_dict, logging_logger)
    head_mismatch = ckpt_num_classes is not None and ckpt_num_classes != model_num_classes
    filtered_state_dict = state_dict
    ignored_keys: list[str] = []
    strict_load = not head_mismatch

    if head_mismatch:
        emit_warning(
            f"[WEIGHTS][RETINANET] class-mismatch ckpt_num_classes={ckpt_num_classes} model_num_classes={model_num_classes}"
        )
        filtered_state_dict, ignored_keys = _strip_retinanet_head(state_dict)
        emit_info(f"[WEIGHTS][RETINANET] Ignorando chaves da cls_logits: {ignored_keys}")

    try:
        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=strict_load)
    except RuntimeError as exc:
        emit_warning(
            f"[WEIGHTS][RETINANET] Falha ao carregar pesos com strict={strict_load}: {exc}; tentando sem cls_logits."
        )
        filtered_state_dict, ignored_keys = _strip_retinanet_head(state_dict)
        head_mismatch = True
        strict_load = False
        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        if ckpt_num_classes is None:
            ckpt_num_classes = _infer_retinanet_num_classes(state_dict, logging_logger)

    if missing:
        emit_info(f"[WEIGHTS][RETINANET] missing_keys: {missing}")
    if unexpected:
        emit_info(f"[WEIGHTS][RETINANET] unexpected_keys: {unexpected}")
    if head_mismatch and not ignored_keys:
        ignored_keys = ["head.classification_head.cls_logits.weight", "head.classification_head.cls_logits.bias"]
        logging_logger.debug("[WEIGHTS][RETINANET] Nenhuma chave cls_logits encontrada; registrando padrões")

    return missing, unexpected, ckpt_num_classes, head_mismatch, ignored_keys, strict_load


def _bbox_xywh_to_xyxy(box: list[float] | tuple[float, ...]) -> torch.Tensor:
    x, y, w, h = box
    return torch.tensor([x, y, x + w, y + h], dtype=torch.float32)


def _compute_iou_xyxy(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(float(xa1), float(xb1))
    inter_y1 = max(float(ya1), float(yb1))
    inter_x2 = min(float(xa2), float(xb2))
    inter_y2 = min(float(ya2), float(yb2))

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, float(xa2 - xa1)) * max(0.0, float(ya2 - ya1))
    area_b = max(0.0, float(xb2 - xb1)) * max(0.0, float(yb2 - yb1))
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _compute_classic_pr_metrics(
    predictions_path: Path,
    gt_annotations: Path,
    logging_logger: logging.Logger,
    *,
    iou_threshold: float = 0.5,
    score_threshold: float | None = None,
) -> dict[str, float]:
    predictions = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    gt_data = json.loads(Path(gt_annotations).read_text(encoding="utf-8"))

    preds_by_image: dict[int, dict[int, list[tuple[float, torch.Tensor]]]] = {}
    for pred in predictions:
        try:
            image_id = int(pred["image_id"])
            category_id = int(pred["category_id"])
            score = float(pred.get("score", 0.0))
            if score_threshold is not None and score < score_threshold:
                continue
            box_xyxy = _bbox_xywh_to_xyxy(pred["bbox"])
        except Exception:
            continue

        preds_by_image.setdefault(image_id, {}).setdefault(category_id, []).append((score, box_xyxy))

    gts_by_image: dict[int, dict[int, list[torch.Tensor]]] = {}
    for ann in gt_data.get("annotations", []):
        try:
            image_id = int(ann["image_id"])
            category_id = int(ann["category_id"])
            box_xyxy = _bbox_xywh_to_xyxy(ann["bbox"])
        except Exception:
            continue

        gts_by_image.setdefault(image_id, {}).setdefault(category_id, []).append(box_xyxy)

    all_image_ids = set(preds_by_image.keys()) | set(gts_by_image.keys())
    tp = fp = fn = 0

    for image_id in all_image_ids:
        preds_per_class = preds_by_image.get(image_id, {})
        gts_per_class = gts_by_image.get(image_id, {})
        all_classes = set(preds_per_class.keys()) | set(gts_per_class.keys())

        for cls in all_classes:
            preds = sorted(preds_per_class.get(cls, []), key=lambda item: item[0], reverse=True)
            gts = gts_per_class.get(cls, [])
            matched = [False] * len(gts)

            for _, pred_box in preds:
                best_iou = 0.0
                best_gt_idx: int | None = None
                for gt_idx, gt_box in enumerate(gts):
                    if matched[gt_idx]:
                        continue
                    iou = _compute_iou_xyxy(pred_box, gt_box)
                    if iou >= iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_gt_idx is not None:
                    matched[best_gt_idx] = True
                    tp += 1
                else:
                    fp += 1

            fn += matched.count(False)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    logging_logger.info(
        "[RETINANET][VAL] IoU@%.2f score_threshold=%s TP=%d FP=%d FN=%d precision=%.4f recall=%.4f f1=%.4f",
        iou_threshold,
        score_threshold if score_threshold is not None else "none",
        tp,
        fp,
        fn,
        precision,
        recall,
        f1,
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def validate_retinanet_post_train(
    model_builder: Callable[[int], torch.nn.Module],
    dataset_dir: Path,
    train_ann: Path,
    val_ann: Path,
    weights_path: Path,
    config: TrainConfig,
    *,
    logger: Optional[Logger] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    output_dir: Optional[Path] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> dict:
    log_dir = Path(config.log_dir).expanduser().resolve()
    safe_stdout = _safe_stream("retinanet_val_stdout", log_dir)
    logging_logger, log_path = _configure_logging(
        config.verbose,
        log_dir,
        logger,
        stream_override=safe_stdout,
        logger_name="retinanet_val",
        log_prefix="retinanet_val",
    )

    def _emit(message: str) -> None:
        if log_cb:
            log_cb(message)
        logging_logger.info(message)

    def _emit_warning(message: str) -> None:
        if log_cb:
            log_cb(message)
        logging_logger.warning(message)

    _emit(f"[RETINANET][VAL-POST] Logger inicializado em {log_path}")
    val_mode_requested = getattr(config, "val_mode", "metrics")
    if val_mode_requested not in {"loss", "metrics"}:
        logging_logger.warning("[RETINANET][VAL-POST] val_mode desconhecido %s; forçando 'metrics'", val_mode_requested)
        val_mode_requested = "metrics"
    _emit(
        f"[RETINANET][VAL-POST] Modo={val_mode_requested} "
        f"conf_threshold={conf_threshold} iou_threshold={iou_threshold}"
    )

    device_str = resolve_device(config.device)
    device = torch.device(device_str)
    _emit(f"[RETINANET][VAL-POST] Dispositivo: {device_str}")

    ensure_weights_size(weights_path, logger=_emit)

    val_loader, model_num_classes, dataset_num_classes, num_workers = _build_val_loader_and_classes(
        dataset_dir,
        train_ann,
        val_ann,
        replace(config, val_mode=val_mode_requested),
        logging_logger,
        override_val_ratio=0.0,
        expects_background=False,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(output_dir).expanduser().resolve() if output_dir else Path(config.log_dir).expanduser().resolve() / "retinanet" / "val_post"
    out_dir = output_root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    _emit(f"[RETINANET][VAL-POST] Diretório de saída: {out_dir}")

    loaded = torch.load(weights_path.expanduser().resolve(), map_location="cpu")
    checkpoint_epoch = loaded.get("epoch") if isinstance(loaded, dict) and isinstance(loaded.get("epoch"), int) else None
    meta = _extract_checkpoint_meta(loaded)
    state_dict, checkpoint_format = _extract_state_dict(loaded)
    _emit(f"[RETINANET][VAL-POST] Formato de checkpoint: {checkpoint_format}")

    meta_num_classes = meta.get("num_classes") if isinstance(meta, dict) else None
    ckpt_inferred = _infer_retinanet_num_classes(state_dict, logging_logger)
    ckpt_num_classes = meta_num_classes if isinstance(meta_num_classes, int) else ckpt_inferred
    legacy_mode = False
    if ckpt_num_classes is not None and ckpt_num_classes != dataset_num_classes:
        if config.legacy_retinanet_compat and ckpt_num_classes == dataset_num_classes + 1 and dataset_num_classes == 1:
            legacy_mode = True
            _emit_warning(
                "[RETINANET][VAL-POST] LEGACY COMPAT: ckpt_num_classes inclui background fantasma; mapeando label=1->human e descartando label=0. Recomenda-se retreinar."
            )
        else:
            raise RuntimeError(
                f"[RETINANET][VAL-POST] Mismatch de classes: ckpt_num_classes={ckpt_num_classes} dataset_num_classes={dataset_num_classes}. Habilite legacy_retinanet_compat ou retreine."
            )

    effective_model_classes = ckpt_num_classes or model_num_classes
    model = model_builder(effective_model_classes)
    model.to(device)

    (
        missing,
        unexpected,
        ckpt_num_classes,
        head_mismatch,
        ignored_keys,
        strict_load,
    ) = _load_retinanet_weights_with_head_guard(
        model,
        state_dict,
        model_num_classes=model_num_classes,
        emit_info=_emit,
        emit_warning=_emit_warning,
        logging_logger=logging_logger,
    )
    metrics_valid = not head_mismatch

    logging_logger.info(
        "[RETINANET][VAL-POST] ckpt_num_classes=%s model_num_classes=%s strict_load=%s",
        ckpt_num_classes,
        model_num_classes,
        strict_load,
    )

    # run_val_coco_metrics converte boxes xyxy -> COCO [x, y, w, h] já reescalando para o tamanho original
    # (orig_size) e preservando o image_id real vindo do target. Isso garante compatibilidade com COCOeval.
    label_map = meta.get("label_to_cat_id") if isinstance(meta, dict) else None
    if not label_map and hasattr(getattr(val_loader, "dataset", None), "label_to_cat_id"):
        try:
            label_map = dict(getattr(val_loader.dataset, "label_to_cat_id"))
        except Exception:
            label_map = None
    if legacy_mode and label_map and len(label_map) == 1:
        only_cat = next(iter(label_map.values()))
        label_map = dict(label_map)
        label_map[1] = only_cat

    if val_mode_requested == "metrics":
        results = run_val_coco_metrics(
            model,
            val_loader,
            device_str,
            val_ann,
            logging_logger,
            tag="[RETINANET][VAL]",
            output_dir=out_dir,
            label_to_cat_id=label_map,
            legacy_drop_label=0 if legacy_mode else None,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            train_ann=train_ann,
            weights_path=weights_path,
            model_name="RetinaNet",
            dataset_name=str(dataset_dir),
            input_size=config.imgsz,
            epoch_relative=checkpoint_epoch,
            epoch_accumulated=checkpoint_epoch,
        )
        metrics_valid = metrics_valid and results.get("metrics_valid", True)
        pr_metrics = {}
    else:
        results = run_val_loss_loop(
            model,
            val_loader,
            device_str,
            num_classes=model_num_classes,
            expects_background=False,
            label_offset=1,
            logging_logger=logging_logger,
            tag="[RETINANET][VAL-LOSS]",
        )
        pr_metrics = {}

    results_payload = {
        "dataset": str(dataset_dir),
        "train_annotations": str(train_ann),
        "val_annotations": str(val_ann),
        "split": "val",
        "val_ratio": config.val_ratio,
        "seed": config.seed,
        "imgsz": config.imgsz,
        "batch_size": config.batch_size,
        "num_workers": num_workers,
        "weights_path": str(weights_path.expanduser().resolve()),
        "timestamp": datetime.now().isoformat(),
        "device": device_str,
        "dataset_num_classes": dataset_num_classes,
        "model_num_classes": model_num_classes,
        "ckpt_num_classes": ckpt_num_classes,
        "weights_head_mismatch": head_mismatch,
        "metrics_valid": metrics_valid,
        "ignored_head_keys": ignored_keys,
        "val_mode_requested": val_mode_requested,
        "val_mode": val_mode_requested,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        **results,
        **pr_metrics,
    }

    results_payload.update({"output_dir": str(out_dir)})
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _emit(f"[RETINANET][VAL-POST] Resultado salvo em {results_path}")

    results_payload.update({"results_path": str(results_path)})
    return results_payload

