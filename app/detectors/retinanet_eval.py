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
    _ensure_pycocotools,
    _extract_state_dict,
    _safe_stream,
    ensure_weights_size,
    resolve_device,
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


def _compute_precision_recall_from_coco(
    predictions_path: Path,
    gt_annotations: Path,
    logging_logger: logging.Logger,
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    # Reutiliza o COCOeval para derivar precisão/recall globais usando apenas IoU>=iou_threshold.
    # A matriz de precisão já contém o mapeamento entre predições e GT após o matching greedy do COCOeval.
    COCO, COCOeval = _ensure_pycocotools()
    coco_gt = COCO(str(gt_annotations))
    coco_dt = coco_gt.loadRes(str(predictions_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.iouThrs = [iou_threshold]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    precisions = coco_eval.eval.get("precision")
    recalls = coco_eval.eval.get("recall")
    precision_mean = 0.0
    recall_mean = 0.0
    if precisions is not None:
        precision_slice = precisions[0, :, :, 0, -1]
        precision_slice = precision_slice[precision_slice > -1]
        precision_mean = float(precision_slice.mean()) if precision_slice.size else 0.0
    if recalls is not None:
        recall_slice = recalls[0, :, 0, -1]
        recall_slice = recall_slice[recall_slice > -1]
        recall_mean = float(recall_slice.mean()) if recall_slice.size else 0.0

    f1 = (2 * precision_mean * recall_mean) / (precision_mean + recall_mean) if (precision_mean + recall_mean) > 0 else 0.0
    logging_logger.info(
        "[RETINANET][VAL] IoU@%.2f precision=%.4f recall=%.4f f1=%.4f",
        iou_threshold,
        precision_mean,
        recall_mean,
        f1,
    )
    return {"precision": precision_mean, "recall": recall_mean, "f1": f1}


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

    device_str = resolve_device(config.device)
    device = torch.device(device_str)
    _emit(f"[RETINANET][VAL-POST] Dispositivo: {device_str}")

    ensure_weights_size(weights_path, logger=_emit)

    val_loader, model_num_classes, dataset_num_classes, num_workers = _build_val_loader_and_classes(
        dataset_dir,
        train_ann,
        val_ann,
        replace(config, val_mode="metrics"),
        logging_logger,
        expects_background=False,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(config.log_dir).expanduser().resolve() / "retinanet" / "val_post" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    model = model_builder(model_num_classes)
    model.to(device)

    loaded = torch.load(weights_path.expanduser().resolve(), map_location="cpu")
    state_dict, checkpoint_format = _extract_state_dict(loaded)
    _emit(f"[RETINANET][VAL-POST] Formato de checkpoint: {checkpoint_format}")

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
    results = run_val_coco_metrics(
        model,
        val_loader,
        device_str,
        val_ann,
        logging_logger,
        tag="[RETINANET][VAL]",
        output_dir=out_dir,
    )

    pr_metrics = _compute_precision_recall_from_coco(
        Path(results["predictions_coco_json"]),
        Path(results["gt_annotations"]),
        logging_logger,
    )

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
        "val_mode_requested": "metrics",
        "val_mode": "metrics",
        **results,
        **pr_metrics,
    }

    results_payload.update({"output_dir": str(out_dir)})
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _emit(f"[RETINANET][VAL-POST] Resultado salvo em {results_path}")

    results_payload.update({"results_path": str(results_path)})
    return results_payload

