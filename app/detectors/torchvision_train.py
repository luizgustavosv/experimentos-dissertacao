from __future__ import annotations

import faulthandler
import importlib.util
import json
import logging
import math
import os
import random
import subprocess
from collections import Counter
import signal
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.retinanet import RetinaNet

from app.datasets.class_mapping import summarize_class_mapping
from app.detectors.base import Logger
from app.detectors.config import TrainConfig
from app.detectors.dataset_coco import (
    CocoDetectionDataset,
    DetectionResize,
    DetectionToTensor,
    DetectionTransformCompose,
)
from app.detectors.utils import (
    atomic_torch_save,
    coco_collate,
    describe_dataloader,
    ensure_weights_size,
    resolve_device,
    ssd_collate_with_diagnostics,
    seed_everything,
)
from app.metrics import Metrics
from app.training.early_stopping import TrainLossEMAStopper

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback silencioso se tqdm não estiver disponível
    tqdm = None

_retinanet_label_log_before = False
_retinanet_label_log_after = False
_retinanet_label_mode_logged = False


def _describe_structure(obj: Any, depth: int = 0, max_depth: int = 4) -> str:
    if depth >= max_depth:
        return "..."
    if isinstance(obj, dict):
        keys = list(obj.keys())
        key_str = ",".join(map(str, keys))
        value_parts = []
        for idx, value in enumerate(obj.values()):
            if idx >= 3:
                value_parts.append("...")
                break
            value_parts.append(_describe_structure(value, depth + 1, max_depth))
        values_desc = ",".join(value_parts)
        return f"dict(keys=[{key_str}], values=[{values_desc}])"
    if isinstance(obj, (list, tuple)):
        parts = []
        for idx, value in enumerate(obj):
            if idx >= 4:
                parts.append("...")
                break
            parts.append(_describe_structure(value, depth + 1, max_depth))
        joined = ",".join(parts)
        type_name = "list" if isinstance(obj, list) else "tuple"
        return f"{type_name}[len={len(obj)}]([{joined}])"
    if torch.is_tensor(obj):
        return "tensor"
    if isinstance(obj, (float, int)):
        return "float"
    if obj is None:
        return "None"
    return type(obj).__name__


# Exemplo de execução:
# python train_ssd.py --dataset heridal --epochs 1 --verbose --log-every 10 --debug-dataloader
# python torchvision_train.py --dataset heridal --epochs 1 --checkpoint-dir /caminho/para/checkpoints


def _normalize_losses(loss_out: Any, device: torch.device) -> Tuple[torch.Tensor, dict]:
    """
    Retorna:
      - loss_total (torch.Tensor)
      - loss_items (dict) apenas para logging (pode ser vazio)
    """
    logger = logging.getLogger("ssd_train")
    warned_types: set[str] = set()
    loss_items: dict[str, float] = {}
    warning_throttle = 0

    def _accumulate(obj: Any, prefix: str = "") -> torch.Tensor:
        nonlocal warning_throttle
        total = torch.tensor(0.0, device=device)

        if obj is None:
            return total

        if isinstance(obj, dict):
            for key, value in obj.items():
                child_prefix = f"{prefix}{key}."
                total = total + _accumulate(value, child_prefix)
            return total

        if isinstance(obj, (list, tuple)):
            for idx, value in enumerate(obj):
                child_prefix = f"{prefix}{idx}."
                total = total + _accumulate(value, child_prefix)
            return total

        if torch.is_tensor(obj):
            if obj.device != device:
                obj = obj.to(device)
            if obj.numel() == 0:
                return torch.tensor(0.0, device=obj.device)
            if obj.is_floating_point() or obj.is_complex():
                reduced = obj.mean() if obj.dim() > 0 else obj
                name = prefix[:-1] if prefix.endswith(".") else prefix
                if name:
                    try:
                        loss_items[name] = float(reduced.detach().cpu())
                    except Exception:
                        logger.debug("Falha ao registrar loss_item para %s", name)
                return reduced
            return torch.tensor(0.0, device=obj.device)

        if isinstance(obj, (int, float)):
            name = prefix[:-1] if prefix.endswith(".") else prefix
            if name:
                loss_items[name] = float(obj)
            return torch.tensor(float(obj), device=device)

        tname = type(obj).__name__
        if tname not in warned_types and warning_throttle == 0:
            warned_types.add(tname)
            logger.warning("_normalize_losses ignorando tipo inesperado em loss_out: %s", tname)
        warning_throttle = (warning_throttle + 1) % 50
        return total

    try:
        if isinstance(loss_out, dict) and loss_out:
            tensors = [
                v
                for v in loss_out.values()
                if torch.is_tensor(v) and (v.is_floating_point() or v.is_complex())
            ]
            if tensors:
                loss_total = sum(tensors)
                for key, value in loss_out.items():
                    if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
                        loss_items[key] = float(value.detach().mean().cpu())
            else:
                loss_total = _accumulate(loss_out)
        else:
            loss_total = _accumulate(loss_out)
    except Exception:
        logger.exception("_normalize_losses falhou; retornando 0.0")
        return torch.tensor(0.0, device=device), {}

    if loss_total.dim() > 0:
        loss_total = loss_total.mean()

    if not loss_total.is_floating_point() and not loss_total.is_complex():
        loss_total = loss_total.float()

    return loss_total, loss_items


class _ForwardToLoggerHandler(logging.Handler):
    def __init__(self, forward: Optional[Logger]):
        super().__init__()
        self.forward = forward

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - integrações externas
        if not self.forward:
            return
        msg = self.format(record)
        self.forward(msg)


def _safe_stream(prefer: str, log_dir: Path):
    cand = getattr(sys, f"__{prefer}__", None)
    if cand is not None and hasattr(cand, "fileno"):
        try:
            cand.fileno()
            return cand
        except Exception:
            pass
    cand2 = getattr(sys, prefer, None)
    if cand2 is not None and hasattr(cand2, "fileno"):
        try:
            cand2.fileno()
            return cand2
        except Exception:
            pass
    log_dir.mkdir(parents=True, exist_ok=True)
    return open(log_dir / f"fallback_{prefer}.log", "a", encoding="utf-8")


def _configure_logging(
    verbose: bool,
    log_dir: Path,
    external_logger: Optional[Logger],
    stream_override=None,
    logger_name: str = "ssd_train",
    log_prefix: str = "ssd_train",
) -> tuple[logging.Logger, Path]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    log_dir = log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{log_prefix}_{datetime.now():%Y%m%d_%H%M%S}.log"
    formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler(stream=stream_override or _safe_stream("stdout", log_dir))
    stream_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream_handler.setFormatter(formatter)
    stream_handler.flush = stream_handler.stream.flush  # type: ignore[assignment]
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if external_logger:
        forward_handler = _ForwardToLoggerHandler(external_logger)
        forward_handler.setLevel(logging.DEBUG)
        forward_handler.setFormatter(formatter)
        logger.addHandler(forward_handler)

    logger.info("Logger configurado.")
    logger.info("Logs serão salvos em %s", log_path)
    return logger, log_path


def _current_git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parent.parent
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        return commit or None
    except Exception:
        return None


def _infer_num_classes_from_model(model: torch.nn.Module) -> Optional[int]:
    candidates = [
        getattr(model, "num_classes", None),
        getattr(model, "nc", None),
    ]
    head = getattr(model, "head", None)
    if head is not None:
        candidates.append(getattr(head, "num_classes", None))
        classification_head = getattr(head, "classification_head", None)
        if classification_head is not None:
            candidates.append(getattr(classification_head, "num_classes", None))
    for cand in candidates:
        if isinstance(cand, int) and cand > 0:
            return cand
    return None


def _infer_num_classes_from_dataset(dataset: Any) -> Optional[int]:
    candidates = [getattr(dataset, "num_classes", None)]
    for cand in candidates:
        if isinstance(cand, int) and cand > 0:
            return cand
    return None


def _normalize_dataset_class_count(raw_num_classes: Optional[int]) -> Optional[int]:
    if raw_num_classes is None:
        return None
    if isinstance(raw_num_classes, int) and raw_num_classes > 0:
        return raw_num_classes
    return None


def _run_ssd_probe(dataloader: DataLoader, logger: logging.Logger, limit: int = 50) -> None:
    logger.info("[STAGE=probe] Iniciando verificação de %d amostras antes do treino.", limit)
    processed = 0
    for batch_idx, (images, targets) in enumerate(dataloader, start=1):
        processed += len(images)
        logger.debug(
            "[STAGE=probe] batch=%d tamanho=%d primeiro_img_shape=%s",
            batch_idx,
            len(images),
            getattr(images[0], "shape", None) if images else None,
        )
        if processed >= limit:
            break
    logger.info("[STAGE=probe] Finalizado. Amostras processadas=%d", processed)


def _dump_ssd_debug_snapshot(out_dir: Path, payload: dict, logger: logging.Logger) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = out_dir / f"failure_{int(time.time())}.json"
    snapshot_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.error("[STAGE=debug] Snapshot salvo em %s", snapshot_path)


def _infer_dataset_classes_from_annotations(
    ann_path: Path, split: str, logging_logger: logging.Logger
) -> tuple[Optional[int], list[int]]:
    """Infer classes directly from the COCO annotations file."""

    try:
        data = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - leitura robusta
        logging_logger.warning("[DATASET] Não foi possível ler %s para inferir classes: %s", ann_path, exc)
        return None, []

    categories = data.get("categories")
    if not isinstance(categories, list):
        logging_logger.warning("[DATASET] Formato de categorias inválido em %s", ann_path)
        return None, []

    coco_cat_ids: list[int] = []
    for cat in categories:
        try:
            coco_cat_ids.append(int(cat["id"]))
        except Exception:
            logging_logger.warning("[DATASET] Categoria sem id válida detectada no split %s: %s", split, cat)

    coco_cat_ids = sorted(coco_cat_ids)
    dataset_num_classes = len(categories)
    logging_logger.info("[DATASET] COCO categories=%d ids=%s", dataset_num_classes, coco_cat_ids)
    return dataset_num_classes, coco_cat_ids


def _log_label_range_from_annotations(
    annotations: dict, split: str, logging_logger: logging.Logger
) -> None:
    labels: list[int] = []
    for ann in annotations.get("annotations", []):
        try:
            labels.append(int(ann.get("category_id")))
        except Exception:
            continue

    if not labels:
        logging_logger.info("[AUDIT][%s] Nenhum label encontrado nas anotações.", split)
        return

    logging_logger.info(
        "[AUDIT][%s] Range de labels observado: min=%d max=%d", split, min(labels), max(labels)
    )


def _is_visdrone_dataset(dataset_dir: Path, dataset: Any) -> bool:
    def _matches(path_like: Any) -> bool:
        if isinstance(path_like, (str, Path)):
            return "visdrone" in str(path_like).lower()
        return False

    if _matches(dataset_dir):
        return True

    for attr in ("root", "dataset_dir", "images_dir", "annotations_dir"):
        if _matches(getattr(dataset, attr, None)):
            return True

    annotations = getattr(dataset, "annotations", None)
    if isinstance(annotations, dict):
        info = annotations.get("info")
        description = info.get("description") if isinstance(info, dict) else None
        if _matches(description):
            return True

    return False


def _remap_retinanet_labels(
    targets: list[dict[str, torch.Tensor]], *, num_classes: int, expects_background: bool, label_offset: int, logger: logging.Logger
) -> None:
    global _retinanet_label_log_before, _retinanet_label_log_after, _retinanet_label_mode_logged

    if num_classes <= 0:
        raise ValueError("[RETINANET] num_classes inválido para remapeamento de labels.")

    observed_keys: set[str] = set()
    for target in targets:
        if isinstance(target, dict):
            observed_keys.update(target.keys())

    def _log_mode(mode: str) -> None:
        nonlocal observed_keys
        if not _retinanet_label_mode_logged:
            _retinanet_label_mode_logged = True
            logger.info(
                "[RETINANET][REMAP] mode=%s keys=%s", mode, sorted(observed_keys)
            )

    labels_before: list[torch.Tensor] = []
    labels_present = False
    zero_based_ok = True
    for target in targets:
        labels = target.get("labels") if isinstance(target, dict) else None
        if not torch.is_tensor(labels):
            zero_based_ok = False
            continue
        labels_present = True
        if labels.dtype != torch.int64:
            zero_based_ok = False
        if labels.numel() > 0:
            labels_before.append(labels)
            min_label = int(labels.min())
            max_label = int(labels.max())
            if min_label < 0 or max_label >= num_classes:
                zero_based_ok = False

    if labels_before and not _retinanet_label_log_before:
        merged = torch.cat(labels_before)
        _retinanet_label_log_before = True
        logger.info(
            "[RETINANET] label range before remap: min=%d max=%d num_classes=%d", int(merged.min()), int(merged.max()), num_classes
        )

    if zero_based_ok and labels_present:
        for target in targets:
            labels = target.get("labels") if isinstance(target, dict) else None
            if torch.is_tensor(labels):
                target["labels"] = labels.to(torch.int64)
        _log_mode("SKIP_ALREADY_0_BASED")
        if labels_before and not _retinanet_label_log_after:
            merged = torch.cat(labels_before)
            _retinanet_label_log_after = True
            logger.info(
                "[RETINANET] label range after remap: min=%d max=%d num_classes=%d (background=%s)",
                int(merged.min()),
                int(merged.max()),
                num_classes,
                "sim" if expects_background else "não",
            )
        return

    fallback_warned = False
    for target in targets:
        if not isinstance(target, dict):
            continue
        labels_coco = target.get("labels_coco")
        if not torch.is_tensor(labels_coco):
            labels_coco = target.get("category_id_coco")
        if not torch.is_tensor(labels_coco):
            category_id = target.get("category_id")
            if torch.is_tensor(category_id) and category_id.numel() > 0 and (category_id >= 1).all():
                labels_coco = category_id
                if not fallback_warned:
                    fallback_warned = True
                    logger.warning(
                        "[RETINANET][REMAP] labels_coco ausente; usando category_id como fallback (presumindo formato COCO)."
                    )
        if not torch.is_tensor(labels_coco):
            raise ValueError(
                "[RETINANET] labels_coco/category_id_coco ausente para remapeamento; verifique o dataset COCO."
            )

        labels_coco = labels_coco.to(torch.int64)
        target["labels_coco"] = labels_coco

        if labels_coco.numel() > 0 and (labels_coco <= 0).any():
            bad_example = int(labels_coco.min())
            raise ValueError(
                f"[RETINANET] category_id COCO inválido (<=0) detectado no remap. Exemplo: {bad_example}."
            )

        target["labels"] = labels_coco.to(torch.int64) - label_offset

    _log_mode("REMAP_FROM_COCO")

    labels_after: list[torch.Tensor] = []
    for target in targets:
        labels = target.get("labels") if isinstance(target, dict) else None
        if torch.is_tensor(labels) and labels.numel() > 0:
            labels_after.append(labels)

    if labels_after:
        merged = torch.cat(labels_after)
        # Internamente as classes são sempre 0-based e contíguas; background é implícito.
        lower_bound = 0
        upper_bound = num_classes - 1
        if (merged < lower_bound).any() or (merged > upper_bound).any():
            raise ValueError(
                (
                    f"[RETINANET] Labels fora do intervalo após remap (esperado {lower_bound}..{upper_bound}). "
                    f"Exemplo min={int(merged.min())} max={int(merged.max())}"
                )
            )
        if not _retinanet_label_log_after:
            _retinanet_label_log_after = True
            logger.info(
                "[RETINANET] label range after remap: min=%d max=%d num_classes=%d (background=%s)",
                int(merged.min()),
                int(merged.max()),
                num_classes,
                "sim" if expects_background else "não",
            )


def _maybe_remap_retinanet_targets(
    model: torch.nn.Module,
    targets: list[dict[str, torch.Tensor]],
    *,
    num_classes: int,
    expects_background: bool,
    label_offset: int,
    logger: logging.Logger,
) -> None:
    if isinstance(model, RetinaNet):
        _remap_retinanet_labels(
            targets,
            num_classes=num_classes,
            expects_background=expects_background,
            label_offset=label_offset,
            logger=logger,
        )


def _validate_targets_batch(
    targets: list[dict[str, torch.Tensor]],
    *,
    num_classes: Optional[int],
    image_sizes: Iterable[tuple[int, int]],
    allow_zero_label: bool = False,
    logger: logging.Logger,
) -> None:
    sizes = list(image_sizes)
    for idx, target in enumerate(targets):
        path_hint = target.get("img_path") if isinstance(target, dict) else None
        path_label = f" (img={path_hint})" if path_hint else ""
        if not isinstance(target, dict):
            raise TypeError(f"Target no índice {idx}{path_label} não é um dicionário: {type(target).__name__}")

        if "boxes" not in target or "labels" not in target:
            raise ValueError(f"Target no índice {idx}{path_label} deve conter 'boxes' e 'labels'")

        boxes = target["boxes"]
        labels = target["labels"]

        if not torch.is_tensor(boxes):
            raise TypeError(f"Target boxes no índice {idx}{path_label} não é tensor")
        if not torch.is_tensor(labels):
            raise TypeError(f"Target labels no índice {idx}{path_label} não é tensor")

        if boxes.dim() != 2 or boxes.size(1) != 4:
            raise ValueError(
                f"Boxes no índice {idx}{path_label} devem ter shape [N,4], recebido {tuple(boxes.shape)}"
            )
        if boxes.dtype != torch.float32:
            logger.debug("Convertendo boxes para float32 no índice %d%s", idx, path_label)
            boxes = boxes.float()
            target["boxes"] = boxes
        if not torch.isfinite(boxes).all():
            raise ValueError(f"Boxes não finitos no índice {idx}{path_label}")
        if (boxes < 0).any():
            raise ValueError(f"Boxes negativos encontrados no índice {idx}{path_label}")

        if labels.dim() != 1:
            raise ValueError(f"Labels no índice {idx}{path_label} devem ser 1D, recebido {tuple(labels.shape)}")
        if boxes.size(0) != labels.numel():
            raise ValueError(
                f"Mismatch boxes/labels no índice {idx}{path_label}: {boxes.size(0)} vs {labels.numel()}"
            )
        if labels.dtype != torch.int64:
            raise TypeError(f"Labels no índice {idx}{path_label} devem ser int64, recebido {labels.dtype}")

        if boxes.numel() == 0:
            if boxes.shape != (0, 4):
                target["boxes"] = boxes.reshape(0, 4)
            if labels.shape != (0,):
                target["labels"] = labels.reshape(0)
            continue

        h, w = sizes[idx] if idx < len(sizes) else (None, None)
        if h is not None and w is not None:
            clamped_boxes = boxes.clone()
            clamped_boxes[:, 0::2] = clamped_boxes[:, 0::2].clamp(0, w)
            clamped_boxes[:, 1::2] = clamped_boxes[:, 1::2].clamp(0, h)
            target["boxes"] = clamped_boxes
            boxes = clamped_boxes

        x_min, y_min, x_max, y_max = boxes.unbind(dim=1)
        if not (x_max > x_min).all():
            raise ValueError(f"xmax <= xmin encontrado no índice {idx}{path_label}")
        if not (y_max > y_min).all():
            raise ValueError(f"ymax <= ymin encontrado no índice {idx}{path_label}")

        # Internamente as classes são sempre 0-based e contíguas; background é implícito.
        lower_bound = 0
        if num_classes is not None and num_classes > 0:
            upper_bound = num_classes - 1
            if (labels < lower_bound).any():
                raise ValueError(
                    (
                        f"Labels fora do intervalo no índice {idx}{path_label}: mínimo {int(labels.min())}"
                        f" < limite inferior ({lower_bound})"
                    )
                )
            if (labels > upper_bound).any():
                raise ValueError(
                    (
                        f"Labels fora do intervalo no índice {idx}{path_label}: máximo {int(labels.max())}"
                        f" > limite superior ({upper_bound})"
                    )
                )
        else:
            if (labels < lower_bound).any():
                raise ValueError(
                    f"Labels fora do intervalo [0,+inf) quando num_classes é desconhecido (índice {idx}{path_label})"
                )

        if not torch.isfinite(labels.float()).all():
            raise ValueError(f"Labels não finitos no índice {idx}{path_label}")



def _collect_batch_identifiers(
    targets: list[dict[str, torch.Tensor]], dataset: Optional[Any]
) -> tuple[list[int], list[str]]:
    image_ids: list[int] = []
    file_paths: list[str] = []

    for tgt in targets:
        img_id = tgt.get("image_id") if isinstance(tgt, dict) else None
        if torch.is_tensor(img_id):
            try:
                image_ids.extend(int(v) for v in img_id.detach().cpu().flatten().tolist())
            except Exception:
                continue
        if isinstance(tgt, dict) and "img_path" in tgt:
            img_path = tgt.get("img_path")
            if isinstance(img_path, (str, Path)):
                try:
                    file_paths.append(str(Path(img_path)))
                except Exception:
                    file_paths.append(str(img_path))

    if dataset is None or not image_ids:
        return image_ids, file_paths

    try:
        if hasattr(dataset, "images_dir") and hasattr(dataset, "images"):
            mapping = {int(img.get("id")): img.get("file_name") for img in getattr(dataset, "images", []) if isinstance(img, dict)}
            images_dir = Path(getattr(dataset, "images_dir"))
            for img_id in image_ids:
                fname = mapping.get(int(img_id))
                if fname:
                    file_paths.append(str(images_dir / Path(fname).name))
        elif hasattr(dataset, "images_dir") and hasattr(dataset, "image_ids"):
            resolver = getattr(dataset, "_resolve_image_path", None)
            images_dir = Path(getattr(dataset, "images_dir"))
            id_list = list(getattr(dataset, "image_ids"))
            for img_id in image_ids:
                if 0 <= img_id < len(id_list):
                    image_key = id_list[img_id]
                    if callable(resolver):
                        file_paths.append(str(resolver(image_key)))
                    else:
                        file_paths.append(str(images_dir / image_key))
    except Exception:
        pass

    return image_ids, file_paths


def _summarize_targets_for_logging(targets: list[dict[str, torch.Tensor]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for tgt in targets:
        if not isinstance(tgt, dict):
            continue
        entry: dict[str, Any] = {}
        img_path = tgt.get("img_path")
        if isinstance(img_path, (str, Path)):
            entry["img_path"] = str(img_path)
        boxes = tgt.get("boxes")
        if torch.is_tensor(boxes):
            try:
                entry["boxes"] = boxes.detach().cpu().tolist()
            except Exception:
                entry["boxes"] = "<unavailable>"
        labels = tgt.get("labels")
        if torch.is_tensor(labels):
            try:
                entry["labels"] = labels.detach().cpu().tolist()
            except Exception:
                entry["labels"] = "<unavailable>"
        if entry:
            summary.append(entry)
    return summary


def _build_detection_transforms(img_size: int, logger: logging.Logger):
    logger.info("[TRANSFORM] Aplicando resize fixo para %dx%d + ToTensor em train/val", img_size, img_size)
    return DetectionTransformCompose([DetectionResize(img_size), DetectionToTensor()])


def _audit_dataset(
    dataset: Any,
    phase: str,
    *,
    num_classes: Optional[int],
    logger: logging.Logger,
    limit: Optional[int] = None,
    allow_zero_label: bool = False,
) -> None:
    logger.info("[AUDIT] Iniciando auditoria do dataset (%s)...", phase)
    total = len(dataset)
    inspected = 0
    with torch.no_grad():
        for idx in range(total):
            if limit is not None and inspected >= limit:
                break
            image, target = dataset[idx]
            h, w = (int(image.shape[-2]), int(image.shape[-1])) if torch.is_tensor(image) else (image.height, image.width)
            try:
                _validate_targets_batch(
                    [target], num_classes=num_classes, image_sizes=[(h, w)], allow_zero_label=allow_zero_label, logger=logger
                )
            except Exception as exc:
                raise ValueError(f"[AUDIT] Falha no {phase} idx={idx} img={target.get('img_path', '<desconhecido>')}: {exc}")
            inspected += 1
    logger.info("[AUDIT] Auditoria concluída para %s: %d/%d amostras verificadas", phase, inspected, total)


def _audit_retinanet_label_distribution(
    dataset: Any, phase: str, logger: logging.Logger, *, max_samples: int = 50
) -> dict[str, Any]:
    labels_observed: list[int] = []
    unique_sample: set[int] = set()
    inspected = 0
    with torch.no_grad():
        for idx in range(len(dataset)):
            if inspected >= max_samples:
                break
            try:
                _, tgt = dataset[idx]
            except Exception:
                continue
            labels = tgt.get("labels") if isinstance(tgt, dict) else None
            if not torch.is_tensor(labels) or labels.numel() == 0:
                continue
            inspected += 1
            labels_np = labels.detach().cpu()
            labels_observed.extend(int(x) for x in labels_np.tolist())
            unique_sample.update(int(x) for x in labels_np.unique().tolist())

    summary: dict[str, Any] = {}
    if labels_observed:
        summary["min"] = min(labels_observed)
        summary["max"] = max(labels_observed)
        summary["unique_sample"] = sorted(unique_sample)[:20]
        summary["starts_at_one"] = summary["min"] >= 1
        summary["counted_images"] = inspected
    else:
        summary = {"min": None, "max": None, "unique_sample": [], "starts_at_one": None, "counted_images": inspected}

    logger.info(
        "[RETINANET][CLASSES] %s labels_observed: min=%s max=%s unique_sample=%s inspected=%d",
        phase,
        summary.get("min"),
        summary.get("max"),
        summary.get("unique_sample"),
        inspected,
    )
    return summary


def _ensure_frcnn_head(model: torch.nn.Module, num_classes: int, logger: logging.Logger) -> None:
    roi_heads = getattr(model, "roi_heads", None)
    if roi_heads is None or not hasattr(roi_heads, "box_predictor"):
        logger.debug("[MODEL] roi_heads.box_predictor indisponível; nada a reconfigurar")
        return
    predictor = getattr(roi_heads, "box_predictor", None)
    in_features = None
    if predictor is not None and hasattr(predictor, "cls_score"):
        in_features = predictor.cls_score.in_features
    if in_features is None and hasattr(roi_heads, "box_predictor"):
        in_features = roi_heads.box_predictor.cls_score.in_features  # type: ignore[attr-defined]
    if in_features is None:
        logger.warning("[MODEL] Não foi possível identificar in_features do box_predictor; mantendo configuração atual")
        return
    roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    logger.info("[MODEL] Faster R-CNN configurado com num_classes=%d (incluindo background)", num_classes)


def _summarize_boxes_for_debug(targets: list[dict[str, Any]]) -> list[str]:
    summaries = []
    for tgt in targets:
        boxes = tgt.get("boxes") if isinstance(tgt, dict) else None
        if torch.is_tensor(boxes) and boxes.numel() > 0:
            summaries.append(
                f"shape={tuple(boxes.shape)} min={boxes.min().item():.2f} max={boxes.max().item():.2f}"
            )
        else:
            summaries.append("shape=(0, 4) [sem boxes]")
    return summaries


def _run_smoke_test_val_loss(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
    *,
    max_images: int = 8,
    num_classes: Optional[int] = None,
) -> None:
    logger.info("[SMOKE] Iniciando smoke_test_val_loss com limite de %d imagens", max_images)
    processed_images = 0
    total_loss = 0.0
    component_sum: dict[str, float] = {}
    batches = 0
    model.eval()
    expects_background = isinstance(model, FasterRCNN)
    allow_zero_label = isinstance(model, RetinaNet)
    label_offset = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [
                {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
                for t in targets
            ]
            image_sizes = [(int(img.shape[-2]), int(img.shape[-1])) for img in images]
            _validate_targets_batch(
                targets,
                num_classes=num_classes,
                image_sizes=image_sizes,
                allow_zero_label=allow_zero_label,
                logger=logger,
            )
            _maybe_remap_retinanet_targets(
                model,
                targets,
                num_classes=num_classes or 0,
                expects_background=False,
                label_offset=label_offset,
                logger=logger,
            )
            loss_out = model(images, targets)
            batch_loss = torch.tensor(0.0, device=device)
            if isinstance(loss_out, dict) and loss_out:
                for key, value in loss_out.items():
                    val_tensor = value.mean() if torch.is_tensor(value) else torch.tensor(float(value), device=device)
                    batch_loss = batch_loss + val_tensor
                    component_sum[key] = component_sum.get(key, 0.0) + float(val_tensor.detach().cpu())
            total_loss += float(batch_loss.detach().cpu())
            batches += 1
            processed_images += len(images)
            logger.info(
                "[SMOKE] batch=%d loss=%.4f boxes=%s",
                batches,
                float(batch_loss.detach().cpu()),
                _summarize_boxes_for_debug(targets),
            )
            if processed_images >= max_images:
                break
    if batches == 0:
        logger.warning("[SMOKE] Nenhum batch processado no smoke_test_val_loss")
        return
    avg_loss = total_loss / batches
    comp_mean = {k: v / batches for k, v in component_sum.items()}
    logger.info("[SMOKE] loss médio/batch=%.4f componentes=%s", avg_loss, comp_mean)


def run_detection_sanity_check(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
    *,
    num_classes: int,
    threshold: float = 0.25,
    max_images: int = 5,
) -> None:
    model.eval()
    checked = 0
    detections = 0
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for output in outputs:
                if not all(key in output for key in ("boxes", "scores", "labels")):
                    raise RuntimeError("[SANITY] Saída do modelo não contém boxes/scores/labels.")
                labels = output.get("labels")
                if torch.is_tensor(labels) and labels.numel() > 0:
                    if labels.min() < 0 or labels.max() > num_classes - 1:
                        raise RuntimeError(
                            f"[SANITY] Labels fora do intervalo [0,{num_classes-1}]: min={int(labels.min())} max={int(labels.max())}"
                        )
                    detections += int((output.get("scores") > threshold).sum().item())
            checked += len(images)
            if checked >= max_images:
                break
    logger.info(
        "[SANITY] Verificação rápida: imagens=%d detecções_acima_threshold=%d threshold=%.2f",
        checked,
        detections,
        threshold,
    )


@contextmanager
def _watchdog_paused(last_progress: list[float], pause_event: Optional[threading.Event]):
    try:
        if pause_event:
            last_progress[0] = time.monotonic()
            pause_event.set()
        yield
    finally:
        if pause_event:
            last_progress[0] = time.monotonic()
            pause_event.clear()


def _start_watchdog(
    log_path: Path,
    last_progress: list[float],
    stop_event: threading.Event,
    logger: logging.Logger,
    safe_stderr=None,
    timeout: int = 300,
    pause_event: Optional[threading.Event] = None,
):
    def _watch() -> None:  # pragma: no cover - monitoramento em tempo real
        while not stop_event.wait(5):
            if pause_event and pause_event.is_set():
                continue
            if time.monotonic() - last_progress[0] > timeout:
                logger.warning("[WATCHDOG] Nenhum progresso de batch há mais de %s segundos.", timeout)
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [WATCHDOG] dump de stack após {timeout}s sem progresso.\\n")
                    faulthandler.dump_traceback(file=fh)
                target_stream = safe_stderr or _safe_stream("stderr", log_path.parent)
                faulthandler.dump_traceback(file=target_stream)
                last_progress[0] = time.monotonic()

    thread = threading.Thread(target=_watch, name="ssd-train-watchdog", daemon=True)
    thread.start()
    return thread


def _split_dataset(dataset, val_ratio: float, seed: int, logging_logger: Optional[logging.Logger] = None):
    """Split determinístico de train/val por índices com auditoria rigorosa."""

    if val_ratio <= 0:
        return dataset, None

    logger = logging_logger or logging.getLogger("ssd_train")

    total_indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(total_indices)

    n_val = int(round(len(total_indices) * val_ratio))
    n_val = min(max(n_val, 0), len(total_indices))

    val_indices = sorted(total_indices[:n_val])
    train_indices = sorted(total_indices[n_val:])

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_set = set(train_indices)
    val_set = set(val_indices)
    intersection = train_set & val_set
    union = train_set | val_set

    if intersection:
        raise RuntimeError(f"[SPLIT] Interseção não vazia entre train e val (size={len(intersection)}).")
    if len(union) != len(dataset):
        raise RuntimeError(
            f"[SPLIT] União train/val ({len(union)}) difere do total do dataset ({len(dataset)})."
        )

    train_min = min(train_indices) if train_indices else None
    train_max = max(train_indices) if train_indices else None
    val_min = min(val_indices) if val_indices else None
    val_max = max(val_indices) if val_indices else None

    logger.info("[SPLIT] Total samples: %d", len(dataset))
    logger.info("[SPLIT] Train samples: %d", len(train_indices))
    logger.info("[SPLIT] Val samples: %d", len(val_indices))
    logger.info("[SPLIT] Intersection(train,val)=%d", len(intersection))
    logger.info("[SPLIT] Union size=%d", len(union))
    logger.info("[SPLIT] Train idx range: %s - %s", train_min, train_max)
    logger.info("[SPLIT] Val idx range: %s - %s", val_min, val_max)

    return train_subset, val_subset


def _extract_state_dict(loaded: Any) -> tuple[Dict[str, torch.Tensor], str]:
    if isinstance(loaded, dict):
        model_obj = loaded.get("model")
        if isinstance(model_obj, dict):
            return model_obj, "checkpoint.model"
        state_dict = loaded.get("state_dict")
        if isinstance(state_dict, dict):
            return state_dict, "checkpoint.state_dict"
        if loaded and all(torch.is_tensor(v) for v in loaded.values()):
            return loaded, "state_dict"
    raise ValueError("Formato de checkpoint não suportado para carregamento de pesos.")


def _extract_checkpoint_meta(loaded: Any) -> dict:
    if isinstance(loaded, dict):
        meta = loaded.get("meta")
        if isinstance(meta, dict):
            return meta
    return {}


def _detect_checkpoint_num_classes(state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
    predictor_weight = state_dict.get("roi_heads.box_predictor.cls_score.weight")
    if predictor_weight is not None and hasattr(predictor_weight, "shape") and len(predictor_weight.shape) > 0:
        return int(predictor_weight.shape[0])
    return None


def _strip_box_predictor_head(
    state_dict: Dict[str, torch.Tensor],
) -> tuple[Dict[str, torch.Tensor], list[str]]:
    box_keys = {
        "roi_heads.box_predictor.cls_score.weight",
        "roi_heads.box_predictor.cls_score.bias",
        "roi_heads.box_predictor.bbox_pred.weight",
        "roi_heads.box_predictor.bbox_pred.bias",
    }
    ignored = [key for key in state_dict if key in box_keys]
    filtered = {key: value for key, value in state_dict.items() if key not in box_keys}
    return filtered, ignored


def _load_frcnn_weights_with_head_guard(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    model_num_classes: int,
    emit_info: Callable[[str], None],
    emit_warning: Callable[[str], None],
    logging_logger: logging.Logger,
) -> tuple[list[str], list[str], Optional[int], bool, list[str], bool]:
    ckpt_num_classes = _detect_checkpoint_num_classes(state_dict)
    head_mismatch = ckpt_num_classes is not None and ckpt_num_classes != model_num_classes
    filtered_state_dict = state_dict
    ignored_keys: list[str] = []
    strict_load = not head_mismatch

    if head_mismatch:
        emit_warning(
            f"[WEIGHTS] class-mismatch ckpt_num_classes={ckpt_num_classes} model_num_classes={model_num_classes}"
        )
        filtered_state_dict, ignored_keys = _strip_box_predictor_head(state_dict)
        emit_info(f"[WEIGHTS] Ignorando chaves do box_predictor: {ignored_keys}")

    try:
        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=strict_load)
    except RuntimeError as exc:
        emit_warning(
            f"[WEIGHTS] Falha ao carregar pesos com strict={strict_load}: {exc}; tentando ignorar box_predictor."
        )
        filtered_state_dict, ignored_keys = _strip_box_predictor_head(state_dict)
        head_mismatch = True
        strict_load = False
        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        if ckpt_num_classes is None:
            ckpt_num_classes = _detect_checkpoint_num_classes(state_dict)

    if missing:
        emit_info(f"[WEIGHTS] missing_keys: {missing}")
    if unexpected:
        emit_info(f"[WEIGHTS] unexpected_keys: {unexpected}")
    if head_mismatch and not ignored_keys:
        # Garantia de logging mesmo se a lista estiver vazia (ex.: state_dict já sem head)
        ignored_keys = ["roi_heads.box_predictor.cls_score", "roi_heads.box_predictor.bbox_pred"]
        logging_logger.debug("[WEIGHTS] Nenhuma chave do head presente para ignorar; registrando padrões")

    return missing, unexpected, ckpt_num_classes, head_mismatch, ignored_keys, strict_load


def _ensure_pycocotools() -> tuple[type, type]:
    spec = importlib.util.find_spec("pycocotools")
    if spec is None:
        raise ImportError(
            "pycocotools ausente. Instale com pip install pycocotools (Windows: pycocotools-windows ou equivalente)."
        )
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    return COCO, COCOeval


def _extract_image_id(target: dict) -> int:
    image_id_val = target.get("image_id")
    if torch.is_tensor(image_id_val):
        return int(image_id_val.item())
    if isinstance(image_id_val, (list, tuple)) and image_id_val:
        return int(image_id_val[0])
    return int(image_id_val)


def _extract_file_name(target: dict) -> str | None:
    file_name = target.get("file_name") or target.get("image_path") or target.get("img_path")
    if file_name is None:
        return None
    try:
        return Path(str(file_name)).name
    except Exception:
        return str(file_name)


def _coco_box_from_xyxy(box: torch.Tensor) -> list[float]:
    xmin, ymin, xmax, ymax = box.tolist()
    return [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]


def _flatten_subset_indices(dataset) -> tuple[Any, list[int] | None]:
    indices: list[int] | None = None
    base = dataset
    while isinstance(base, Subset):
        current_indices = list(base.indices)
        if indices is None:
            indices = current_indices
        else:  # handle nested subsets
            indices = [indices[i] for i in current_indices]
        base = base.dataset
    return base, indices


def _extract_coco_subset_from_dataset(dataset: Any) -> dict[str, Any] | None:
    base_ds, indices = _flatten_subset_indices(dataset)
    images = getattr(base_ds, "images", None)
    annotations = getattr(base_ds, "annotations", None)
    if images is None or annotations is None:
        return None

    images_list = images
    if indices is not None:
        images_list = [images[i] for i in indices]

    image_ids = [img.get("id") for img in images_list if img.get("id") is not None]
    image_id_set = set(int(img_id) for img_id in image_ids)
    filtered_annotations = [
        ann for ann in annotations.get("annotations", []) if int(ann.get("image_id", -1)) in image_id_set
    ]

    file_name_to_id = {}
    for img in images_list:
        img_id = img.get("id")
        file_name = img.get("file_name")
        if img_id is None or file_name is None:
            continue
        try:
            file_name_to_id[Path(str(file_name)).name] = int(img_id)
        except Exception:
            continue

    return {
        "images": images_list,
        "annotations": filtered_annotations,
        "categories": annotations.get("categories", []),
        "image_ids": image_id_set,
        "file_name_to_id": file_name_to_id,
    }


def _rescale_boxes_to_original(
    boxes: torch.Tensor, target: dict[str, Any], current_size: tuple[int, int], logger: logging.Logger, *, tag: str
) -> torch.Tensor:
    orig_size = target.get("orig_size")
    if torch.is_tensor(orig_size):
        orig_h, orig_w = int(orig_size[0]), int(orig_size[1])
    elif isinstance(orig_size, (list, tuple)) and len(orig_size) >= 2:
        orig_h, orig_w = int(orig_size[0]), int(orig_size[1])
    else:
        logger.warning("%s [AUDIT] orig_size ausente para image_id=%s; boxes não serão reescaladas.", tag, target.get("image_id"))
        return boxes

    cur_h, cur_w = current_size
    if cur_h <= 0 or cur_w <= 0:
        logger.warning(
            "%s [AUDIT] current_size inválido (%s, %s) para image_id=%s; boxes não serão reescaladas.",
            tag,
            cur_h,
            cur_w,
            target.get("image_id"),
        )
        return boxes

    scale_x = float(orig_w) / float(cur_w)
    scale_y = float(orig_h) / float(cur_h)
    scaled = boxes.clone()
    scaled[:, 0] = boxes[:, 0] * scale_x
    scaled[:, 2] = boxes[:, 2] * scale_x
    scaled[:, 1] = boxes[:, 1] * scale_y
    scaled[:, 3] = boxes[:, 3] * scale_y
    return scaled


def _build_val_loader_and_classes(
    dataset_dir: Path,
    train_ann: Path,
    val_ann: Path,
    config: TrainConfig,
    logging_logger: logging.Logger,
    *,
    override_val_ratio: Optional[float] = None,
    expects_background: bool = True,
    force_dataset_num_classes: Optional[int] = None,
) -> tuple[DataLoader, int, int, int]:
    transform = _build_detection_transforms(config.imgsz, logging_logger)
    logging_logger.info("[SETUP] Construindo datasets COCO para validação pós-treinamento")
    train_ds_full = CocoDetectionDataset(dataset_dir / "images" / "train", train_ann, transforms=transform)
    val_ds_full = CocoDetectionDataset(dataset_dir / "images" / "val", val_ann, transforms=transform)

    def _sample_labels(ds: Any, limit: int = 32) -> list[int]:
        labels: list[int] = []
        for idx in range(min(len(ds), limit)):
            try:
                _, tgt = ds[idx]
            except Exception:
                continue
            tensor_labels = tgt.get("labels") if isinstance(tgt, dict) else None
            if torch.is_tensor(tensor_labels):
                labels.extend(int(x) for x in tensor_labels.detach().cpu().tolist())
        return labels

    if hasattr(train_ds_full, "cat_id_to_label"):
        logging_logger.info(
            summarize_class_mapping(
                dataset_name=getattr(train_ds_full, "dataset_name", None),
                k=getattr(train_ds_full, "num_classes", 0),
                class_names=getattr(train_ds_full, "class_names", []),
                categories=getattr(train_ds_full, "annotations", {}).get("categories", []),
                cat_id_to_label=getattr(train_ds_full, "cat_id_to_label", {}),
                label_to_cat_id=getattr(train_ds_full, "label_to_cat_id", {}),
                observed_labels=_sample_labels(train_ds_full),
            )
        )

    _log_label_range_from_annotations(getattr(val_ds_full, "annotations", {}), "val", logging_logger)

    total_before_split = len(train_ds_full)
    val_ratio = config.val_ratio if override_val_ratio is None else override_val_ratio
    split_applied = bool(val_ratio and val_ratio > 0)
    if split_applied:
        logging_logger.info("[SETUP] Aplicando split adicional train/val com val_ratio=%.3f seed=%s", val_ratio, config.seed)
        train_ds_full, extra_val = _split_dataset(train_ds_full, val_ratio, config.seed, logging_logger)
        val_ds_full = extra_val

    train_after_split = len(train_ds_full)
    val_after_split = len(val_ds_full) if split_applied else 0
    total_after_split = train_after_split + val_after_split

    if split_applied:
        logging_logger.info("[DATA] Dataset base (antes do split): total=%d", total_before_split)
        logging_logger.info(
            "[DATA] Após split: train=%d | val=%d | total=%d",
            train_after_split,
            val_after_split,
            total_after_split,
        )
    else:
        logging_logger.info(
            "[DATA] Sem split adicional: train=total=%d | val=%d",
            train_after_split,
            val_after_split,
        )

    describe_dataloader(train_ds_full, logging_logger.info, label="train subset")
    logging_logger.info("[SETUP] Tamanho train=%d | val=%d", len(train_ds_full), len(val_ds_full))

    configured_dataset_classes = None if force_dataset_num_classes is not None else getattr(config, "dataset_num_classes", None)
    configured_model_num_classes = None if force_dataset_num_classes is not None else getattr(config, "num_classes", None)

    ann_dataset_classes, _ = _infer_dataset_classes_from_annotations(val_ann, "val", logging_logger)
    inferred_train_classes = _normalize_dataset_class_count(_infer_num_classes_from_dataset(train_ds_full))
    inferred_val_classes = _normalize_dataset_class_count(_infer_num_classes_from_dataset(val_ds_full))

    dataset_num_classes: Optional[int] = None
    if force_dataset_num_classes is not None:
        dataset_num_classes = force_dataset_num_classes
        logging_logger.info(
            "[DATASET] Forçando dataset_num_classes do COCO JSON (split val): %s", dataset_num_classes
        )
    elif ann_dataset_classes is not None:
        dataset_num_classes = ann_dataset_classes
    elif configured_dataset_classes is not None:
        dataset_num_classes = configured_dataset_classes
    elif configured_model_num_classes is not None and configured_model_num_classes > 0:
        dataset_num_classes = (
            max(1, configured_model_num_classes - 1) if expects_background else configured_model_num_classes
        )
    else:
        dataset_num_classes = inferred_train_classes

    looks_like_visdrone = _is_visdrone_dataset(dataset_dir, train_ds_full) or _is_visdrone_dataset(dataset_dir, val_ds_full)

    if dataset_num_classes is None:
        dataset_num_classes = inferred_val_classes

    if ann_dataset_classes is not None and dataset_num_classes is not None and dataset_num_classes != ann_dataset_classes:
        logging_logger.warning(
            "[AUDIT] num_classes configurado/inferido (%s) difere do COCO (val=%s); usando configurado/inferido.",
            dataset_num_classes,
            ann_dataset_classes,
        )

    if dataset_num_classes is None:
        raise ValueError("Não foi possível inferir num_classes a partir do dataset; verifique as categorias.")

    if inferred_val_classes is not None and inferred_val_classes != dataset_num_classes:
        logging_logger.warning(
            "[AUDIT] num_classes do val (%s) difere do train (%s); usando o do train.",
            inferred_val_classes,
            dataset_num_classes,
        )

    expected_model_classes = dataset_num_classes + 1 if expects_background and dataset_num_classes > 0 else dataset_num_classes
    model_num_classes = expected_model_classes
    if force_dataset_num_classes is None and configured_model_num_classes is not None and configured_model_num_classes > 0:
        model_num_classes = configured_model_num_classes
        if configured_model_num_classes != expected_model_classes:
            logging_logger.warning(
                "[DATASET] num_classes configurado (%s) difere do esperado (%s); respeitando o configurado.",
                configured_model_num_classes,
                expected_model_classes,
            )

    logging_logger.info(
        "[MODEL] dataset_num_classes=%d model_num_classes=%d (incluindo background)",
        dataset_num_classes,
        model_num_classes,
    )

    if looks_like_visdrone:
        logging_logger.info(
            "[DATASET] VisDrone multi-class: dataset_classes=%d -> model_num_classes=%d (%s)",
            dataset_num_classes,
            model_num_classes,
            "incluindo background" if expects_background else "foreground only",
        )

    dataloader_kwargs = dict(
        batch_size=config.batch_size,
        collate_fn=coco_collate,
        drop_last=config.drop_last,
    )
    val_workers = config.num_workers
    pin_memory = config.pin_memory
    persistent_workers = config.persistent_workers
    prefetch_factor = config.prefetch_factor
    if config.debug_dataloader:
        logging_logger.warning("[DATALOADER] Modo debug ativado -> num_workers=0, persistent_workers=False, pin_memory=False")
        val_workers = 0
        pin_memory = False
        persistent_workers = False
        prefetch_factor = None
        dataloader_kwargs["drop_last"] = False
    if val_workers > 0:
        dataloader_kwargs.update(
            num_workers=val_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor
    else:
        dataloader_kwargs["num_workers"] = 0

    logging_logger.info(
        "[DATALOADER] (val) batch_size=%s num_workers=%s pin_memory=%s persistent_workers=%s prefetch_factor=%s drop_last=%s",
        dataloader_kwargs.get("batch_size"),
        dataloader_kwargs.get("num_workers"),
        dataloader_kwargs.get("pin_memory"),
        dataloader_kwargs.get("persistent_workers"),
        dataloader_kwargs.get("prefetch_factor"),
        dataloader_kwargs.get("drop_last"),
    )

    val_loader = DataLoader(val_ds_full, shuffle=False, **dataloader_kwargs)
    return val_loader, int(model_num_classes), int(dataset_num_classes), int(dataloader_kwargs.get("num_workers", 0))


def run_val_loss_loop(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device_str: str,
    num_classes: int,
    *,
    expects_background: bool,
    label_offset: int,
    logging_logger: logging.Logger,
    tag: str = "[VAL-POST]",
) -> dict:
    model.eval()
    val_loss_sum = 0.0
    val_component_sum: dict[str, float] = {}
    total_images = 0
    total_batches = 0
    return_type_logged = False

    with torch.no_grad():
        for vstep, (images, targets) in enumerate(val_loader, start=1):
            images = [img.to(device_str) for img in images]
            targets = [
                {k: (v.to(device_str) if torch.is_tensor(v) else v) for k, v in t.items()}
                for t in targets
            ]
            image_sizes = [(int(img.shape[-2]), int(img.shape[-1])) for img in images]
            _validate_targets_batch(
                targets,
                num_classes=num_classes,
                image_sizes=image_sizes,
                allow_zero_label=expects_background,
                logger=logging_logger,
            )

            model.train()
            _maybe_remap_retinanet_targets(
                model,
                targets,
                num_classes=num_classes,
                expects_background=expects_background,
                label_offset=label_offset,
                logger=logging_logger,
            )
            loss_out = model(images, targets)
            model.eval()

            if not return_type_logged:
                keys = list(loss_out.keys()) if isinstance(loss_out, dict) else None
                logging_logger.info("%s Return type detected: %s keys=%s", tag, type(loss_out).__name__, keys)
                return_type_logged = True

            if not isinstance(loss_out, dict) or not all(
                torch.is_tensor(v) and (v.is_floating_point() or v.is_complex()) for v in loss_out.values()
            ):
                raise RuntimeError(
                    "Validação pós-treinamento esperava um dicionário de losses; recebeu predições."
                )

            batch_loss_tensor = sum(loss_out.values())

            if not torch.isfinite(batch_loss_tensor):
                raise RuntimeError(f"Loss não finito detectado no val (step={vstep}).")

            total_batches += 1
            total_images += len(images)
            batch_loss_value = float(batch_loss_tensor.detach().cpu())
            val_loss_sum += batch_loss_value

            for key, value in loss_out.items():
                val_component_sum[key] = val_component_sum.get(key, 0.0) + float(value.detach().cpu())

    val_loss_mean_per_batch = val_loss_sum / max(1, total_batches)
    val_loss_mean_per_image = val_loss_sum / max(1, total_images)
    breakdown_mean = {k: v / max(1, total_batches) for k, v in val_component_sum.items()}
    logging_logger.info(f"{tag} loss/batch={val_loss_mean_per_batch:.4f} | loss/img={val_loss_mean_per_image:.4f}")
    if breakdown_mean:
        logging_logger.info("%s Breakdown médio por batch: %s", tag, breakdown_mean)

    return {
        "loss_per_batch": val_loss_mean_per_batch,
        "loss_per_image": val_loss_mean_per_image,
        "breakdown": breakdown_mean,
        "total_batches": total_batches,
        "total_images": total_images,
    }


def run_val_coco_metrics(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device_str: str,
    val_ann: Path,
    logging_logger: logging.Logger,
    *,
    tag: str = "[VAL-METRICS]",
    output_dir: Path,
    label_to_cat_id: dict[int, int] | None = None,
    legacy_drop_label: int | None = None,
) -> dict:
    COCO, COCOeval = _ensure_pycocotools()
    val_image_ids_set: set[int] = set()
    val_dataset = getattr(val_loader, "dataset", None)
    try:
        val_coco_imgs = getattr(getattr(val_dataset, "coco", None), "imgs", None)
        if val_coco_imgs:
            val_image_ids_set = set(int(x) for x in val_coco_imgs.keys())
    except Exception as exc:  # pragma: no cover - defensive logging
        logging_logger.warning("%s [AUDIT] Falha ao obter image_ids do val_dataset: %s", tag, exc)
    val_dataset_info = _extract_coco_subset_from_dataset(val_dataset)
    expected_val_images = len(val_dataset) if val_dataset is not None else None
    gt_source = "original"

    coco_gt = COCO(str(val_ann))

    if val_dataset_info is not None:
        val_image_ids = set(int(x) for x in val_dataset_info.get("image_ids", set()))
        gt_image_ids_from_loader = set(int(x) for x in coco_gt.getImgIds())
        if val_image_ids and val_image_ids != gt_image_ids_from_loader:
            effective_gt = {
                "images": val_dataset_info.get("images", []),
                "annotations": val_dataset_info.get("annotations", []),
                "categories": val_dataset_info.get("categories", []),
            }
            output_dir = output_dir.expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            effective_gt_path = output_dir / "gt_instances_val_effective.json"
            effective_gt_path.write_text(json.dumps(effective_gt, indent=2), encoding="utf-8")
            logging_logger.warning(
                "%s [AUDIT] ImageId mismatch (dataset=%d GT=%d). Usando GT efetivo: %s",
                tag,
                len(val_image_ids),
                len(gt_image_ids_from_loader),
                effective_gt_path,
            )
            coco_gt = COCO(str(effective_gt_path))
            val_ann = effective_gt_path
            gt_source = "effective_subset"

    logging_logger.info("%s Ground truth annotations: %s (source=%s)", tag, val_ann, gt_source)
    gt_img_ids = set(int(cat_id) for cat_id in coco_gt.getImgIds())
    gt_cat_ids = set(int(cat_id) for cat_id in coco_gt.getCatIds())
    gt_dict = coco_gt.dataset or {}
    gt_images = gt_dict.get("images", [])
    gt_filename_to_id: dict[str, int] = {}
    gt_id_to_filename: dict[int, str | None] = {}
    for img in gt_images:
        img_id = img.get("id")
        file_name = img.get("file_name")
        if img_id is None:
            continue
        int_img_id = int(img_id)
        gt_id_to_filename[int_img_id] = file_name
        if file_name:
            gt_filename_to_id[file_name] = int_img_id
            gt_filename_to_id[Path(str(file_name)).name] = int_img_id

    dataset_num_classes = len(gt_cat_ids)
    logging_logger.info("%s [AUDIT] GT imagens=%d categorias=%d", tag, len(gt_img_ids), len(gt_cat_ids))
    if gt_img_ids:
        logging_logger.info("%s [AUDIT] img_id range=[%s, %s]", tag, min(gt_img_ids), max(gt_img_ids))
    if gt_cat_ids:
        logging_logger.info("%s [AUDIT] cat_id set=%s", tag, sorted(gt_cat_ids))
    if not val_image_ids_set:
        val_image_ids_set = set(int(x) for x in val_dataset_info.get("image_ids", set())) if val_dataset_info else set()
    if not val_image_ids_set:
        val_image_ids_set = gt_img_ids

    is_retinanet = isinstance(model, RetinaNet)
    retinanet_uses_background = False

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions_coco.json"
    predictions_path.unlink(missing_ok=True)
    logging_logger.info("%s Gerando predictions_coco.json em %s", tag, predictions_path)

    if dataset_num_classes <= 0:
        raise RuntimeError(f"{tag} Não há categorias no GT para gerar métricas COCO.")

    categories = gt_dict.get("categories", [])
    class_to_cat_id = [int(cat["id"]) for cat in categories if "id" in cat]
    if not class_to_cat_id:
        class_to_cat_id = sorted(gt_cat_ids)
    mapping_override = label_to_cat_id if label_to_cat_id else None
    if legacy_drop_label is not None and mapping_override and len(mapping_override) == 1:
        only_cat = next(iter(mapping_override.values()))
        mapping_override = dict(mapping_override)
        mapping_override[legacy_drop_label + 1] = only_cat
    logging_logger.info("%s [AUDIT] class_to_cat_id mapping=%s override=%s", tag, class_to_cat_id, mapping_override)

    predictions: list[dict[str, float | int | list[float]]] = []
    model.eval()
    predicted_image_ids: set[int] = set()
    predicted_category_ids: set[int] = set()
    max_label_observed = 0
    invalid_label_count = 0
    invalid_label_examples: list[dict[str, float | int]] = []
    missing_image_mappings: list[dict[str, Any]] = []
    smoke_checked = 0
    smoke_invalid: list[dict[str, Any]] = []
    processed_images = 0
    fallback_target_image_ids: set[int] = set()
    dataset_file_name_to_id = val_dataset_info.get("file_name_to_id", {}) if val_dataset_info else {}

    def _map_label_to_category(raw_label: int) -> int | None:
        nonlocal invalid_label_count
        nonlocal invalid_label_examples
        if legacy_drop_label is not None and raw_label == legacy_drop_label:
            return None
        if mapping_override is not None:
            category_id = mapping_override.get(int(raw_label))
            idx = None
        else:
            idx = raw_label - 1 if not is_retinanet else raw_label
            category_id = None
        if idx is not None and (idx < 0 or idx >= len(class_to_cat_id)):
            invalid_label_count += 1
            if len(invalid_label_examples) < 5:
                invalid_label_examples.append({"label": raw_label})
            return None
        if category_id is None:
            category_id = class_to_cat_id[idx] if idx is not None else None
        if category_id not in gt_cat_ids:
            invalid_label_count += 1
            if len(invalid_label_examples) < 5:
                invalid_label_examples.append({"label": raw_label, "category_id": category_id})
            return None
        return category_id

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = [img.to(device_str) for img in images]
            outputs = model(images)
            processed_images += len(images)

            for output, target, image_tensor in zip(outputs, targets, images):
                raw_target_image_id = target.get("image_id")
                if raw_target_image_id is not None:
                    fallback_target_image_ids.add(int(raw_target_image_id))
                target_image_id = target.get("image_id")
                file_name = _extract_file_name(target)
                pred_image_id: int | None = None
                if target_image_id is not None:
                    candidate = _extract_image_id(target)
                    if candidate in val_image_ids_set:
                        pred_image_id = candidate
                if pred_image_id is None and file_name is not None:
                    pred_image_id = (
                        dataset_file_name_to_id.get(file_name)
                        or dataset_file_name_to_id.get(Path(file_name).name)
                        or gt_filename_to_id.get(file_name)
                        or gt_filename_to_id.get(Path(file_name).name)
                    )
                if pred_image_id is None and target_image_id is not None:
                    candidate = _extract_image_id(target)
                    if candidate in gt_img_ids:
                        pred_image_id = candidate

                if pred_image_id is None:
                    if len(missing_image_mappings) < 5:
                        missing_image_mappings.append({
                            "file_name": file_name,
                            "raw_image_id": target_image_id,
                            "batch_idx": batch_idx,
                        })
                    continue

                if smoke_checked < 5:
                    smoke_checked += 1
                    if pred_image_id not in gt_img_ids:
                        smoke_invalid.append({"pred_image_id": pred_image_id, "file_name": file_name})

                boxes = output.get("boxes")
                scores = output.get("scores")
                labels = output.get("labels")

                if boxes is None or scores is None or labels is None:
                    raise RuntimeError("Saída do modelo incompatível: esperado boxes, scores e labels.")

                boxes_cpu = boxes.detach().cpu()
                scores_cpu = scores.detach().cpu()
                labels_cpu = labels.detach().cpu()
                boxes_cpu = _rescale_boxes_to_original(
                    boxes_cpu,
                    target,
                    (int(image_tensor.shape[-2]), int(image_tensor.shape[-1])),
                    logging_logger,
                    tag=tag,
                )

                if labels_cpu.numel() > 0:
                    max_label_observed = max(max_label_observed, int(labels_cpu.max()))

                for box, score, label in zip(boxes_cpu, scores_cpu, labels_cpu):
                    raw_label = int(label)
                    category_id = _map_label_to_category(raw_label)
                    if category_id is None:
                        continue

                    prediction = {
                        "image_id": int(pred_image_id),
                        "category_id": int(category_id),
                        "bbox": _coco_box_from_xyxy(box),
                        "score": float(score),
                    }
                    predictions.append(prediction)
                    predicted_image_ids.add(int(pred_image_id))
                    predicted_category_ids.add(int(category_id))

    if not val_image_ids_set and fallback_target_image_ids:
        val_image_ids_set = set(fallback_target_image_ids)
        logging_logger.warning(
            "%s [AUDIT] val_image_ids_set recuperado dos targets após falha no dataset (total=%d)",
            tag,
            len(val_image_ids_set),
        )

    if smoke_invalid:
        raise AssertionError(f"{tag} Smoke test falhou: image_ids fora do GT: {smoke_invalid}")

    logging_logger.info("%s Máximo label observado: %d", tag, max_label_observed)
    if invalid_label_count > 0:
        logging_logger.warning("%s [AUDIT] Labels inválidos detectados: %d. Exemplos: %s", tag, invalid_label_count, invalid_label_examples)
    if missing_image_mappings:
        logging_logger.error("%s [AUDIT] Falha ao mapear image_id para GT: exemplos=%s", tag, missing_image_mappings)

    logging_logger.info(
        "%s [AUDIT] expected_val_images=%s processed_images=%d",
        tag,
        expected_val_images,
        processed_images,
    )
    if expected_val_images is not None and processed_images < expected_val_images:
        logging_logger.warning(
            "%s [AUDIT] processed_images (%d) menor que esperado (%d)",
            tag,
            processed_images,
            expected_val_images,
        )

    total_preds = len(predictions)
    unique_pred_img_ids = set(int(p["image_id"]) for p in predictions)
    unique_pred_cat_ids = set(int(p["category_id"]) for p in predictions)
    invalid_img_ids = sorted(list(unique_pred_img_ids - gt_img_ids))
    invalid_cat_ids = sorted(list(unique_pred_cat_ids - gt_cat_ids))
    num_invalid_img_ids = len(invalid_img_ids)
    num_invalid_cat_ids = len(invalid_cat_ids)
    cat_counter = Counter(int(p["category_id"]) for p in predictions)
    top_categories = cat_counter.most_common(10)
    logging_logger.info(
        "%s [VAL-METRICS][AUDIT] detecções=%d unique_img_ids=%d unique_cat_ids=%d",
        tag,
        total_preds,
        len(unique_pred_img_ids),
        len(unique_pred_cat_ids),
    )
    if unique_pred_img_ids:
        logging_logger.info(
            "%s [VAL-METRICS][AUDIT] img_id range=[%s, %s]",
            tag,
            min(unique_pred_img_ids),
            max(unique_pred_img_ids),
        )
    if not val_image_ids_set:
        val_image_ids_set = set(predicted_image_ids)
        logging_logger.warning(
            "%s [AUDIT] val_image_ids_set vazio; usando ids das predições (total=%d)",
            tag,
            len(val_image_ids_set),
        )
    missing_pred_ids = set(int(x) for x in val_image_ids_set) - unique_pred_img_ids
    logging_logger.info("%s [VAL-METRICS][AUDIT] missing_pred_ids (primeiros 20)=%s", tag, sorted(list(missing_pred_ids))[:20])
    if expected_val_images is not None and expected_val_images > 0 and len(unique_pred_img_ids) < expected_val_images * 0.9:
        logging_logger.warning(
            "%s [AUDIT] Cobertura de image_id baixa: %d/%d",
            tag,
            len(unique_pred_img_ids),
            expected_val_images,
        )
    logging_logger.info("%s [VAL-METRICS][AUDIT] top categorias (id, count)=%s", tag, top_categories)
    if invalid_img_ids:
        logging_logger.warning(
            "%s [VAL-METRICS][AUDIT] image_id inválidos detectados (exemplos=%s)", tag, invalid_img_ids[:5]
        )
    if invalid_cat_ids:
        logging_logger.warning(
            "%s [VAL-METRICS][AUDIT] category_id inválidos detectados (exemplos=%s)",
            tag,
            invalid_cat_ids[:5],
        )

    filtered_predictions = [p for p in predictions if int(p["image_id"]) in gt_img_ids and int(p["category_id"]) in gt_cat_ids]
    if len(filtered_predictions) != total_preds:
        logging_logger.info("%s [VAL-METRICS][AUDIT] Filtro aplicado: antes=%d depois=%d", tag, total_preds, len(filtered_predictions))

    predictions_path.write_text(json.dumps(filtered_predictions, indent=2), encoding="utf-8")

    valid_pred_img_ids = set(int(p["image_id"]) for p in filtered_predictions)
    valid_pred_cat_ids = set(int(p["category_id"]) for p in filtered_predictions)
    if not filtered_predictions:
        if not is_retinanet:
            raise RuntimeError(f"{tag} Nenhuma predição válida após auditoria. Abortando COCOeval.")

        logging_logger.warning(
            "%s No predictions to evaluate (likely undertrained in early epochs). Skipping COCOeval.",
            tag,
        )

        stats = [0.0] * 12
        metrics = {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "APs": 0.0,
            "APm": 0.0,
            "APl": 0.0,
            "AR1": 0.0,
            "AR10": 0.0,
            "AR100": 0.0,
            "ARs": 0.0,
            "ARm": 0.0,
            "ARl": 0.0,
        }

        gt_summary = {
            "num_images": len(gt_img_ids),
            "num_categories": len(gt_cat_ids),
            "min_image_id": min(gt_img_ids) if gt_img_ids else None,
            "max_image_id": max(gt_img_ids) if gt_img_ids else None,
        }
        pred_summary = {
            "num_detections": 0,
            "unique_image_ids": 0,
            "unique_category_ids": 0,
        }

        return {
            "coco_metrics": metrics,
            "coco_stats": stats,
            "predictions_coco_json": str(predictions_path),
            "per_class": {},
            "gt_summary": gt_summary,
            "pred_summary": pred_summary,
            "gt_annotations": str(val_ann),
            "num_predictions": 0,
            "reason": "no_predictions",
            "metrics_valid": False,
        }
    if num_invalid_img_ids > 0:
        logging_logger.info(
            "%s [VAL-METRICS][AUDIT] %d image_id(s) inválidos foram filtrados", tag, num_invalid_img_ids
        )
    if num_invalid_cat_ids > 0:
        logging_logger.info(
            "%s [VAL-METRICS][AUDIT] %d category_id(s) inválidos foram filtrados", tag, num_invalid_cat_ids
        )
    if not valid_pred_cat_ids <= gt_cat_ids:
        raise RuntimeError(f"{tag} category_id inválidos após filtro: {sorted(list(valid_pred_cat_ids - gt_cat_ids))}")

    logging_logger.info("%s Executando COCOeval...", tag)

    coco_dt = coco_gt.loadRes(str(predictions_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = [float(x) for x in coco_eval.stats.tolist()]
    metrics = {
        "AP": stats[0],
        "AP50": stats[1],
        "AP75": stats[2],
        "APs": stats[3],
        "APm": stats[4],
        "APl": stats[5],
        "AR1": stats[6],
        "AR10": stats[7],
        "AR100": stats[8],
        "ARs": stats[9],
        "ARm": stats[10],
        "ARl": stats[11],
    }

    precisions = coco_eval.eval.get("precision")
    per_class: dict[str, dict[str, float | str]] = {}
    if precisions is not None:
        for idx, cat_id in enumerate(coco_eval.params.catIds):
            cls_prec = precisions[:, :, idx, 0, 2]
            cls_prec = cls_prec[cls_prec > -1]
            ap = float(cls_prec.mean()) if cls_prec.size else float("nan")
            per_class[str(cat_id)] = {"name": coco_gt.cats.get(cat_id, {}).get("name", str(cat_id)), "AP": ap}

    logging_logger.info("%s AP=%.4f, AP50=%.4f, AP75=%.4f, AR100=%.4f", tag, metrics["AP"], metrics["AP50"], metrics["AP75"], metrics["AR100"])

    gt_summary = {
        "num_images": len(gt_img_ids),
        "num_categories": len(gt_cat_ids),
        "min_image_id": min(gt_img_ids) if gt_img_ids else None,
        "max_image_id": max(gt_img_ids) if gt_img_ids else None,
    }
    pred_summary = {
        "num_detections": len(filtered_predictions),
        "unique_image_ids": len(valid_pred_img_ids),
        "unique_category_ids": len(valid_pred_cat_ids),
    }

    return {
        "coco_metrics": metrics,
        "coco_stats": stats,
        "predictions_coco_json": str(predictions_path),
        "per_class": per_class,
        "gt_summary": gt_summary,
        "pred_summary": pred_summary,
        "gt_annotations": str(val_ann),
        "num_predictions": len(filtered_predictions),
    }


def run_post_training_validation(
    model_builder: Callable[[int], torch.nn.Module],
    dataset_dir: Path,
    train_ann: Path,
    val_ann: Path,
    weights_path: Path,
    config: TrainConfig,
    *,
    logger: Optional[Logger] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    run_tag: str = "torchvision",
) -> dict:
    log_dir = Path(config.log_dir).expanduser().resolve()
    safe_stdout = _safe_stream(f"{run_tag}_val_stdout", log_dir)
    logging_logger, log_path = _configure_logging(
        config.verbose,
        log_dir,
        logger,
        stream_override=safe_stdout,
        logger_name=f"{run_tag}_val",
        log_prefix=f"{run_tag}_val",
    )

    def _emit(message: str) -> None:
        if log_cb:
            log_cb(message)
        logging_logger.info(message)

    _emit(f"[{run_tag.upper()}][VAL-POST] Logger inicializado em {log_path}")

    device_str = resolve_device(config.device)
    device = torch.device(device_str)
    _emit(f"[{run_tag.upper()}][VAL-POST] Dispositivo: {device_str}")

    ensure_weights_size(weights_path, logger=_emit)

    expects_background = getattr(config, "num_classes", None) != getattr(config, "dataset_num_classes", None)
    val_mode_requested = getattr(config, "val_mode", "loss")
    if val_mode_requested not in {"loss", "metrics"}:
        logging_logger.warning("[VAL] val_mode desconhecido %s; forçando 'loss'", val_mode_requested)
        val_mode_requested = "loss"
    val_mode = val_mode_requested
    metrics_valid = True

    force_dataset_classes = None
    if run_tag == "faster_rcnn" and val_mode in {"loss", "metrics"}:
        force_dataset_classes, _ = _infer_dataset_classes_from_annotations(val_ann, "val", logging_logger)
        if force_dataset_classes is None:
            raise ValueError("[VAL] Não foi possível derivar dataset_num_classes do COCO JSON (categories).")
        expects_background = True

    val_loader, model_num_classes, dataset_num_classes, num_workers = _build_val_loader_and_classes(
        dataset_dir,
        train_ann,
        val_ann,
        config,
        logging_logger,
        expects_background=expects_background,
        force_dataset_num_classes=force_dataset_classes,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = log_dir / run_tag / "val_post" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    model = model_builder(model_num_classes)
    model.to(device)

    def _emit_warning(message: str) -> None:
        if log_cb:
            log_cb(message)
        logging_logger.warning(message)

    loaded = torch.load(weights_path.expanduser().resolve(), map_location="cpu")
    state_dict, checkpoint_format = _extract_state_dict(loaded)
    _emit(f"[{run_tag.upper()}][VAL-POST] Formato de checkpoint: {checkpoint_format}")
    (
        missing,
        unexpected,
        ckpt_num_classes,
        head_mismatch,
        ignored_keys,
        strict_load,
    ) = _load_frcnn_weights_with_head_guard(
        model,
        state_dict,
        model_num_classes=model_num_classes,
        emit_info=_emit,
        emit_warning=_emit_warning,
        logging_logger=logging_logger,
    )
    logging_logger.info(
        "[WEIGHTS] ckpt_num_classes=%s model_num_classes=%s strict_load=%s",
        ckpt_num_classes,
        model_num_classes,
        strict_load,
    )
    if head_mismatch and val_mode in {"loss", "metrics"} and isinstance(model, FasterRCNN):
        message = (
            f"Checkpoint incompatível: ckpt_num_classes={ckpt_num_classes}, esperado={model_num_classes}. "
            "Retreine para gerar pesos compatíveis."
        )
        logging_logger.error(message)
        raise RuntimeError(message)

    if head_mismatch:
        metrics_valid = False
        logging_logger.error(
            "[WEIGHTS] ckpt_num_classes=%s model_num_classes=%s -> head incompatível; métricas COCO inválidas.",
            ckpt_num_classes,
            model_num_classes,
        )
        if val_mode_requested == "metrics":
            warning_msg = (
                "Pesos carregados sem head compatível; métricas COCO não serão executadas. "
                "Recomenda-se retreinar/fine-tunar com dataset_num_classes correto."
            )
            _emit_warning(warning_msg)
            val_mode = "loss"
        _emit(f"[{run_tag.upper()}][VAL-POST] Head incompatível; ignorados: {ignored_keys}")

    if isinstance(model, FasterRCNN):
        _ensure_frcnn_head(model, model_num_classes, logging_logger)

    try:
        if val_mode == "metrics":
            results = run_val_coco_metrics(
                model,
                val_loader,
                device_str,
                val_ann,
                logging_logger,
                tag="[VAL-METRICS]",
                output_dir=out_dir,
            )
        else:
            results = run_val_loss_loop(
                model,
                val_loader,
                device_str,
                num_classes=model_num_classes,
                expects_background=retinanet_expects_background,
                label_offset=retinanet_label_offset,
                logging_logger=logging_logger,
            )
    except Exception as exc:
        error_payload = {
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat(),
            "val_annotations": str(val_ann),
        }
        try:
            COCO, _ = _ensure_pycocotools()
            coco_gt = COCO(str(val_ann))
            img_ids = list(int(x) for x in coco_gt.getImgIds())
            cat_ids = list(int(x) for x in coco_gt.getCatIds())
            error_payload["gt_summary"] = {
                "num_images": len(img_ids),
                "min_image_id": min(img_ids) if img_ids else None,
                "max_image_id": max(img_ids) if img_ids else None,
                "num_categories": len(cat_ids),
            }
        except Exception as inner_exc:  # pragma: no cover - tentativa auxiliar de debug
            error_payload["gt_summary_error"] = str(inner_exc)

        results_error_path = out_dir / "results_error.json"
        results_error_path.write_text(json.dumps(error_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        _emit_warning(f"[{run_tag.upper()}][VAL-POST] Falha; detalhes em {results_error_path}")
        raise

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
        "val_mode": val_mode,
        **results,
    }
    results_payload.update({"output_dir": str(out_dir)})
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _emit(f"[{run_tag.upper()}][VAL-POST] Resultado salvo em {results_path}")

    results_payload.update({"results_path": str(results_path)})

    return results_payload


def train_torchvision_detector(
    model: torch.nn.Module,
    dataset_dir: Path,
    train_ann: Path,
    val_ann: Path,
    weights_out: Path,
    config: TrainConfig,
    logger: Optional[Logger] = None,
    val_ratio: Optional[float] = None,
    train_dataset=None,
    val_dataset=None,
    checkpoint_dir: Optional[Path] = None,
) -> Metrics:
    log_dir = Path(config.log_dir).expanduser().resolve()
    safe_stdout = _safe_stream("stdout", log_dir)
    safe_stderr = _safe_stream("stderr", log_dir)
    algorithm_name = None
    model_label = getattr(model, "__class__", type("Obj", (), {})).__name__
    is_retinanet = isinstance(model, RetinaNet)
    if isinstance(model, FasterRCNN):
        algorithm_name = "Faster R-CNN"
    elif is_retinanet:
        algorithm_name = "RetinaNet"
    else:
        algorithm_name = "SSD"

    ssd_debug = bool(int(os.getenv("SSD_DEBUG", "0"))) if algorithm_name == "SSD" else False
    logger_name = "train.ssd" if algorithm_name == "SSD" else "ssd_train"
    log_prefix = "train_ssd" if algorithm_name == "SSD" else "ssd_train"

    logging_logger, log_path = _configure_logging(
        config.verbose or ssd_debug, log_dir, logger, stream_override=safe_stdout, logger_name=logger_name, log_prefix=log_prefix
    )
    try:
        faulthandler.enable(file=safe_stderr)
    except Exception as exc:  # pragma: no cover - compatibilidade com ambientes sem fileno
        logging_logger.warning("Não foi possível habilitar faulthandler no stream seguro: %s", exc)
    try:
        signal.signal(signal.SIGTERM, lambda _sig, _frame: sys.exit(1))
    except Exception:  # pragma: no cover - compatibilidade com ambientes sem suporte
        logging_logger.warning("Não foi possível registrar handler de SIGTERM neste ambiente.")

    logging_logger.info("Iniciando setup de treinamento do %s...", algorithm_name)
    device_str = resolve_device(config.device)
    device = torch.device(device_str)
    seed_everything(config.seed)
    logging_logger.info("Dispositivo: %s | torch=%s | cuda_available=%s", device_str, torch.__version__, torch.cuda.is_available())
    logging_logger.info("Configuração: %s", config)
    epochs_to_run = config.epochs if config.max_epochs is None else min(config.epochs, config.max_epochs)
    if config.max_epochs is not None:
        logging_logger.info("[TRAIN] max_epochs ativo=%s -> epochs_to_run=%s", config.max_epochs, epochs_to_run)

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else Path(weights_out).expanduser().resolve().parent
    ckpt_dir = ckpt_dir.expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logging_logger.info("Checkpoints serão salvos em: %s", ckpt_dir)

    watchdog_stop: Optional[threading.Event] = None
    watchdog_pause: Optional[threading.Event] = None
    watchdog_thread: Optional[threading.Thread] = None
    last_progress = [time.monotonic()]
    epoch = 0
    epochs_completed = 0
    optimizer: Optional[torch.optim.Optimizer] = None
    meta: dict[str, Any] = {}
    try:
        val_ratio_to_use = config.val_ratio if val_ratio is None else val_ratio

        if train_dataset is None or val_dataset is None:
            transform = _build_detection_transforms(config.imgsz, logging_logger)
            logging_logger.info("[SETUP] Construindo datasets COCO a partir de %s", dataset_dir)
            train_ds_full = CocoDetectionDataset(dataset_dir / "images" / "train", train_ann, transforms=transform)
            val_ds_full = CocoDetectionDataset(dataset_dir / "images" / "val", val_ann, transforms=transform)

            total_before_split = len(train_ds_full)
            split_applied = bool(val_ratio_to_use and val_ratio_to_use > 0)

            # Dividir train em train/val adicionais se desejado
            if split_applied:
                logging_logger.info(
                    "[SETUP] Aplicando split adicional train/val com val_ratio=%.3f seed=%s", val_ratio_to_use, config.seed
                )
                train_ds_full, extra_val = _split_dataset(train_ds_full, val_ratio_to_use, config.seed, logging_logger)
                val_ds_full = extra_val
        else:
            logging_logger.info("[SETUP] Usando datasets pré-construídos (train/val).")
            train_ds_full = train_dataset
            val_ds_full = val_dataset
            total_before_split = len(train_ds_full)
            split_applied = False

        train_after_split = len(train_ds_full)
        val_after_split = len(val_ds_full) if split_applied else 0
        total_after_split = train_after_split + val_after_split

        if split_applied:
            logging_logger.info("[DATA] Dataset base (antes do split): total=%d", total_before_split)
            logging_logger.info(
                "[DATA] Após split: train=%d | val=%d | total=%d",
                train_after_split,
                val_after_split,
                total_after_split,
            )
        else:
            logging_logger.info(
                "[DATA] Sem split adicional: train=total=%d | val=%d",
                train_after_split,
                val_after_split,
            )

        def _sample_labels(ds: Any, limit: int = 32) -> list[int]:
            labels: list[int] = []
            for idx in range(min(len(ds), limit)):
                try:
                    _, tgt = ds[idx]
                except Exception:
                    continue
                tensor_labels = tgt.get("labels") if isinstance(tgt, dict) else None
                if torch.is_tensor(tensor_labels):
                    labels.extend(int(x) for x in tensor_labels.detach().cpu().tolist())
            return labels

        if hasattr(train_ds_full, "cat_id_to_label"):
            logging_logger.info(
                summarize_class_mapping(
                    dataset_name=getattr(train_ds_full, "dataset_name", None),
                    k=getattr(train_ds_full, "num_classes", 0),
                    class_names=getattr(train_ds_full, "class_names", []),
                    categories=getattr(train_ds_full, "annotations", {}).get("categories", []),
                    cat_id_to_label=getattr(train_ds_full, "cat_id_to_label", {}),
                    label_to_cat_id=getattr(train_ds_full, "label_to_cat_id", {}),
                    observed_labels=_sample_labels(train_ds_full),
                )
            )

        describe_dataloader(train_ds_full, logging_logger.info, label="train subset")
        logging_logger.info("[SETUP] Tamanho train=%d | val=%d", len(train_ds_full), len(val_ds_full))

        configured_dataset_classes = getattr(config, "dataset_num_classes", None)
        configured_model_num_classes = getattr(config, "num_classes", None)
        use_background = not is_retinanet
        retinanet_expects_background = False
        retinanet_label_stats: dict[str, Any] = {}
        allow_zero_label = is_retinanet
        retinanet_label_offset = 0

        train_ann_classes, train_cat_ids = _infer_dataset_classes_from_annotations(train_ann, "train", logging_logger)
        val_ann_classes, val_cat_ids = _infer_dataset_classes_from_annotations(val_ann, "val", logging_logger)
        ann_dataset_classes = train_ann_classes or val_ann_classes

        inferred_train_classes = _normalize_dataset_class_count(_infer_num_classes_from_dataset(train_ds_full))
        inferred_val_classes = _normalize_dataset_class_count(_infer_num_classes_from_dataset(val_ds_full))

        dataset_num_classes: Optional[int] = None
        if ann_dataset_classes is not None:
            dataset_num_classes = ann_dataset_classes
        elif configured_dataset_classes is not None:
            dataset_num_classes = configured_dataset_classes
        elif configured_model_num_classes is not None and configured_model_num_classes > 0:
            dataset_num_classes = (
                max(1, configured_model_num_classes - 1) if use_background else configured_model_num_classes
            )
        else:
            dataset_num_classes = inferred_train_classes

        looks_like_visdrone = _is_visdrone_dataset(dataset_dir, train_ds_full) or _is_visdrone_dataset(
            dataset_dir, val_ds_full
        )

        if dataset_num_classes is None:
            dataset_num_classes = inferred_val_classes

        if ann_dataset_classes is not None and dataset_num_classes is not None and dataset_num_classes != ann_dataset_classes:
            logging_logger.warning(
                "[AUDIT] num_classes configurado/inferido (%s) difere do COCO (json=%s); usando configurado/inferido.",
                dataset_num_classes,
                ann_dataset_classes,
            )

        if dataset_num_classes is None:
            raise ValueError("Não foi possível inferir num_classes a partir do dataset; verifique as categorias.")

        if inferred_val_classes is not None and inferred_val_classes != dataset_num_classes:
            logging_logger.warning(
                "[AUDIT] num_classes do val (%s) difere do train (%s); usando o do train.",
                inferred_val_classes,
                dataset_num_classes,
            )

        class_names = getattr(train_ds_full, "class_names", [])
        cat_id_to_label = getattr(train_ds_full, "cat_id_to_label", {})
        label_to_cat_id = getattr(train_ds_full, "label_to_cat_id", {})
        meta = {
            "num_classes": dataset_num_classes,
            "class_names": class_names,
            "cat_id_to_label": cat_id_to_label,
            "label_to_cat_id": label_to_cat_id,
            "dataset_name": getattr(train_ds_full, "dataset_name", None),
            "git_commit": _current_git_commit(),
        }

        if is_retinanet:
            retinanet_label_stats = _audit_retinanet_label_distribution(train_ds_full, "train", logging_logger)
            val_label_stats = _audit_retinanet_label_distribution(val_ds_full, "val", logging_logger)
            combined_min = min(
                x
                for x in (retinanet_label_stats.get("min"), val_label_stats.get("min"))
                if x is not None
            ) if any(x is not None for x in (retinanet_label_stats.get("min"), val_label_stats.get("min"))) else None
            combined_max = max(
                x
                for x in (retinanet_label_stats.get("max"), val_label_stats.get("max"))
                if x is not None
            ) if any(x is not None for x in (retinanet_label_stats.get("max"), val_label_stats.get("max"))) else None
            if combined_min is not None and combined_min < 0:
                raise ValueError("[RETINANET][CLASSES] Labels negativos detectados; esperados labels 0-based sem background.")
            retinanet_inferred_from_labels = (combined_max + 1) if combined_max is not None else None
            logging_logger.info(
                "[RETINANET][CLASSES] dataset_num_classes=%s inferred_from_labels=%s",
                dataset_num_classes,
                retinanet_inferred_from_labels,
            )

        if is_retinanet:
            use_background = False
            retinanet_expects_background = False
        expected_model_classes = dataset_num_classes + 1 if use_background and dataset_num_classes > 0 else dataset_num_classes
        model_num_classes = expected_model_classes
        if configured_model_num_classes is not None and configured_model_num_classes > 0:
            model_num_classes = configured_model_num_classes
            if configured_model_num_classes != expected_model_classes:
                logging_logger.warning(
                    "[DATASET] num_classes configurado (%s) difere do esperado (%s); respeitando o configurado.",
                    configured_model_num_classes,
                    expected_model_classes,
                )

        logging_logger.info(
            "[MODEL] dataset_num_classes=%d model_num_classes=%d (%s)",
            dataset_num_classes,
            model_num_classes,
            "incluindo background" if use_background else "foreground only",
        )

        if is_retinanet:
            cls_head = getattr(getattr(getattr(model, "head", None), "classification_head", None), "cls_logits", None)
            head_shape = tuple(cls_head.weight.shape) if getattr(cls_head, "weight", None) is not None else None
            logging_logger.info(
                "[RETINANET][CLASSES] model_num_classes=%d (foreground only) head_cls_shape=%s",
                model_num_classes,
                head_shape,
            )

        if looks_like_visdrone:
            logging_logger.info(
                "[DATASET] VisDrone multi-class: dataset_classes=%d -> model_num_classes=%d (%s)",
                dataset_num_classes,
                model_num_classes,
                "incluindo background" if use_background else "foreground only",
            )

        if hasattr(train_ds_full, "transforms") and hasattr(val_ds_full, "transforms"):
            if type(getattr(train_ds_full, "transforms")) != type(getattr(val_ds_full, "transforms")):
                logging_logger.warning(
                    "[TRANSFORM] Transforms de train e val divergem em tipo (%s vs %s)",
                    type(getattr(train_ds_full, "transforms")),
                    type(getattr(val_ds_full, "transforms")),
                )

        if config.audit_datasets:
            _audit_dataset(
                train_ds_full,
                "train",
                num_classes=model_num_classes,
                logger=logging_logger,
                allow_zero_label=allow_zero_label,
            )
            _audit_dataset(
                val_ds_full,
                "val",
                num_classes=model_num_classes,
                logger=logging_logger,
                allow_zero_label=allow_zero_label,
            )

        collate_fn = ssd_collate_with_diagnostics if algorithm_name == "SSD" else coco_collate
        dataloader_kwargs = dict(
            batch_size=config.batch_size,
            collate_fn=collate_fn,
            drop_last=config.drop_last,
        )
        train_workers = config.num_workers
        pin_memory = config.pin_memory
        persistent_workers = config.persistent_workers
        prefetch_factor = config.prefetch_factor
        if config.debug_dataloader:
            logging_logger.warning("[DATALOADER] Modo debug ativado -> num_workers=0, persistent_workers=False, pin_memory=False")
            train_workers = 0
            pin_memory = False
            persistent_workers = False
            prefetch_factor = None
            dataloader_kwargs["drop_last"] = False
        if train_workers > 0:
            dataloader_kwargs.update(
                num_workers=train_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )
            if prefetch_factor is not None:
                dataloader_kwargs["prefetch_factor"] = prefetch_factor
        else:
            dataloader_kwargs["num_workers"] = 0
        logging_logger.info(
            "[DATALOADER] batch_size=%s num_workers=%s pin_memory=%s persistent_workers=%s prefetch_factor=%s drop_last=%s",
            dataloader_kwargs.get("batch_size"),
            dataloader_kwargs.get("num_workers"),
            dataloader_kwargs.get("pin_memory"),
            dataloader_kwargs.get("persistent_workers"),
            dataloader_kwargs.get("prefetch_factor"),
            dataloader_kwargs.get("drop_last"),
        )

        logging_logger.info("[SETUP] Construindo DataLoaders...")
        train_loader = DataLoader(train_ds_full, shuffle=True, **dataloader_kwargs)
        val_loader = DataLoader(val_ds_full, shuffle=False, **dataloader_kwargs)
        logging_logger.info("DataLoader construído. num_batches=%d", len(train_loader))

        if is_retinanet:
            run_detection_sanity_check(
                model,
                val_loader,
                device,
                logging_logger,
                num_classes=model_num_classes,
            )

        if algorithm_name == "SSD" and ssd_debug:
            _run_ssd_probe(train_loader, logging_logger, limit=50)

        stopper: Optional[TrainLossEMAStopper] = None
        best_by_trainloss_path = ckpt_dir / "best_by_trainloss.pth"
        if config.early_stop_enabled:
            try:
                stopper = TrainLossEMAStopper(
                    patience=config.early_stop_patience,
                    min_delta=config.early_stop_min_delta,
                    min_epochs=config.early_stop_min_epochs,
                    ema_alpha=config.early_stop_ema_alpha,
                )
                logging_logger.info(
                    "[EARLY] Early stopping ativado: patience=%d min_delta=%.4f min_epochs=%d ema_alpha=%.3f",
                    config.early_stop_patience,
                    config.early_stop_min_delta,
                    config.early_stop_min_epochs,
                    config.early_stop_ema_alpha,
                )
            except Exception:
                logging_logger.exception("[EARLY] Falha ao inicializar early stopping; desabilitando por segurança.")
                stopper = None

        if config.smoke_test_val_loss:
            _run_smoke_test_val_loss(
                model,
                val_loader,
                device,
                logging_logger,
                max_images=config.smoke_test_samples,
                num_classes=model_num_classes,
            )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging_logger.info("[MODEL] Modelo preparado. params=%d | treináveis=%d", total_params, trainable_params)
        logging_logger.info("[SETUP] Movendo modelo para %s e preparando otimizador.", device_str)
        model.to(device_str)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)
        logging_logger.info("Starting training...")

        num_classes = model_num_classes
        if hasattr(model, "roi_heads"):
            _ensure_frcnn_head(model, num_classes, logging_logger)

        last_progress = [time.monotonic()]
        watchdog_stop = threading.Event()
        watchdog_pause = threading.Event()
        watchdog_thread = _start_watchdog(
            log_path, last_progress, watchdog_stop, logging_logger, safe_stderr=safe_stderr, pause_event=watchdog_pause
        )

        last_loss = None
        logged_loss_structure = False
        heartbeat_seconds = 10
        log_every_batches = max(1, config.log_every)
        stop_training_flag = False
        last_ok_sample: list[str] = []

        epochs_completed = 0
        for epoch in range(1, epochs_to_run + 1):
            model.train()
            running_loss = 0.0
            epoch_wall_start = time.perf_counter()
            last_heartbeat = time.perf_counter()
            total_batches = len(train_loader)
            logging_logger.info("Epoch %d/%d | num_batches=%d", epoch, epochs_to_run, total_batches)
            progress = None
            if tqdm:
                try:
                    progress = tqdm(
                        total=total_batches,
                        desc=f"Epoch {epoch}/{epochs_to_run}",
                        leave=False,
                        file=safe_stderr,
                        dynamic_ncols=False,
                    )
                except Exception as exc:  # pragma: no cover - fallback automático
                    logging_logger.warning("tqdm não pôde iniciar; usando logs periódicos. erro=%s", exc)
            for step, (images, targets) in enumerate(train_loader, start=1):
                batch_start = time.perf_counter()
                images = [img.to(device_str) for img in images]
                targets = [
                    {k: (v.to(device_str) if torch.is_tensor(v) else v) for k, v in t.items()}
                    for t in targets
                ]

                if algorithm_name == "SSD":
                    logging_logger.debug(
                        "[STAGE=forward_prep] epoch=%d step=%d batch_size=%d", epoch, step, len(images)
                    )
                    for img_idx, img in enumerate(images):
                        if img.device.type != torch.device(device_str).type:
                            raise RuntimeError(
                                f"[STAGE=forward_prep] imagem no device incorreto (esperado {device_str}) no batch {step}"
                            )
                        if not torch.is_floating_point(img):
                            raise TypeError(
                                f"[STAGE=forward_prep] dtype inesperado para imagem {img_idx} no batch {step}: {img.dtype}"
                            )
                        if not torch.isfinite(img).all():
                            raise ValueError(
                                f"[STAGE=forward_prep] valores não finitos detectados na imagem {img_idx} batch {step}"
                            )

                image_sizes = [(int(img.shape[-2]), int(img.shape[-1])) for img in images]
                _validate_targets_batch(
                    targets,
                    num_classes=num_classes,
                    image_sizes=image_sizes,
                    allow_zero_label=retinanet_expects_background,
                    logger=logging_logger,
                )

                if algorithm_name == "SSD":
                    last_ok_sample = [t.get("img_path", "<desconhecido>") for t in targets]
                    logging_logger.debug(
                        "[LAST_OK_SAMPLE] epoch=%d step=%d paths=%s", epoch, step, last_ok_sample
                    )

                _maybe_remap_retinanet_targets(
                    model,
                    targets,
                    num_classes=num_classes,
                    expects_background=retinanet_expects_background,
                    label_offset=retinanet_label_offset,
                    logger=logging_logger,
                )
                loss_out = model(images, targets)
                if not logged_loss_structure:
                    try:
                        logging_logger.info("Loss return structure: %s", _describe_structure(loss_out))
                    except Exception:
                        logging_logger.exception("Falha ao descrever estrutura de loss_out.")
                    finally:
                        logged_loss_structure = True
                try:
                    losses, loss_items = _normalize_losses(loss_out, device)
                except Exception:
                    logging_logger.exception("Falha ao normalizar losses; usando 0.0 para continuar.")
                    losses = torch.tensor(0.0, device=device)
                    loss_items = {}
                if loss_items:
                    logging_logger.debug(f"Loss components: {loss_items}")

                if not torch.isfinite(losses):
                    image_ids, file_paths = _collect_batch_identifiers(targets, train_loader.dataset)
                    target_dump = _summarize_targets_for_logging(targets)
                    logging_logger.error(
                        "Loss não finito detectado. epoch=%s step=%s paths=%s",
                        epoch,
                        step,
                        file_paths if file_paths else "<indisponível>",
                    )
                    bad_batch = {
                        "epoch": epoch,
                        "step": step,
                        "image_ids": image_ids,
                        "paths": file_paths,
                        "targets": target_dump,
                        "loss_items": loss_items,
                    }
                    if algorithm_name == "SSD":
                        bad_batch["last_ok_sample"] = last_ok_sample
                        _dump_ssd_debug_snapshot(ckpt_dir / "ssd_debug", bad_batch, logging_logger)
                    try:
                        bad_path_json = ckpt_dir / "bad_batch.json"
                        bad_path_txt = ckpt_dir / "bad_batch.txt"
                        bad_path_json.write_text(json.dumps(bad_batch, indent=2, ensure_ascii=False), encoding="utf-8")
                        lines = [
                            f"epoch={epoch}",
                            f"step={step}",
                            f"paths={file_paths if file_paths else '<indisponível>'}",
                        ]
                        for entry in target_dump:
                            lines.append("")
                            lines.append(f"img_path: {entry.get('img_path', '<desconhecido>')}")
                            lines.append(f"boxes: {entry.get('boxes', '<sem boxes>')}")
                            lines.append(f"labels: {entry.get('labels', '<sem labels>')}")
                        bad_path_txt.write_text("\n".join(lines), encoding="utf-8")
                        logging_logger.error("Batch problemático salvo em %s e %s", bad_path_json, bad_path_txt)
                    except Exception:
                        logging_logger.exception("Falha ao salvar informações do bad batch")
                    raise RuntimeError("Non-finite loss detected")

                optimizer.zero_grad()
                losses.backward()
                clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                running_loss += losses.item()
                last_progress[0] = time.monotonic()
                elapsed_batch = time.perf_counter() - batch_start
                avg_loss = running_loss / step
                lr = lr_scheduler.get_last_lr()[0]
                if progress:
                    try:
                        progress.set_postfix(loss=losses.item(), avg_loss=avg_loss, lr=f"{lr:.2e}", batch_time=f"{elapsed_batch:.3f}s")
                        progress.update(1)
                        if step % log_every_batches == 0 or time.perf_counter() - last_heartbeat >= heartbeat_seconds:
                            logging_logger.info(
                                "[heartbeat] epoch=%d step=%d/%d loss=%.4f avg_loss=%.4f lr=%.6f batch_time=%.3fs",
                                epoch,
                                step,
                                total_batches,
                                losses.item(),
                                avg_loss,
                                lr,
                                elapsed_batch,
                            )
                            last_heartbeat = time.perf_counter()
                    except Exception as exc:  # pragma: no cover - fallback automático
                        logging_logger.warning("tqdm falhou durante a atualização; revertendo para logs periódicos. erro=%s", exc)
                        try:
                            progress.close()
                        finally:
                            progress = None
                else:
                    now = time.perf_counter()
                    if step % log_every_batches == 0 or (now - last_heartbeat) >= heartbeat_seconds:
                        eta_batches = total_batches - step
                        eta = eta_batches * elapsed_batch
                        logging_logger.info(
                            "[heartbeat] epoch=%d step=%d/%d loss=%.4f avg_loss=%.4f lr=%.6f elapsed=%.2fs ETA=%.2fs",
                            epoch,
                            step,
                            total_batches,
                            losses.item(),
                            avg_loss,
                            lr,
                            now - epoch_wall_start,
                            eta,
                        )
                        last_heartbeat = now

            lr_scheduler.step()
            avg_loss = running_loss / max(1, len(train_loader))
            last_loss = avg_loss
            if progress:
                progress.close()
            logging_logger.info(
                "[TRAIN] Época %d/%d concluída | lr=%.6f | loss=%.4f | tempo=%.2fs",
                epoch,
                epochs_to_run,
                lr_scheduler.get_last_lr()[0],
                avg_loss,
                time.perf_counter() - epoch_wall_start,
            )

            val_mode = getattr(config, "val_mode", "loss")
            if val_mode not in {"loss", "metrics"}:
                logging_logger.warning("[VAL] val_mode desconhecido %s; forçando 'loss'", val_mode)
                val_mode = "loss"

            original_mode = model.training
            model.eval()
            if val_mode == "metrics":
                metrics_dir = ckpt_dir / "val_metrics" / f"epoch_{epoch:03d}"
                with _watchdog_paused(last_progress, watchdog_pause):
                    run_val_coco_metrics(
                        model,
                        val_loader,
                        device_str,
                        val_ann,
                        logging_logger,
                        tag="[VAL-METRICS]",
                        output_dir=metrics_dir,
                    )
            else:
                val_loss_sum = 0.0
                val_component_sum: dict[str, float] = {}
                total_images = 0
                total_batches = 0
                return_type_logged = False

                with _watchdog_paused(last_progress, watchdog_pause):
                    with torch.no_grad():
                        for vstep, (images, targets) in enumerate(val_loader, start=1):
                            images = [img.to(device_str) for img in images]
                            targets = [
                                {k: (v.to(device_str) if torch.is_tensor(v) else v) for k, v in t.items()}
                                for t in targets
                            ]
                            image_sizes = [(int(img.shape[-2]), int(img.shape[-1])) for img in images]
                            _validate_targets_batch(
                                targets,
                                num_classes=num_classes,
                                image_sizes=image_sizes,
                                allow_zero_label=retinanet_expects_background,
                                logger=logging_logger,
                            )

                            model.train()
                            _maybe_remap_retinanet_targets(
                                model,
                                targets,
                                num_classes=num_classes,
                                expects_background=retinanet_expects_background,
                                label_offset=retinanet_label_offset,
                                logger=logging_logger,
                            )
                            loss_out = model(images, targets)
                            model.eval()

                            if not return_type_logged:
                                keys = list(loss_out.keys()) if isinstance(loss_out, dict) else None
                                logging_logger.info(
                                    "[VAL] Return type detected: %s keys=%s", type(loss_out).__name__, keys
                                )
                                return_type_logged = True

                            if not isinstance(loss_out, dict) or not all(
                                torch.is_tensor(v) for v in loss_out.values()
                            ):
                                raise RuntimeError(
                                    "VAL expected loss dict, got predictions. Ensure calling model(images, targets) with model in "
                                    "train() under no_grad()."
                                )

                            batch_loss_tensor = sum(loss_out.values())

                            if not torch.isfinite(batch_loss_tensor):
                                image_ids, file_paths = _collect_batch_identifiers(targets, val_loader.dataset)
                                target_dump = _summarize_targets_for_logging(targets)
                                logging_logger.error(
                                    "Loss não finito detectado no val. epoch=%s step=%s paths=%s",
                                    epoch,
                                    vstep,
                                    file_paths if file_paths else "<indisponível>",
                                )
                                bad_batch = {
                                    "epoch": epoch,
                                    "step": vstep,
                                    "phase": "val",
                                    "image_ids": image_ids,
                                    "paths": file_paths,
                                    "targets": target_dump,
                                }
                                try:
                                    bad_path_json = ckpt_dir / "bad_batch.json"
                                    bad_path_txt = ckpt_dir / "bad_batch.txt"
                                    bad_path_json.write_text(json.dumps(bad_batch, indent=2, ensure_ascii=False), encoding="utf-8")
                                    lines = [
                                        f"epoch={epoch}",
                                        f"step={vstep}",
                                        f"paths={file_paths if file_paths else '<indisponível>'}",
                                    ]
                                    for entry in target_dump:
                                        lines.append("")
                                        lines.append(f"img_path: {entry.get('img_path', '<desconhecido>')}")
                                        lines.append(f"boxes: {entry.get('boxes', '<sem boxes>')}")
                                        lines.append(f"labels: {entry.get('labels', '<sem labels>')}")
                                    bad_path_txt.write_text("\n".join(lines), encoding="utf-8")
                                    logging_logger.error(
                                        "Batch problemático de validação salvo em %s e %s", bad_path_json, bad_path_txt
                                    )
                                except Exception:
                                    logging_logger.exception("Falha ao salvar informações do bad batch de validação")
                                raise RuntimeError("Non-finite loss detected")

                            total_batches += 1
                            total_images += len(images)
                            batch_loss_value = float(batch_loss_tensor.detach().cpu())
                            val_loss_sum += batch_loss_value

                            for key, value in loss_out.items():
                                val_component_sum[key] = val_component_sum.get(key, 0.0) + float(value.detach().cpu())

            if original_mode:
                model.train()
            else:
                model.eval()

            if val_mode == "metrics":
                val_loss_mean_per_batch = 0.0
                val_loss_mean_per_image = 0.0
                breakdown_mean: dict[str, float] = {}
                logging_logger.info("[VAL] Época %d | métricas COCO calculadas.", epoch)
            else:
                val_loss_mean_per_batch = val_loss_sum / max(1, total_batches)
                val_loss_mean_per_image = val_loss_sum / max(1, total_images)
                breakdown_mean = {k: v / max(1, total_batches) for k, v in val_component_sum.items()}
                logging_logger.info(
                    f"[VAL] Época {epoch} | loss/batch={val_loss_mean_per_batch:.4f} | loss/img={val_loss_mean_per_image:.4f}"
                )
                if breakdown_mean:
                    logging_logger.info("[VAL] Breakdown médio por batch: %s", breakdown_mean)

            if stopper:
                if not math.isfinite(avg_loss):
                    logging_logger.warning(
                        "[EARLY] avg_train_loss não finito (%.4f); desativando early stopping para segurança.", avg_loss
                    )
                    stopper = None
                else:
                    try:
                        should_stop, ema_loss, best_ema, bad_epochs = stopper.update(epoch, avg_loss)
                        logging_logger.info(
                            "[EARLY] epoch=%d avg_loss=%.4f ema_loss=%.4f best_ema=%.4f bad_epochs=%d/%d",
                            epoch,
                            avg_loss,
                            ema_loss,
                            best_ema if best_ema is not None else float("nan"),
                            bad_epochs,
                            stopper.patience,
                        )
                        if best_ema is not None and best_ema == ema_loss:
                            with _watchdog_paused(last_progress, watchdog_pause):
                                atomic_torch_save({"model": model.state_dict(), "meta": meta}, best_by_trainloss_path)
                            logging_logger.info("[EARLY] Novo melhor ema_loss; checkpoint salvo em %s", best_by_trainloss_path)
                        if should_stop:
                            logging_logger.info(
                                "[EARLY] Critério de paciência alcançado após min_epochs=%d; encerrando treinamento.",
                                stopper.min_epochs,
                            )
                            stop_training_flag = True
                    except Exception:
                        logging_logger.exception("[EARLY] Falha ao processar early stopping; desabilitando.")
                        stopper = None

            with _watchdog_paused(last_progress, watchdog_pause):
                epoch_ckpt = ckpt_dir / f"epoch_{epoch:03d}.pth"
                atomic_torch_save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict() if optimizer else None,
                        "loss": float(avg_loss),
                        "meta": meta,
                    },
                    epoch_ckpt,
                )
                logging_logger.info("Checkpoint da época %d salvo em %s", epoch, epoch_ckpt)

            epochs_completed = epoch
            if stop_training_flag:
                break

        watchdog_stop.set()
        watchdog_thread.join(timeout=1)

        weights_out = weights_out.expanduser().resolve()
        out = weights_out
        if out.suffix.lower() not in (".pth", ".pt"):
            out.mkdir(parents=True, exist_ok=True)
            out = out / "last.pth"
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
        out = out.with_suffix(".pth")

        with _watchdog_paused(last_progress, watchdog_pause):
            logging_logger.info("Salvando pesos em %s", out)
            atomic_torch_save({"model": model.state_dict(), "meta": meta}, out)
            ensure_weights_size(out, logger=logging_logger.info)
            weights_out = out
            logging_logger.info("Treinamento finalizado com sucesso.")

        return Metrics(
            precision=0.0,
            recall=0.0,
            map50=0.0,
            map50_95=0.0,
            loss_final=last_loss,
            epochs=epochs_completed or epochs_to_run,
            train_images=len(train_ds_full),
            device=device_str,
            weights_path=weights_out,
            map_computed=False,
        )
    except Exception:
        logging_logger.exception("Exceção não tratada durante o treinamento.")
        raise
    finally:
        try:
            with _watchdog_paused(last_progress, watchdog_pause):
                final_epoch = epochs_completed if epochs_completed else 0
                last_ckpt = ckpt_dir / "last.pth"
                atomic_torch_save(
                    {
                        "epoch": final_epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict() if optimizer else None,
                        "meta": meta,
                    },
                    last_ckpt,
                )
                logging_logger.info("Checkpoint final salvo em %s", last_ckpt)
        except Exception:
            logging_logger.exception("Falha ao salvar checkpoint final.")
        if watchdog_stop:
            watchdog_stop.set()
        if watchdog_thread:
            watchdog_thread.join(timeout=1)

# Bloco de teste rápido (manual) para ambientes com sys.stderr sem fileno:
# -------------------------------------------------------------------------
# class _NoFileno:
#     def write(self, msg): ...
#     def flush(self): ...
# sys.stderr = _NoFileno()  # simula PromptLogForwarder sem fileno
# cfg = TrainConfig(epochs=1, log_dir=Path("logs/test_fileno"))
# train_torchvision_detector(model, dataset_dir, train_ann, val_ann, weights_out, cfg)
# # Esperado: treino inicia, logs no console ou em logs/test_fileno/fallback_stderr.log.
