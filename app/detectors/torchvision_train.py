from __future__ import annotations

import faulthandler
import importlib.util
import json
import logging
import signal
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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
    seed_everything,
)
from app.metrics import Metrics

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback silencioso se tqdm não estiver disponível
    tqdm = None


VISDRONE_MULTICLASS_DATASET_CLASSES = 11


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
    if hasattr(dataset, "annotations") and isinstance(getattr(dataset, "annotations"), dict):
        categories = dataset.annotations.get("categories", [])
        if isinstance(categories, list):
            candidates.append(len(categories) + 1)
    for cand in candidates:
        if isinstance(cand, int) and cand > 0:
            return cand
    return None


def _normalize_dataset_class_count(raw_num_classes: Optional[int]) -> Optional[int]:
    if raw_num_classes is None:
        return None
    if isinstance(raw_num_classes, int) and raw_num_classes > 0:
        return raw_num_classes - 1 if raw_num_classes > 1 else raw_num_classes
    return None


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


def _validate_targets_batch(
    targets: list[dict[str, torch.Tensor]],
    *,
    num_classes: Optional[int],
    image_sizes: Iterable[tuple[int, int]],
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

        if num_classes is not None and num_classes > 0:
            if (labels < 1).any():
                raise ValueError(
                    f"Labels fora do intervalo no índice {idx}{path_label}: mínimo {int(labels.min())} (esperado >=1)"
                )
            if (labels >= num_classes).any():
                raise ValueError(
                    f"Labels fora do intervalo no índice {idx}{path_label}: máximo {int(labels.max())} >= num_classes ({num_classes})"
                )
        else:
            if (labels < 1).any():
                raise ValueError(f"Labels devem ser >=1 quando num_classes é desconhecido (índice {idx}{path_label})")

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
    dataset: Any, phase: str, *, num_classes: Optional[int], logger: logging.Logger, limit: Optional[int] = None
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
                _validate_targets_batch([target], num_classes=num_classes, image_sizes=[(h, w)], logger=logger)
            except Exception as exc:
                raise ValueError(f"[AUDIT] Falha no {phase} idx={idx} img={target.get('img_path', '<desconhecido>')}: {exc}")
            inspected += 1
    logger.info("[AUDIT] Auditoria concluída para %s: %d/%d amostras verificadas", phase, inspected, total)


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
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [
                {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
                for t in targets
            ]
            image_sizes = [(int(img.shape[-2]), int(img.shape[-1])) for img in images]
            _validate_targets_batch(targets, num_classes=num_classes, image_sizes=image_sizes, logger=logger)
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


def _split_dataset(dataset, val_ratio: float, seed: int):
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))


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


def _coco_box_from_xyxy(box: torch.Tensor) -> list[float]:
    xmin, ymin, xmax, ymax = box.tolist()
    return [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]


def _build_val_loader_and_classes(
    dataset_dir: Path,
    train_ann: Path,
    val_ann: Path,
    config: TrainConfig,
    logging_logger: logging.Logger,
    *,
    override_val_ratio: Optional[float] = None,
) -> tuple[DataLoader, int, int, int]:
    transform = _build_detection_transforms(config.imgsz, logging_logger)
    logging_logger.info("[SETUP] Construindo datasets COCO para validação pós-treinamento")
    train_ds_full = CocoDetectionDataset(dataset_dir / "images" / "train", train_ann, transforms=transform)
    val_ds_full = CocoDetectionDataset(dataset_dir / "images" / "val", val_ann, transforms=transform)

    val_ratio = config.val_ratio if override_val_ratio is None else override_val_ratio
    if val_ratio and val_ratio > 0:
        logging_logger.info("[SETUP] Aplicando split adicional train/val com val_ratio=%.3f seed=%s", val_ratio, config.seed)
        train_ds_full, extra_val = _split_dataset(train_ds_full, val_ratio, config.seed)
        val_ds_full = extra_val

    describe_dataloader(train_ds_full, logging_logger.info)
    logging_logger.info("[SETUP] Tamanho train=%d | val=%d", len(train_ds_full), len(val_ds_full))

    configured_dataset_classes = getattr(config, "dataset_num_classes", None)
    configured_model_num_classes = getattr(config, "num_classes", None)

    inferred_train_classes = _normalize_dataset_class_count(_infer_num_classes_from_dataset(train_ds_full))
    inferred_val_classes = _normalize_dataset_class_count(_infer_num_classes_from_dataset(val_ds_full))

    dataset_num_classes = configured_dataset_classes if configured_dataset_classes is not None else inferred_train_classes

    looks_like_visdrone = _is_visdrone_dataset(dataset_dir, train_ds_full) or _is_visdrone_dataset(dataset_dir, val_ds_full)
    if dataset_num_classes is None and looks_like_visdrone:
        dataset_num_classes = VISDRONE_MULTICLASS_DATASET_CLASSES

    if dataset_num_classes is None and configured_model_num_classes is not None and configured_model_num_classes > 0:
        dataset_num_classes = max(1, configured_model_num_classes - 1)

    if dataset_num_classes is None:
        raise ValueError("Não foi possível inferir num_classes a partir do dataset; verifique as categorias.")

    if inferred_val_classes is not None and inferred_val_classes != dataset_num_classes:
        logging_logger.warning(
            "[AUDIT] num_classes do val (%s) difere do train (%s); usando o do train.",
            inferred_val_classes,
            dataset_num_classes,
        )

    model_num_classes = configured_model_num_classes
    if model_num_classes is None:
        model_num_classes = dataset_num_classes + 1 if dataset_num_classes > 0 else dataset_num_classes

    if looks_like_visdrone:
        logging_logger.info(
            "[DATASET] VisDrone multi-class: dataset_classes=%d -> model_num_classes=%d (incluindo background)",
            dataset_num_classes,
            model_num_classes,
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
    logging_logger: logging.Logger,
    *,
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
            _validate_targets_batch(targets, num_classes=num_classes, image_sizes=image_sizes, logger=logging_logger)

            model.train()
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
) -> dict:
    COCO, COCOeval = _ensure_pycocotools()
    coco_gt = COCO(str(val_ann))

    logging_logger.info("%s Ground truth annotations: %s", tag, val_ann)

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions_coco.json"
    predictions_path.unlink(missing_ok=True)
    logging_logger.info("%s Gerando predictions_coco.json em %s", tag, predictions_path)

    predictions: list[dict[str, float | int | list[float]]] = []
    categories = coco_gt.dataset.get("categories", []) if hasattr(coco_gt, "dataset") else []
    ordered_cat_ids = [cat.get("id") for cat in categories if "id" in cat]
    fallback_cat_ids = sorted(coco_gt.getCatIds())
    cat_ids_order = ordered_cat_ids or fallback_cat_ids

    label_mapping: dict[int, int] = {int(cat_id): int(cat_id) for cat_id in cat_ids_order}
    contiguous_mapping = {idx + 1: int(cat_id) for idx, cat_id in enumerate(cat_ids_order)}
    for lbl, cat_id in contiguous_mapping.items():
        label_mapping.setdefault(lbl, cat_id)
    logging_logger.info("%s label->category_id mapping: %s", tag, label_mapping)

    model.eval()
    predicted_image_ids: set[int] = set()
    predicted_category_ids: set[int] = set()
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device_str) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                image_id = _extract_image_id(target)
                boxes = output.get("boxes")
                scores = output.get("scores")
                labels = output.get("labels")

                if boxes is None or scores is None or labels is None:
                    raise RuntimeError("Saída do modelo incompatível: esperado boxes, scores e labels.")

                boxes_cpu = boxes.detach().cpu()
                scores_cpu = scores.detach().cpu()
                labels_cpu = labels.detach().cpu()

                for box, score, label in zip(boxes_cpu, scores_cpu, labels_cpu):
                    category_id = label_mapping.get(int(label))
                    if category_id is None:
                        raise RuntimeError(
                            f"label {int(label)} não encontrado no mapping; revise categorias ou mapeamento."
                        )
                    predicted_image_ids.add(int(image_id))
                    predicted_category_ids.add(int(category_id))
                    predictions.append(
                        {
                            "image_id": int(image_id),
                            "category_id": category_id,
                            "bbox": _coco_box_from_xyxy(box),
                            "score": float(score),
                        }
                    )

    predictions_path.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    gt_image_ids = set(int(x) for x in coco_gt.getImgIds())
    gt_category_ids = set(int(x) for x in coco_gt.getCatIds())

    invalid_image_ids = sorted(predicted_image_ids - gt_image_ids)
    invalid_category_ids = sorted(predicted_category_ids - gt_category_ids)

    if invalid_image_ids:
        raise RuntimeError(
            (
                f"{tag} image_id inválidos detectados: {len(invalid_image_ids)} fora do conjunto GT. "
                f"Exemplos: {invalid_image_ids[:5]} | GT={val_ann} | preds={predictions_path}"
            )
        )
    logging_logger.info("%s image_id check OK (%d imgs)", tag, len(predicted_image_ids))

    if invalid_category_ids:
        raise RuntimeError(
            (
                f"{tag} category_id inválidos detectados: {len(invalid_category_ids)} fora do conjunto GT. "
                f"Exemplos: {invalid_category_ids[:5]} | GT={val_ann} | preds={predictions_path}"
            )
        )
    logging_logger.info("%s category_id check OK (%d cats)", tag, len(predicted_category_ids))

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

    logging_logger.info(
        "%s AP=%.4f, AP50=%.4f, AP75=%.4f, AR100=%.4f",
        tag,
        metrics["AP"],
        metrics["AP50"],
        metrics["AP75"],
        metrics["AR100"],
    )

    return {
        "coco_metrics": metrics,
        "coco_stats": stats,
        "predictions_coco_json": str(predictions_path),
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

    val_loader, model_num_classes, dataset_num_classes, num_workers = _build_val_loader_and_classes(
        dataset_dir, train_ann, val_ann, config, logging_logger
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = log_dir / run_tag / "val_post" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    model = model_builder(model_num_classes)
    model.to(device)

    loaded = torch.load(weights_path.expanduser().resolve(), map_location="cpu")
    state_dict, checkpoint_format = _extract_state_dict(loaded)
    _emit(f"[{run_tag.upper()}][VAL-POST] Formato de checkpoint: {checkpoint_format}")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        _emit(f"[{run_tag.upper()}][VAL-POST] missing_keys: {missing}")
    if unexpected:
        _emit(f"[{run_tag.upper()}][VAL-POST] unexpected_keys: {unexpected}")

    val_mode = getattr(config, "val_mode", "loss")
    if val_mode not in {"loss", "metrics"}:
        logging_logger.warning("[VAL] val_mode desconhecido %s; forçando 'loss'", val_mode)
        val_mode = "loss"

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
            logging_logger=logging_logger,
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
    logging_logger, log_path = _configure_logging(config.verbose, log_dir, logger, stream_override=safe_stdout)
    try:
        faulthandler.enable(file=safe_stderr)
    except Exception as exc:  # pragma: no cover - compatibilidade com ambientes sem fileno
        logging_logger.warning("Não foi possível habilitar faulthandler no stream seguro: %s", exc)
    try:
        signal.signal(signal.SIGTERM, lambda _sig, _frame: sys.exit(1))
    except Exception:  # pragma: no cover - compatibilidade com ambientes sem suporte
        logging_logger.warning("Não foi possível registrar handler de SIGTERM neste ambiente.")

    logging_logger.info("Iniciando setup de treinamento SSD...")
    device_str = resolve_device(config.device)
    device = torch.device(device_str)
    seed_everything(config.seed)
    logging_logger.info("Dispositivo: %s | torch=%s | cuda_available=%s", device_str, torch.__version__, torch.cuda.is_available())
    logging_logger.info("Configuração: %s", config)

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else Path(weights_out).expanduser().resolve().parent
    ckpt_dir = ckpt_dir.expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logging_logger.info("Checkpoints serão salvos em: %s", ckpt_dir)

    watchdog_stop: Optional[threading.Event] = None
    watchdog_pause: Optional[threading.Event] = None
    watchdog_thread: Optional[threading.Thread] = None
    last_progress = [time.monotonic()]
    epoch = 0
    optimizer: Optional[torch.optim.Optimizer] = None
    try:
        val_ratio_to_use = config.val_ratio if val_ratio is None else val_ratio

        if train_dataset is None or val_dataset is None:
            transform = _build_detection_transforms(config.imgsz, logging_logger)
            logging_logger.info("[SETUP] Construindo datasets COCO a partir de %s", dataset_dir)
            train_ds_full = CocoDetectionDataset(dataset_dir / "images" / "train", train_ann, transforms=transform)
            val_ds_full = CocoDetectionDataset(dataset_dir / "images" / "val", val_ann, transforms=transform)

            # Dividir train em train/val adicionais se desejado
            if val_ratio_to_use and val_ratio_to_use > 0:
                logging_logger.info(
                    "[SETUP] Aplicando split adicional train/val com val_ratio=%.3f seed=%s", val_ratio_to_use, config.seed
                )
                train_ds_full, extra_val = _split_dataset(train_ds_full, val_ratio_to_use, config.seed)
                val_ds_full = extra_val
        else:
            logging_logger.info("[SETUP] Usando datasets pré-construídos (train/val).")
            train_ds_full = train_dataset
            val_ds_full = val_dataset

        describe_dataloader(train_ds_full, logging_logger.info)
        logging_logger.info("[SETUP] Tamanho train=%d | val=%d", len(train_ds_full), len(val_ds_full))

        configured_dataset_classes = getattr(config, "dataset_num_classes", None)
        configured_model_num_classes = getattr(config, "num_classes", None)

        inferred_train_classes = _normalize_dataset_class_count(_infer_num_classes_from_dataset(train_ds_full))
        inferred_val_classes = _normalize_dataset_class_count(_infer_num_classes_from_dataset(val_ds_full))

        dataset_num_classes = configured_dataset_classes if configured_dataset_classes is not None else inferred_train_classes

        looks_like_visdrone = _is_visdrone_dataset(dataset_dir, train_ds_full) or _is_visdrone_dataset(
            dataset_dir, val_ds_full
        )
        if dataset_num_classes is None and looks_like_visdrone:
            dataset_num_classes = VISDRONE_MULTICLASS_DATASET_CLASSES

        if dataset_num_classes is None and configured_model_num_classes is not None and configured_model_num_classes > 0:
            dataset_num_classes = max(1, configured_model_num_classes - 1)

        if dataset_num_classes is None:
            raise ValueError("Não foi possível inferir num_classes a partir do dataset; verifique as categorias.")

        if inferred_val_classes is not None and inferred_val_classes != dataset_num_classes:
            logging_logger.warning(
                "[AUDIT] num_classes do val (%s) difere do train (%s); usando o do train.",
                inferred_val_classes,
                dataset_num_classes,
            )

        model_num_classes = configured_model_num_classes
        if model_num_classes is None:
            model_num_classes = dataset_num_classes + 1 if dataset_num_classes > 0 else dataset_num_classes

        if looks_like_visdrone:
            logging_logger.info(
                "[DATASET] VisDrone multi-class: dataset_classes=%d -> model_num_classes=%d (incluindo background)",
                dataset_num_classes,
                model_num_classes,
            )

        if hasattr(train_ds_full, "transforms") and hasattr(val_ds_full, "transforms"):
            if type(getattr(train_ds_full, "transforms")) != type(getattr(val_ds_full, "transforms")):
                logging_logger.warning(
                    "[TRANSFORM] Transforms de train e val divergem em tipo (%s vs %s)",
                    type(getattr(train_ds_full, "transforms")),
                    type(getattr(val_ds_full, "transforms")),
                )

        if config.audit_datasets:
            _audit_dataset(train_ds_full, "train", num_classes=model_num_classes, logger=logging_logger)
            _audit_dataset(val_ds_full, "val", num_classes=model_num_classes, logger=logging_logger)

        dataloader_kwargs = dict(
            batch_size=config.batch_size,
            collate_fn=coco_collate,
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

        for epoch in range(1, config.epochs + 1):
            model.train()
            running_loss = 0.0
            epoch_wall_start = time.perf_counter()
            last_heartbeat = time.perf_counter()
            total_batches = len(train_loader)
            logging_logger.info("Epoch %d/%d | num_batches=%d", epoch, config.epochs, total_batches)
            progress = None
            if tqdm:
                try:
                    progress = tqdm(
                        total=total_batches,
                        desc=f"Epoch {epoch}/{config.epochs}",
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

                image_sizes = [(int(img.shape[-2]), int(img.shape[-1])) for img in images]
                _validate_targets_batch(targets, num_classes=num_classes, image_sizes=image_sizes, logger=logging_logger)

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
                    }
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
                config.epochs,
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
                                targets, num_classes=num_classes, image_sizes=image_sizes, logger=logging_logger
                            )

                            model.train()
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

            with _watchdog_paused(last_progress, watchdog_pause):
                epoch_ckpt = ckpt_dir / f"epoch_{epoch:03d}.pth"
                atomic_torch_save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict() if optimizer else None,
                        "loss": float(avg_loss),
                    },
                    epoch_ckpt,
                )
                logging_logger.info("Checkpoint da época %d salvo em %s", epoch, epoch_ckpt)

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
            atomic_torch_save(model.state_dict(), out)
            ensure_weights_size(out, logger=logging_logger.info)
            weights_out = out
            logging_logger.info("Treinamento finalizado com sucesso.")

        return Metrics(
            precision=0.0,
            recall=0.0,
            map50=0.0,
            map50_95=0.0,
            loss_final=last_loss,
            epochs=config.epochs,
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
                final_epoch = epoch if epoch else 0
                last_ckpt = ckpt_dir / "last.pth"
                atomic_torch_save(
                    {
                        "epoch": final_epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict() if optimizer else None,
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
