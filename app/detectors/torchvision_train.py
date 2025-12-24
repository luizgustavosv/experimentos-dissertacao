from __future__ import annotations

import faulthandler
import json
import logging
import signal
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from app.detectors.base import Logger
from app.detectors.config import TrainConfig
from app.detectors.dataset_coco import CocoDetectionDataset
from app.detectors.utils import coco_collate, describe_dataloader, ensure_weights_size, resolve_device, seed_everything
from app.metrics import Metrics

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback silencioso se tqdm não estiver disponível
    tqdm = None


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

    def _accumulate(obj: Any, prefix: str = "") -> torch.Tensor:
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
                reduced = torch.tensor(0.0, device=device)
            elif obj.dim() == 0:
                reduced = obj
            else:
                reduced = obj.mean()
            name = prefix[:-1] if prefix.endswith(".") else prefix
            if name:
                try:
                    loss_items[name] = float(reduced.detach().cpu())
                except Exception:
                    logger.debug("Falha ao registrar loss_item para %s", name)
            return reduced

        if isinstance(obj, (int, float)):
            name = prefix[:-1] if prefix.endswith(".") else prefix
            if name:
                loss_items[name] = float(obj)
            return torch.tensor(float(obj), device=device)

        tname = type(obj).__name__
        if tname not in warned_types:
            warned_types.add(tname)
            logger.warning("_normalize_losses ignorando tipo inesperado em loss_out: %s", tname)
        return total

    try:
        loss_total = _accumulate(loss_out)
    except Exception:
        logger.exception("_normalize_losses falhou; retornando 0.0")
        return torch.tensor(0.0, device=device), {}

    if loss_total.dim() > 0:
        loss_total = loss_total.mean()

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
    verbose: bool, log_dir: Path, external_logger: Optional[Logger], stream_override=None
) -> tuple[logging.Logger, Path]:
    logger = logging.getLogger("ssd_train")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    log_dir = log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"ssd_train_{datetime.now():%Y%m%d_%H%M%S}.log"
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


def _validate_targets_batch(
    targets: list[dict[str, torch.Tensor]],
    *,
    num_classes: Optional[int],
    image_sizes: Iterable[tuple[int, int]],
    logger: logging.Logger,
) -> None:
    sizes = list(image_sizes)
    for idx, target in enumerate(targets):
        if not isinstance(target, dict):
            raise TypeError(f"Target no índice {idx} não é um dicionário: {type(target).__name__}")

        if "boxes" not in target or "labels" not in target:
            raise ValueError(f"Target no índice {idx} deve conter 'boxes' e 'labels'")

        boxes = target["boxes"]
        labels = target["labels"]

        if not torch.is_tensor(boxes):
            raise TypeError(f"Target boxes no índice {idx} não é tensor")
        if not torch.is_tensor(labels):
            raise TypeError(f"Target labels no índice {idx} não é tensor")

        if boxes.dim() != 2 or boxes.size(1) != 4:
            raise ValueError(f"Boxes no índice {idx} devem ter shape [N,4], recebido {tuple(boxes.shape)}")
        if not torch.isfinite(boxes).all():
            raise ValueError(f"Boxes não finitos no índice {idx}")

        if labels.dim() != 1:
            raise ValueError(f"Labels no índice {idx} devem ser 1D, recebido {tuple(labels.shape)}")
        if boxes.size(0) != labels.numel():
            raise ValueError(f"Mismatch boxes/labels no índice {idx}: {boxes.size(0)} vs {labels.numel()}")
        if labels.dtype != torch.int64:
            raise TypeError(f"Labels no índice {idx} devem ser int64, recebido {labels.dtype}")

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
            raise ValueError(f"xmax <= xmin encontrado no índice {idx}")
        if not (y_max > y_min).all():
            raise ValueError(f"ymax <= ymin encontrado no índice {idx}")

        if num_classes is not None and num_classes > 0:
            if (labels < 1).any():
                raise ValueError(f"Labels fora do intervalo no índice {idx}: mínimo {int(labels.min())}")
            if (labels >= num_classes).any():
                raise ValueError(
                    f"Labels fora do intervalo no índice {idx}: máximo {int(labels.max())} >= num_classes ({num_classes})"
                )
        else:
            if (labels < 1).any():
                raise ValueError(f"Labels devem ser >=1 quando num_classes é desconhecido (índice {idx})")

        if not torch.isfinite(labels.float()).all():
            raise ValueError(f"Labels não finitos no índice {idx}")

        if boxes.dtype != torch.float32:
            logger.debug("Convertendo boxes para float32 no índice %d", idx)
            target["boxes"] = boxes.float()


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


def train_torchvision_detector(
    model: torch.nn.Module,
    dataset_dir: Path,
    train_ann: Path,
    val_ann: Path,
    weights_out: Path,
    config: TrainConfig,
    logger: Optional[Logger] = None,
    val_ratio: float = 0.1,
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
        if train_dataset is None or val_dataset is None:
            transform = transforms.Compose([transforms.ToTensor()])
            logging_logger.info("[SETUP] Construindo datasets COCO a partir de %s", dataset_dir)
            train_ds_full = CocoDetectionDataset(dataset_dir / "images" / "train", train_ann, transforms=transform)
            val_ds_full = CocoDetectionDataset(dataset_dir / "images" / "val", val_ann, transforms=transform)

            # Dividir train em train/val adicionais se desejado
            if val_ratio > 0:
                logging_logger.info("[SETUP] Aplicando split adicional train/val com val_ratio=%.3f seed=%s", val_ratio, config.seed)
                train_ds_full, extra_val = _split_dataset(train_ds_full, val_ratio, config.seed)
                val_ds_full = extra_val
        else:
            logging_logger.info("[SETUP] Usando datasets pré-construídos (train/val).")
            train_ds_full = train_dataset
            val_ds_full = val_dataset

        describe_dataloader(train_ds_full, logging_logger.info)
        logging_logger.info("[SETUP] Tamanho train=%d | val=%d", len(train_ds_full), len(val_ds_full))

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

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging_logger.info("[MODEL] Modelo preparado. params=%d | treináveis=%d", total_params, trainable_params)
        logging_logger.info("[SETUP] Movendo modelo para %s e preparando otimizador.", device_str)
        model.to(device_str)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)
        logging_logger.info("Starting training...")

        num_classes = _infer_num_classes_from_model(model)
        if num_classes is None:
            logging_logger.debug("Não foi possível inferir num_classes do modelo; validação usará apenas limites mínimos.")
        else:
            logging_logger.info("num_classes inferido: %s", num_classes)

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
                targets = [{k: v.to(device_str) for k, v in t.items()} for t in targets]

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
                    logging_logger.error(
                        "Loss não finito detectado. epoch=%s step=%s image_ids=%s paths=%s",
                        epoch,
                        step,
                        image_ids,
                        file_paths if file_paths else "<indisponível>",
                    )
                    bad_batch = {
                        "epoch": epoch,
                        "step": step,
                        "image_ids": image_ids,
                        "paths": file_paths,
                    }
                    try:
                        bad_path = ckpt_dir / "bad_batch.json"
                        bad_path.write_text(json.dumps(bad_batch, indent=2, ensure_ascii=False), encoding="utf-8")
                        logging_logger.error("Batch problemático salvo em %s", bad_path)
                    except Exception:
                        logging_logger.exception("Falha ao salvar bad_batch.json")
                    raise RuntimeError("Non-finite loss")

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

            model.eval()
            val_loss = 0.0
            with _watchdog_paused(last_progress, watchdog_pause):
                with torch.no_grad():
                    for vstep, (images, targets) in enumerate(val_loader, start=1):
                        images = [img.to(device_str) for img in images]
                        targets = [{k: v.to(device_str) for k, v in t.items()} for t in targets]
                        image_sizes = [(int(img.shape[-2]), int(img.shape[-1])) for img in images]
                        _validate_targets_batch(targets, num_classes=num_classes, image_sizes=image_sizes, logger=logging_logger)
                        loss_out = model(images, targets)
                        try:
                            losses, loss_items = _normalize_losses(loss_out, device)
                        except Exception:
                            logging_logger.exception("Falha ao normalizar losses no val; usando 0.0 para continuar.")
                            losses = torch.tensor(0.0, device=device)
                            loss_items = {}
                        if not torch.isfinite(losses):
                            image_ids, file_paths = _collect_batch_identifiers(targets, val_loader.dataset)
                            logging_logger.error(
                                "Loss não finito detectado no val. epoch=%s step=%s image_ids=%s paths=%s",
                                epoch,
                                vstep,
                                image_ids,
                                file_paths if file_paths else "<indisponível>",
                            )
                            bad_batch = {
                                "epoch": epoch,
                                "step": vstep,
                                "phase": "val",
                                "image_ids": image_ids,
                                "paths": file_paths,
                            }
                            try:
                                bad_path = ckpt_dir / "bad_batch.json"
                                bad_path.write_text(json.dumps(bad_batch, indent=2, ensure_ascii=False), encoding="utf-8")
                                logging_logger.error("Batch problemático de validação salvo em %s", bad_path)
                            except Exception:
                                logging_logger.exception("Falha ao salvar bad_batch.json de validação")
                            raise RuntimeError("Non-finite loss")
                        if loss_items:
                            logging_logger.debug(f"Loss components: {loss_items}")
                        val_loss += losses.item()
            val_loss = val_loss / max(1, len(val_loader))
            logging_logger.info(f"[VAL] Época {epoch} | loss={val_loss:.4f}")

            with _watchdog_paused(last_progress, watchdog_pause):
                epoch_ckpt = ckpt_dir / f"epoch_{epoch:03d}.pth"
                torch.save(
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
            torch.save(model.state_dict(), str(out))
            ensure_weights_size(out)
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
                torch.save(
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
