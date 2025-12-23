from __future__ import annotations

import faulthandler
import logging
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
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


# Exemplo de execução:
# python train_ssd.py --dataset heridal --epochs 1 --verbose --log-every 10 --debug-dataloader


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


def _start_watchdog(
    log_path: Path, last_progress: list[float], stop_event: threading.Event, logger: logging.Logger, safe_stderr=None, timeout: int = 300
):
    def _watch() -> None:  # pragma: no cover - monitoramento em tempo real
        while not stop_event.wait(5):
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
    seed_everything(config.seed)
    logging_logger.info("Dispositivo: %s | torch=%s | cuda_available=%s", device_str, torch.__version__, torch.cuda.is_available())
    logging_logger.info("Configuração: %s", config)

    watchdog_stop: Optional[threading.Event] = None
    watchdog_thread: Optional[threading.Thread] = None
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

        last_progress = [time.monotonic()]
        watchdog_stop = threading.Event()
        watchdog_thread = _start_watchdog(log_path, last_progress, watchdog_stop, logging_logger, safe_stderr=safe_stderr)

        last_loss = None
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

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
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
            with torch.no_grad():
                for images, targets in val_loader:
                    images = [img.to(device_str) for img in images]
                    targets = [{k: v.to(device_str) for k, v in t.items()} for t in targets]
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    val_loss += losses.item()
            val_loss = val_loss / max(1, len(val_loader))
            logging_logger.info(f"[VAL] Época {epoch} | loss={val_loss:.4f}")

        watchdog_stop.set()
        watchdog_thread.join(timeout=1)

        weights_out = weights_out.expanduser().resolve()
        weights_out.parent.mkdir(parents=True, exist_ok=True)
        logging_logger.info("Salvando pesos em %s", weights_out)
        torch.save(model.state_dict(), weights_out)
        ensure_weights_size(weights_out)
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
