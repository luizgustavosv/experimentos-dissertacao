"""
Script de treinamento YOLO (Ultralytics) instrumentado para o dataset VisDrone.

Exemplo de uso no Windows (PowerShell/CMD):
python C:\\experimentos-dissertacao\\app\\detectors\\train_yolo_visdrone_instrumented.py ^
  --data E:\\datasets\\visdrone\\visdrone.yaml ^
  --model yolo11n.pt ^
  --epochs 50 ^
  --imgsz 640 ^
  --batch 16 ^
  --device 0 ^
  --workers 2 ^
  --project-dir C:\\experimentos-dissertacao\\runs_yolo ^
  --name visdrone_yolo ^
  --seed 42 ^
  --log-every 10 ^
  --verbose

Requisitos:
- Não altera o algoritmo de treinamento; apenas adiciona observabilidade e registros em disco.
- Usa callbacks reais da API Python do Ultralytics (YOLO).
- Garante saída contínua no console (stdout “seguro” via sys.__stdout__).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import random
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Dict, Iterable, Optional

import numpy as np


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


def _resolve_console_stream(fallback_path: Path) -> IO[str]:
    """Escolhe um stream seguro para o console, com fallback para arquivo."""

    for candidate in (getattr(sys, "__stdout__", None), getattr(sys, "__stderr__", None), sys.stdout, sys.stderr):
        if candidate:
            try:
                candidate.write("")
                candidate.flush()
            except Exception:
                continue
            return candidate  # type: ignore[return-value]

    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    return fallback_path.open("a", encoding="utf-8")


class _FlushStreamHandler(logging.StreamHandler):
    """StreamHandler que sempre força flush após cada registro."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - robustez de logging
        super().emit(record)
        try:
            self.flush()
        except Exception:
            return


def _setup_logging(log_dir: Path, verbose: bool) -> tuple[logging.Logger, Path]:
    """Configura logger com saída em console seguro (stdout) + arquivo."""

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = logging.getLogger("app.detectors.yolo_visdrone_instrumented")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)
    console_handler = _FlushStreamHandler(_resolve_console_stream(log_path))
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info("Logger inicializado. log_path=%s", log_path)
    return logger, log_path


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch pode não estar disponível em alguns ambientes; seguir mesmo assim.
        return


def _capture_env() -> Dict[str, Any]:
    env = {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
    try:
        import torch

        env.update(
            {
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cudnn_enabled": bool(torch.backends.cudnn.is_available()),
            }
        )
        if torch.cuda.is_available():
            try:
                env["cuda_device_name"] = torch.cuda.get_device_name(0)
            except Exception:
                env["cuda_device_name"] = None
    except Exception:
        env.update({"torch_version": None, "cuda_available": None, "cudnn_enabled": None})

    try:
        import ultralytics

        env["ultralytics_version"] = getattr(ultralytics, "__version__", None)
    except Exception:
        env["ultralytics_version"] = None
    return env


@dataclass
class RunConfig:
    data: str
    model: str
    epochs: int
    imgsz: int
    batch: int
    device: str
    workers: int
    project_dir: str
    name: str
    run_id: str
    run_dir: str
    seed: int
    log_every: int
    verbose: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _format_loss_items(loss_items: Any) -> Dict[str, Optional[float]]:
    """Extrai perdas conhecidas (box/cls/dfl) e total se possível."""

    result: Dict[str, Optional[float]] = {"box_loss": None, "cls_loss": None, "dfl_loss": None, "total_loss": None}
    if loss_items is None:
        return result
    if isinstance(loss_items, dict):
        for key in ("box", "cls", "dfl", "loss"):
            if key in loss_items and isinstance(loss_items[key], (int, float)):
                if key == "loss":
                    result["total_loss"] = float(loss_items[key])
                elif key == "box":
                    result["box_loss"] = float(loss_items[key])
                elif key == "cls":
                    result["cls_loss"] = float(loss_items[key])
                elif key == "dfl":
                    result["dfl_loss"] = float(loss_items[key])
        if result["total_loss"] is None and all(v is not None for v in (result["box_loss"], result["cls_loss"])):
            parts = [result["box_loss"], result["cls_loss"], result["dfl_loss"]]
            result["total_loss"] = float(sum(p for p in parts if p is not None))
        return result

    if isinstance(loss_items, (list, tuple)):
        if len(loss_items) > 0 and isinstance(loss_items[0], (float, int)):
            result["box_loss"] = float(loss_items[0])
        if len(loss_items) > 1 and isinstance(loss_items[1], (float, int)):
            result["cls_loss"] = float(loss_items[1])
        if len(loss_items) > 2 and isinstance(loss_items[2], (float, int)):
            result["dfl_loss"] = float(loss_items[2])
        result["total_loss"] = float(sum(v for v in loss_items if isinstance(v, (float, int))))
        return result

    if isinstance(loss_items, (float, int)):
        result["total_loss"] = float(loss_items)
        return result

    return result


def _safe_lr(trainer) -> Optional[Iterable[float]]:
    try:
        if getattr(trainer, "optimizer", None):
            return [group.get("lr") for group in trainer.optimizer.param_groups]
    except Exception:
        return None
    return None


def _append_csv(path: Path, fieldnames: Iterable[str], row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


class TelemetryCallbacks:
    """Callbacks para registrar telemetria em tempo real e persistir métricas."""

    def __init__(
        self,
        logger: logging.Logger,
        metrics_epoch_path: Path,
        metrics_step_path: Path,
        log_every: int,
    ) -> None:
        self.logger = logger
        self.metrics_epoch_path = metrics_epoch_path
        self.metrics_step_path = metrics_step_path
        self.log_every = max(1, log_every)
        self.start_time = time.perf_counter()
        self.epoch_start_time = time.perf_counter()
        self.nb_batches: Optional[int] = None
        self.total_epochs: Optional[int] = None
        self.save_dir: Optional[Path] = None
        self.current_epoch: Optional[int] = None
        self.current_batch: Optional[int] = None

    def _eta_seconds(self, epoch: int, batch_i: int) -> Optional[float]:
        if self.nb_batches and self.total_epochs:
            total_steps = self.nb_batches * self.total_epochs
            steps_done = epoch * self.nb_batches + batch_i + 1
            elapsed = time.perf_counter() - self.start_time
            if steps_done > 0 and steps_done < total_steps:
                return (elapsed / steps_done) * (total_steps - steps_done)
        return None

    def on_train_start(self, trainer) -> None:  # pragma: no cover - depende de ultralytics
        self.start_time = time.perf_counter()
        self.epoch_start_time = self.start_time
        self.nb_batches = getattr(trainer, "nb", None)
        self.total_epochs = getattr(trainer, "epochs", None)
        self.save_dir = Path(getattr(trainer, "save_dir", "")) if getattr(trainer, "save_dir", None) else None
        args = getattr(trainer, "args", None)
        self.logger.info(
            "[train_start] epochs=%s batches_per_epoch=%s device=%s imgsz=%s batch=%s workers=%s save_dir=%s",
            self.total_epochs,
            self.nb_batches,
            getattr(args, "device", None),
            getattr(args, "imgsz", None),
            getattr(args, "batch", None),
            getattr(args, "workers", None),
            self.save_dir,
        )

    def on_train_epoch_start(self, trainer) -> None:  # pragma: no cover - depende de ultralytics
        self.current_epoch = getattr(trainer, "epoch", None)
        self.epoch_start_time = time.perf_counter()
        self.logger.info("[epoch_start] epoch=%s", (self.current_epoch or 0) + 1)

    def on_train_batch_end(self, trainer) -> None:  # pragma: no cover - depende de ultralytics
        batch_i = getattr(trainer, "batch_i", getattr(trainer, "i", None))
        epoch = getattr(trainer, "epoch", 0)
        if batch_i is None:
            return
        self.current_epoch = epoch
        self.current_batch = batch_i
        if (batch_i + 1) % self.log_every != 0:
            return

        losses = _format_loss_items(getattr(trainer, "loss_items", None))
        lrs = _safe_lr(trainer)
        elapsed = time.perf_counter() - self.start_time
        iter_time = time.perf_counter() - getattr(trainer, "batch_time", time.perf_counter())
        eta_seconds = self._eta_seconds(epoch, batch_i)

        log_parts = [
            f"[train][epoch {epoch+1}/{self.total_epochs or '?'}]",
            f"batch {batch_i+1}/{self.nb_batches or '?'}",
        ]
        for key in ("box_loss", "cls_loss", "dfl_loss", "total_loss"):
            if losses.get(key) is not None:
                log_parts.append(f"{key}={losses[key]:.4f}")
        if lrs:
            lr_str = ", ".join(f"{lr:.6f}" for lr in lrs if lr is not None)
            if lr_str:
                log_parts.append(f"lr=[{lr_str}]")
        log_parts.append(f"elapsed={elapsed:.1f}s")
        if eta_seconds is not None:
            log_parts.append(f"eta={eta_seconds/60:.1f}min")
        self.logger.info(" ".join(log_parts))

        row = {
            "epoch": epoch + 1,
            "batch": batch_i + 1,
            "box_loss": losses.get("box_loss"),
            "cls_loss": losses.get("cls_loss"),
            "dfl_loss": losses.get("dfl_loss"),
            "total_loss": losses.get("total_loss"),
            "lr": ";".join(f"{lr:.8f}" for lr in lrs if lr is not None) if lrs else None,
            "elapsed_seconds": round(elapsed, 4),
            "iter_time_seconds": round(iter_time, 4),
        }
        _append_csv(self.metrics_step_path, row.keys(), row)

    def on_train_epoch_end(self, trainer) -> None:  # pragma: no cover - depende de ultralytics
        epoch = getattr(trainer, "epoch", 0)
        losses = _format_loss_items(getattr(trainer, "loss_items", None))
        metrics = getattr(trainer, "metrics", None) or {}
        lrs = _safe_lr(trainer)
        epoch_time = time.perf_counter() - self.epoch_start_time
        total_elapsed = time.perf_counter() - self.start_time

        log_parts = [f"[epoch_end] epoch={epoch+1}/{self.total_epochs or '?'}"]
        for key in ("box_loss", "cls_loss", "dfl_loss", "total_loss"):
            if losses.get(key) is not None:
                log_parts.append(f"{key}={losses[key]:.4f}")
        if metrics:
            metric_text = " ".join(
                f"{k}={v:.4f}" if isinstance(v, (float, int)) else f"{k}={v}" for k, v in metrics.items()
            )
            log_parts.append(f"metrics=({metric_text})")
        if lrs:
            lr_str = ", ".join(f"{lr:.6f}" for lr in lrs if lr is not None)
            if lr_str:
                log_parts.append(f"lr=[{lr_str}]")
        log_parts.append(f"epoch_time={epoch_time:.2f}s total_elapsed={total_elapsed:.2f}s")
        self.logger.info(" ".join(log_parts))

        row = {
            "epoch": epoch + 1,
            "box_loss": losses.get("box_loss"),
            "cls_loss": losses.get("cls_loss"),
            "dfl_loss": losses.get("dfl_loss"),
            "total_loss": losses.get("total_loss"),
            "precision": metrics.get("precision") if isinstance(metrics, dict) else None,
            "recall": metrics.get("recall") if isinstance(metrics, dict) else None,
            "map50": metrics.get("map50") if isinstance(metrics, dict) else None,
            "map50_95": metrics.get("map50-95") or metrics.get("map50_95") if isinstance(metrics, dict) else None,
            "lr": ";".join(f"{lr:.8f}" for lr in lrs if lr is not None) if lrs else None,
            "epoch_time_seconds": round(epoch_time, 4),
            "total_elapsed_seconds": round(total_elapsed, 4),
        }
        _append_csv(self.metrics_epoch_path, row.keys(), row)

    def on_fit_epoch_end(self, trainer) -> None:  # pragma: no cover - compatibilidade API
        self.on_train_epoch_end(trainer)

    def on_val_end(self, trainer) -> None:  # pragma: no cover - depende de ultralytics
        metrics = getattr(trainer, "metrics", None) or {}
        if metrics:
            safe_metrics = " ".join(
                f"{k}={v:.4f}" if isinstance(v, (float, int)) else f"{k}={v}" for k, v in metrics.items()
            )
            self.logger.info("[val_end] %s", safe_metrics)

    def on_train_end(self, trainer) -> None:  # pragma: no cover - depende de ultralytics
        self.logger.info(
            "[train_end] treinamento concluído. save_dir=%s total_elapsed=%.2fs",
            getattr(trainer, "save_dir", None),
            time.perf_counter() - self.start_time,
        )


def _start_heartbeat(logger: logging.Logger, stop_event: threading.Event, telemetry: TelemetryCallbacks) -> threading.Thread:
    """Thread daemon que emite heartbeat a cada 10 segundos."""

    def _beat() -> None:
        while not stop_event.wait(10):
            logger.info(
                "[heartbeat] running | elapsed=%.1fs | epoch=%s | batch=%s | save_dir=%s",
                time.perf_counter() - telemetry.start_time,
                (telemetry.current_epoch or 0) + 1 if telemetry.current_epoch is not None else "?",
                (telemetry.current_batch or 0) + 1 if telemetry.current_batch is not None else "?",
                telemetry.save_dir,
            )

    thread = threading.Thread(target=_beat, daemon=True)
    thread.start()
    return thread


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino YOLO/VisDrone com telemetria reprodutível.")
    parser.add_argument("--data", type=Path, required=True, help="Caminho para o yaml do VisDrone.")
    parser.add_argument("--model", type=Path, required=True, help="Checkpoint base (ex.: yolo11n.pt ou yolov8n.pt).")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas.")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolução das imagens (padrão 640).")
    parser.add_argument("--batch", type=int, default=16, help="Tamanho do batch.")
    parser.add_argument("--device", type=str, default="0", help='Dispositivo (ex.: "0" ou "cpu").')
    parser.add_argument("--workers", type=int, default=2, help="Número de workers do dataloader.")
    parser.add_argument("--project-dir", type=Path, required=True, help="Diretório raiz dos experimentos.")
    parser.add_argument("--name", type=str, default="visdrone_yolo", help="Nome do run.")
    parser.add_argument("--seed", type=int, default=42, help="Seed do experimento.")
    parser.add_argument("--log-every", type=int, default=10, help="Logar a cada N batches.")
    parser.add_argument("--verbose", action="store_true", help="Habilita logs detalhados.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = args.project_dir.expanduser().resolve()
    run_root = project_dir / args.name
    run_dir = run_root / run_id
    logs_dir = run_dir / "logs"
    artifacts_dir = run_dir / "artifacts"
    metrics_epoch_path = artifacts_dir / "metrics_epoch.csv"
    metrics_step_path = artifacts_dir / "metrics_step.csv"

    logger, log_path = _setup_logging(logs_dir, verbose=args.verbose)
    print(f"[train] Iniciando run {run_id}. log_file={log_path}", flush=True)

    data_path = args.data.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    run_config = RunConfig(
        data=str(data_path),
        model=str(model_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project_dir=str(project_dir),
        name=args.name,
        run_id=run_id,
        run_dir=str(run_dir),
        seed=args.seed,
        log_every=args.log_every,
        verbose=args.verbose,
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _write_json(artifacts_dir / "env.json", _capture_env())
    _write_json(artifacts_dir / "run_config.json", run_config.to_dict())

    status_path = artifacts_dir / "status.json"
    _write_json(
        status_path,
        {"status": "running", "start": datetime.now().isoformat(), "run_dir": str(run_dir), "log_file": str(log_path)},
    )

    _seed_everything(args.seed)

    from ultralytics import YOLO  # import tardio para evitar custo no carregamento do script

    model = YOLO(str(model_path))
    telemetry = TelemetryCallbacks(logger, metrics_epoch_path, metrics_step_path, log_every=args.log_every)
    callback_map = {
        "on_train_start": telemetry.on_train_start,
        "on_train_epoch_start": telemetry.on_train_epoch_start,
        "on_train_batch_end": telemetry.on_train_batch_end,
        "on_train_epoch_end": telemetry.on_train_epoch_end,
        "on_fit_epoch_end": telemetry.on_fit_epoch_end,
        "on_val_end": telemetry.on_val_end,
        "on_train_end": telemetry.on_train_end,
    }

    for event_name, cb in callback_map.items():
        add_cb = getattr(model, "add_callback", None)
        if callable(add_cb):
            try:
                add_cb(event_name, cb)
            except Exception as exc:  # pragma: no cover - robustez API
                logger.warning("Falha ao registrar callback %s: %s", event_name, exc)
        else:
            logger.warning("API add_callback indisponível; callback %s não registrado.", event_name)

    stop_event = threading.Event()
    heartbeat_thread = _start_heartbeat(logger, stop_event, telemetry)

    best_checkpoint: Optional[str] = None
    last_checkpoint: Optional[str] = None

    try:
        train_results = model.train(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(run_root),
            name=run_id,
            verbose=args.verbose,
        )
        save_dir = Path(getattr(train_results, "save_dir", run_dir))
        best_checkpoint = getattr(train_results, "best", None)
        last_checkpoint = getattr(train_results, "last", None)
        logger.info("[train] Save dir: %s | best=%s | last=%s", save_dir, best_checkpoint, last_checkpoint)
        _write_json(
            status_path,
            {
                "status": "success",
                "end": datetime.now().isoformat(),
                "run_dir": str(save_dir),
                "best": str(best_checkpoint) if best_checkpoint else None,
                "last": str(last_checkpoint) if last_checkpoint else None,
            },
        )
        print(f"[train] Concluído. Run dir: {save_dir}", flush=True)
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        _write_json(
            status_path,
            {
                "status": "failed",
                "end": datetime.now().isoformat(),
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "run_dir": str(run_dir),
            },
        )
        raise
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=2)
        try:
            for handler in list(logger.handlers):
                handler.flush()
        except Exception:
            pass
        print("[train] Finalizado.", flush=True)


if __name__ == "__main__":
    main()
