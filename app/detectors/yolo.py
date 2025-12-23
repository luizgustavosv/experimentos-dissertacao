from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import IO, List, Optional

import yaml

from app.detectors.base import DetectionAlgorithm, DetectorContext, Logger
from app.detectors.config import TrainConfig
from app.detectors.utils import copy_ultralytics_checkpoint, resolve_device, seed_everything, validate_yolo_dataset
from app.metrics import InferencePerformance, Metrics
from app.reporting.reports import ReportBuilder


MODULE_LOGGER_NAME = "app.detectors.yolo_train"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


class _UiLoggerHandler(logging.Handler):
    """Entrega logs para o logger da interface gráfica sem depender de fileno()."""

    def __init__(self, ui_logger: Logger):
        super().__init__()
        self.ui_logger = ui_logger

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - compatibilidade UI
        try:
            msg = self.format(record)
            self.ui_logger(msg)
        except Exception:
            # Falhas não podem interromper o treinamento.
            return


class _FlushStreamHandler(logging.StreamHandler):
    """StreamHandler que sempre força flush após cada registro."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        try:
            self.flush()
        except Exception:
            return


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


def _build_train_logger(log_dir: Path, ui_logger: Optional[Logger]) -> tuple[logging.Logger, Path]:
    """Cria logger com saída em console + arquivo, seguro para Windows."""

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"yolo_train_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = logging.getLogger(MODULE_LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)
    console_stream = _resolve_console_stream(log_path)
    console_handler = _FlushStreamHandler(console_stream)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    if ui_logger:
        ui_handler = _UiLoggerHandler(ui_logger)
        ui_handler.setFormatter(formatter)
        logger.addHandler(ui_handler)

    logger.info("Logger de treinamento inicializado. log_path=%s", log_path)
    return logger, log_path


class UltralyticsProgressLogger:
    """Callbacks para garantir logs por batch e época em tempo real."""

    def __init__(self, logger: logging.Logger, log_every: int, log_every_seconds: int = 10) -> None:
        self.logger = logger
        self.log_every = max(1, log_every)
        self.log_every_seconds = max(1, log_every_seconds)
        self.start_time = time.perf_counter()
        self.last_batch_log = 0.0
        self.nb_batches: Optional[int] = None
        self.total_epochs: Optional[int] = None
        self.save_dir: Optional[Path] = None

    def _format_losses(self, losses) -> str:
        if losses is None:
            return "loss=?"
        if isinstance(losses, dict):
            parts = [f"{k}={v:.4f}" for k, v in losses.items() if isinstance(v, (int, float))]
            return " ".join(parts) if parts else str(losses)
        if isinstance(losses, (list, tuple)):
            parts = []
            if len(losses) > 0:
                parts.append(f"box={losses[0]:.4f}")
            if len(losses) > 1:
                parts.append(f"cls={losses[1]:.4f}")
            if len(losses) > 2:
                parts.append(f"dfl={losses[2]:.4f}")
            return " ".join(parts) if parts else str(losses)
        try:
            return f"loss={float(losses):.4f}"
        except Exception:
            return str(losses)

    def _eta(self, epoch: int, batch_i: int) -> Optional[float]:
        if self.nb_batches and self.total_epochs:
            total_steps = self.nb_batches * self.total_epochs
            steps_done = epoch * self.nb_batches + batch_i
            elapsed = time.perf_counter() - self.start_time
            if steps_done > 0 and total_steps > steps_done:
                remaining_steps = total_steps - steps_done
                return remaining_steps * (elapsed / steps_done)
        return None

    def on_train_start(self, trainer) -> None:  # pragma: no cover - depende de ultralytics
        self.start_time = time.perf_counter()
        self.nb_batches = getattr(trainer, "nb", None)
        self.total_epochs = getattr(trainer, "epochs", None)
        save_dir = getattr(trainer, "save_dir", None)
        self.save_dir = Path(save_dir) if save_dir else None
        self.logger.info(
            "[train_start] epochs=%s batches_per_epoch=%s device=%s imgsz=%s batch=%s workers=%s save_dir=%s",
            self.total_epochs,
            self.nb_batches,
            getattr(trainer, "args", None).imgsz if getattr(trainer, "args", None) else None,
            getattr(trainer, "args", None).device if getattr(trainer, "args", None) else None,
            getattr(trainer, "args", None).batch if getattr(trainer, "args", None) else None,
            getattr(trainer, "args", None).workers if getattr(trainer, "args", None) else None,
            self.save_dir,
        )

    def on_train_batch_end(self, trainer) -> None:  # pragma: no cover - depende de ultralytics
        batch_i = getattr(trainer, "batch_i", getattr(trainer, "i", None))
        epoch = getattr(trainer, "epoch", 0)
        nb = getattr(trainer, "nb", self.nb_batches)
        if batch_i is None or nb is None:
            return
        now = time.perf_counter()
        if batch_i % self.log_every != 0 and (now - self.last_batch_log) < self.log_every_seconds:
            return
        self.last_batch_log = now
        lr = None
        try:
            lr = trainer.optimizer.param_groups[0].get("lr") if getattr(trainer, "optimizer", None) else None
        except Exception:
            lr = None
        losses = getattr(trainer, "loss_items", None)
        eta_seconds = self._eta(epoch, batch_i)
        msg = (
            f"[train][epoch {epoch+1}/{self.total_epochs or '?'}] "
            f"batch {batch_i+1}/{nb} {self._format_losses(losses)}"
        )
        if lr is not None:
            msg += f" lr={lr:.6f}"
        if eta_seconds is not None:
            msg += f" eta={eta_seconds/60:.1f}min"
        msg += f" elapsed={now - self.start_time:.1f}s"
        self.logger.info(msg)

    def on_train_epoch_end(self, trainer) -> None:  # pragma: no cover - depende de ultralytics
        metrics = getattr(trainer, "metrics", None) or {}
        losses = getattr(trainer, "loss_items", None)
        msg = f"[epoch_end] epoch={getattr(trainer, 'epoch', '?')+1}/{self.total_epochs or '?'} {self._format_losses(losses)}"
        if metrics:
            safe_metrics = " ".join(
                f"{k}={v:.4f}" if isinstance(v, (float, int)) else f"{k}={v}" for k, v in metrics.items()
            )
            msg += f" metrics=({safe_metrics})"
        self.logger.info(msg)

    def on_fit_epoch_end(self, trainer) -> None:  # pragma: no cover - depende de ultralytics
        # Alias para versões que usam on_fit_epoch_end
        return self.on_train_epoch_end(trainer)

    def on_val_end(self, trainer) -> None:  # pragma: no cover - depende de ultralytics
        metrics = getattr(trainer, "metrics", None) or {}
        if metrics:
            safe_metrics = " ".join(
                f"{k}={v:.4f}" if isinstance(v, (float, int)) else f"{k}={v}" for k, v in metrics.items()
            )
            self.logger.info("[val_end] %s", safe_metrics)

    def on_train_end(self, trainer) -> None:  # pragma: no cover - depende de ultralytics
        self.logger.info("[train_end] treinamento concluído. save_dir=%s", getattr(trainer, "save_dir", None))


def _start_heartbeat(logger: logging.Logger, stop_event: threading.Event, save_dir_supplier) -> threading.Thread:
    """Imprime heartbeat a cada 10s para indicar que o treino está ativo."""

    def _beat() -> None:
        while not stop_event.wait(10):
            save_dir_value = save_dir_supplier()
            logger.info(
                "[heartbeat] training running... elapsed=%.1fs save_dir=%s",
                time.perf_counter() - start_time,
                save_dir_value,
            )

    start_time = time.perf_counter()
    thread = threading.Thread(target=_beat, daemon=True)
    thread.start()
    return thread


def _start_results_tail(logger: logging.Logger, stop_event: threading.Event, save_dir_supplier) -> threading.Thread:
    """Tail não bloqueante de results.csv/results.txt para garantir logs."""

    def _tail_file(path: Path, offsets: dict[Path, int]) -> None:
        offsets.setdefault(path, 0)
        try:
            with path.open("r", encoding="utf-8") as fh:
                fh.seek(offsets[path])
                new_data = fh.read()
                offsets[path] = fh.tell()
        except FileNotFoundError:
            return
        except Exception as exc:  # pragma: no cover - robustez
            logger.debug("Tail falhou para %s: %s", path, exc)
            return
        if not new_data:
            return
        for line in new_data.splitlines():
            if line.strip():
                logger.info("[tail]%s %s", " csv" if path.suffix == ".csv" else "", line.strip())

    def _runner() -> None:
        offsets: dict[Path, int] = {}
        while not stop_event.wait(3):
            save_dir_value = save_dir_supplier()
            if not save_dir_value:
                continue
            save_dir_path = Path(save_dir_value)
            csv_path = save_dir_path / "results.csv"
            txt_path = save_dir_path / "results.txt"
            if csv_path.exists():
                _tail_file(csv_path, offsets)
            if txt_path.exists():
                _tail_file(txt_path, offsets)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    return thread


class YoloDetector(DetectionAlgorithm):
    def __init__(self, context: DetectorContext, config: Optional[TrainConfig] = None) -> None:
        super().__init__(context)
        self.config = config or TrainConfig()

    def train(
        self,
        dataset_dir: Path,
        pretrained_weights: Optional[Path],
        weights_out: Path,
        epochs: int,
        early_stop: bool,
        logger: Optional[Logger] = None,
    ) -> Metrics:
        """
        Exemplo (Windows):
            python -m app.detectors.yolo --data C:\\...\\visdrone.yaml --epochs 50 --batch 16 --imgsz 640 \
                --workers 2 --verbose --log-every 10 --yolo-save-dir C:\\experimentos-dissertacao\\runs_yolo_visdrone
        """

        from ultralytics import YOLO  # import tardio para evitar dependências pesadas em import

        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        yaml_path = validate_yolo_dataset(dataset_dir).resolve()
        seed_everything(self.config.seed)
        device_str = resolve_device(self.config.device)
        num_epochs = epochs or self.config.epochs
        log_dir = self.config.log_dir.expanduser().resolve()
        log_every = self.config.log_every or 10
        log_every_seconds = self.config.log_every_seconds
        imgsz = getattr(self.config, "imgsz", 640)
        train_logger, log_path = _build_train_logger(log_dir, logger)
        save_root = (self.config.yolo_save_dir or Path("runs/yolo_visdrone")).expanduser().resolve()
        save_root.mkdir(parents=True, exist_ok=True)

        train_logger.info("[TRAIN] %s iniciando treino com %s épocas em %s", self.context.name, num_epochs, device_str)
        train_logger.info("YOLO train(): dataset.yaml=%s", yaml_path)
        train_logger.info(
            "Parâmetros: imgsz=%s batch=%s workers=%s lr0=%s seed=%s verbose=%s log_every=%s log_every_seconds=%s",
            imgsz,
            self.config.batch_size,
            self.config.num_workers,
            self.config.lr,
            self.config.seed,
            True,
            log_every,
            log_every_seconds,
        )
        train_logger.info("Saídas: log_file=%s yolo_project=%s target_weights=%s", log_path, save_root, weights_out)

        base_weights = pretrained_weights or "yolov8n.pt"
        model = YOLO(base_weights)

        progress_logger = UltralyticsProgressLogger(train_logger, log_every=log_every, log_every_seconds=log_every_seconds)
        callbacks = {
            "on_train_start": progress_logger.on_train_start,
            "on_train_batch_end": progress_logger.on_train_batch_end,
            "on_train_epoch_end": progress_logger.on_train_epoch_end,
            "on_fit_epoch_end": progress_logger.on_fit_epoch_end,
            "on_val_end": progress_logger.on_val_end,
            "on_train_end": progress_logger.on_train_end,
        }
        for event, cb in callbacks.items():
            add_cb = getattr(model, "add_callback", None)
            if callable(add_cb):
                try:
                    add_cb(event, cb)
                except Exception as exc:  # pragma: no cover - compatibilidade API
                    train_logger.debug("Falha ao registrar callback %s: %s", event, exc)

        stop_event = threading.Event()
        heartbeat_thread = _start_heartbeat(train_logger, stop_event, lambda: progress_logger.save_dir)
        tail_thread = _start_results_tail(train_logger, stop_event, lambda: progress_logger.save_dir)
        results = None
        run_dir: Optional[Path] = None
        status_payload = {
            "status": "unknown",
            "message": "",
            "log_file": str(log_path),
            "weights_out": str(weights_out),
        }

        try:
            results = model.train(
                data=str(yaml_path),
                epochs=num_epochs,
                imgsz=imgsz,
                batch=self.config.batch_size,
                device=device_str,
                workers=self.config.num_workers,
                lr0=self.config.lr,
                seed=self.config.seed,
                save=True,
                plots=True,
                verbose=True,
                project=str(save_root),
                name=weights_out.stem,
                exist_ok=True,
                callbacks=callbacks,
            )
            run_dir = Path(results.save_dir)
            weights_path = copy_ultralytics_checkpoint(run_dir, weights_out)
            train_logger.info("[TRAIN] Checkpoint copiado de %s para %s", run_dir, weights_path)
            status_payload.update(
                {
                    "status": "success",
                    "message": "Training finished",
                    "save_dir": str(run_dir),
                    "elapsed_seconds": getattr(results, "train_time", None),
                }
            )
            return Metrics(
                precision=0.0,
                recall=0.0,
                map50=0.0,
                map50_95=0.0,
                loss_final=results.results_dict.get("loss", None) if hasattr(results, "results_dict") else None,
                epochs=num_epochs,
                train_images=len(list((dataset_dir / "images" / "train").glob("*"))),
                device=device_str,
                weights_path=weights_path,
                map_computed=False,
            )
        except Exception as exc:
            train_logger.exception("Training aborted por erro: %s", exc)
            status_payload.update({"status": "failed", "message": str(exc)})
            raise
        finally:
            stop_event.set()
            heartbeat_thread.join(timeout=2)
            tail_thread.join(timeout=2)
            if run_dir and run_dir.exists():
                status_file = run_dir / "train_status.json"
            else:
                status_file = log_dir / "train_status.json"
            try:
                status_file.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")
                train_logger.info("Status final salvo em %s", status_file)
            except Exception as write_exc:  # pragma: no cover - robustez
                train_logger.debug("Falha ao salvar status final: %s", write_exc)
            train_logger.info("Training finished" if status_payload.get("status") == "success" else "Training aborted")

    def infer(
        self,
        images_dir: Path,
        weights_path: Optional[Path],
        report_out: Path,
        pedestrian_only: bool = False,
        logger: Optional[Logger] = None,
    ):
        from ultralytics import YOLO  # import tardio para evitar dependências pesadas em import

        images_dir = images_dir.expanduser().resolve()
        report_out = report_out.expanduser().resolve()

        if weights_path is None:
            raise FileNotFoundError("Pesos obrigatórios para inferência com YOLO não foram informados.")
        weights_path = weights_path.expanduser().resolve()

        if not images_dir.exists() or not images_dir.is_dir():
            raise FileNotFoundError(f"Pasta de imagens inexistente: {images_dir}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Pesos não encontrados: {weights_path}")

        image_paths = self._list_images(images_dir)
        if not image_paths:
            raise ValueError(f"Nenhuma imagem encontrada em {images_dir}")

        device_str = resolve_device(self.config.device)
        if logger:
            logger(f"[INFER] {self.context.name} usando pesos {weights_path} em {device_str}")
            logger(f"[INFER] Total de imagens: {len(image_paths)}")

        model = YOLO(str(weights_path))
        predictions_root = report_out.parent / "predictions"
        start = time.perf_counter()
        predict_kwargs = {
            "source": str(images_dir),
            "imgsz": 640,
            "device": device_str,
            "save": True,
            "project": str(predictions_root),
            "name": report_out.stem,
            "exist_ok": True,
        }
        if pedestrian_only:
            predict_kwargs["classes"] = [0]
        results = model.predict(**predict_kwargs)
        elapsed = time.perf_counter() - start

        total_images = len(image_paths)
        images_per_second = total_images / elapsed if elapsed > 0 else 0.0
        milliseconds_per_image = (elapsed / total_images * 1000) if total_images else 0.0
        performance = InferencePerformance(
            images_per_second=images_per_second,
            milliseconds_per_image=milliseconds_per_image,
        )

        save_dir = (
            Path(results[0].save_dir)
            if results and hasattr(results[0], "save_dir")
            else predictions_root / report_out.stem
        )
        previews = self._collect_detection_previews(results, save_dir)

        report_builder = ReportBuilder(self.context.name)
        report_builder.save_report(
            report_path=report_out,
            metrics=None,
            operation="Inferência",
            source_dir=images_dir,
            inference_performance=performance,
            detection_previews=previews,
            weights_path=weights_path,
        )

        if logger:
            logger(
                f"[INFER] Latência: {performance.images_per_second:.2f} img/s ({performance.milliseconds_per_image:.2f} ms/imagem)"
            )
            logger(f"[INFER] Relatório salvo em {report_out}")

        return performance

    def validate(
        self,
        dataset_yaml_path: Path,
        weights_path: Optional[Path],
        report_out: Path,
        plots_dir: Path,
        pedestrian_only: bool = False,
        logger: Optional[Logger] = None,
    ):
        from ultralytics import YOLO  # import tardio para evitar dependências pesadas em import

        dataset_yaml_path = Path(dataset_yaml_path)
        yaml_path_abs = dataset_yaml_path.resolve(strict=True)
        report_out = report_out.expanduser().resolve()
        plots_dir = plots_dir.expanduser().resolve()

        weights_resolved = weights_path.expanduser().resolve() if weights_path else Path("yolov8n.pt")
        if weights_path is not None and not weights_resolved.exists():
            raise FileNotFoundError(f"Pesos não encontrados: {weights_resolved}")

        device_str = resolve_device(self.config.device)
        if logger:
            logger(f"[VAL] {self.context.name} validando {yaml_path_abs} em {device_str} usando {weights_resolved}")
            if pedestrian_only:
                logger("[VAL] Filtrando apenas classe pedestrian (0) durante a validação")

        cfg = yaml.safe_load(yaml_path_abs.read_text(encoding="utf-8"))
        root = Path(cfg["path"]).expanduser()
        if not root.is_absolute():
            root = (yaml_path_abs.parent / root).resolve()
        train_dir = (root / cfg["train"]).resolve()
        val_dir = (root / cfg["val"]).resolve()

        self._log(
            f"[VAL][DATA] yaml_path_abs={repr(yaml_path_abs)} cfg_path={repr(cfg['path'])} root={repr(root)} "
            f"train_dir={repr(train_dir)} val_dir={repr(val_dir)}",
            logger,
        )
        self._log(
            f"[VAL][DATA] exists? train_dir={train_dir.exists()} val_dir={val_dir.exists()}",
            logger,
        )
        if not val_dir.exists():
            raise Exception(
                "[VAL][DATA] Diretório de validação inexistente "
                f"yaml_path_abs={repr(yaml_path_abs)} cfg_path={repr(cfg['path'])} root={repr(root)} "
                f"train_dir={repr(train_dir)} val_dir={repr(val_dir)} "
                f"train_exists={train_dir.exists()} val_exists={val_dir.exists()}"
            )

        model = YOLO(weights_resolved)
        with self._temporary_cwd(yaml_path_abs.parent):
            results = model.val(
                data=str(yaml_path_abs),
                project=str(plots_dir),
                name=report_out.stem,
                save_json=True,
                plots=True,
                classes=[0] if pedestrian_only else None,
            )

        run_dir = Path(getattr(results, "save_dir", plots_dir / report_out.stem))
        metrics = self._build_metrics_from_results(results, device_str, weights_resolved)

        report_builder = ReportBuilder(self.context.name)
        metrics_plot = run_dir / "metrics_summary.png"
        report_builder.save_plot(metrics_plot, metrics)
        report_builder.save_report(
            report_path=report_out,
            metrics=metrics,
            operation="Validação",
            source_dir=yaml_path_abs.parent,
            plot_path=metrics_plot,
            weights_path=weights_resolved,
        )

        if logger:
            logger(
                f"[VAL] Precisão: {metrics.precision:.3f} | Recall: {metrics.recall:.3f} | mAP@0.50: {metrics.map50:.3f} | mAP@0.50:0.95: {metrics.map50_95:.3f}"
            )
            logger(f"[VAL] Gráficos salvos em {run_dir}")
            logger(f"[VAL] Relatório salvo em {report_out}")

        return metrics

    @staticmethod
    @contextmanager
    def _temporary_cwd(path: Path):
        previous = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(previous)

    def normalize_dataset(self, dataset_type: str, dataset_dir: Path, normalized_dir: Path, logger: Optional[Logger] = None):
        from app.datasets.normalizer import normalize_dataset as normalize_pipeline

        result = normalize_pipeline(dataset_type, self.context.architecture, dataset_dir, normalized_dir, logger=logger)
        yaml_path = result.output_dir / "dataset.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"dataset.yaml não encontrado após normalização em {result.output_dir}")
        if logger:
            logger(f"YOLO normalize(): root = {result.output_dir.resolve()}")
            logger(f"YOLO normalize(): dataset.yaml criado em {yaml_path.resolve()}")
        return result

    @staticmethod
    def _list_images(root: Path) -> List[Path]:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        return sorted([p for p in root.iterdir() if p.suffix.lower() in extensions and p.is_file()])

    @staticmethod
    def _collect_previews(preview_dir: Path, limit: int = 10) -> List[Path]:
        if not preview_dir.exists():
            return []
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        previews = [p for p in preview_dir.iterdir() if p.suffix.lower() in extensions and p.is_file()]
        return sorted(previews)[:limit]

    @staticmethod
    def _collect_detection_previews(results, save_dir: Path, limit: int = 10) -> List[Path]:
        if not results:
            return []
        previews: List[Path] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            img_name = Path(res.path).name
            candidate = save_dir / img_name
            if candidate.exists():
                previews.append(candidate)
            if len(previews) >= limit:
                break
        if len(previews) < limit and save_dir.exists():
            for extra in sorted(save_dir.iterdir()):
                if len(previews) >= limit:
                    break
                if extra in previews:
                    continue
                if extra.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                previews.append(extra)
        return previews[:limit]

    @staticmethod
    def _build_metrics_from_results(results, device_str: str, weights_path: Path) -> Metrics:
        box = getattr(results, "box", None)
        precision = float(getattr(box, "mp", 0.0)) if box is not None else 0.0
        recall = float(getattr(box, "mr", 0.0)) if box is not None else 0.0
        map50 = float(getattr(box, "map50", 0.0)) if box is not None else 0.0
        map50_95 = float(getattr(box, "map", 0.0)) if box is not None else 0.0

        speed = getattr(results, "speed", {}) if results is not None else {}
        extra = {}
        for key in ["preprocess", "inference", "postprocess"]:
            if key in speed:
                extra[f"speed_{key}_ms"] = float(speed[key])

        return Metrics(
            precision=precision,
            recall=recall,
            map50=map50,
            map50_95=map50_95,
            epochs=None,
            train_images=None,
            device=device_str,
            weights_path=weights_path,
            map_computed=True,
            extra=extra,
        )


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Treino YOLO com logs em tempo real (VisDrone ou compatível).")
    parser.add_argument("--data", type=Path, required=True, help="Diretório do dataset YOLO (contendo dataset.yaml).")
    parser.add_argument("--pretrained-weights", type=Path, default=None, help="Pesos base YOLO (ex.: yolov8n.pt).")
    parser.add_argument("--weights-out", type=Path, required=True, help="Arquivo de saída para salvar os pesos finais.")
    parser.add_argument("--epochs", type=int, default=0, help="Número de épocas (0 usa valor do TrainConfig).")
    parser.add_argument("--batch", type=int, default=None, help="Tamanho do batch para treinamento.")
    parser.add_argument("--imgsz", type=int, default=None, help="Resolução de entrada (padrão 640).")
    parser.add_argument("--workers", type=int, default=None, help="Número de workers do dataloader.")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo a ser usado (ex.: cuda, cpu).")
    parser.add_argument("--verbose", action="store_true", help="Habilita verbosidade extra.")
    parser.add_argument("--log-every", type=int, default=10, help="Logar a cada N batches (default 10).")
    parser.add_argument("--log-every-seconds", type=int, default=10, help="Logar a cada N segundos (default 10).")
    parser.add_argument("--yolo-save-dir", type=Path, default=None, help="Diretório base para os runs do YOLO.")
    parser.add_argument("--log-dir", type=Path, default=None, help="Diretório para armazenar arquivos de log.")
    parser.add_argument("--seed", type=int, default=None, help="Seed de treinamento.")
    return parser


def _run_cli() -> None:
    args = _build_cli_parser().parse_args()
    base_config = TrainConfig()
    config = TrainConfig(
        epochs=base_config.epochs,
        batch_size=args.batch or base_config.batch_size,
        lr=base_config.lr,
        device=args.device or base_config.device,
        num_workers=args.workers or base_config.num_workers,
        imgsz=args.imgsz or base_config.imgsz,
        seed=args.seed if args.seed is not None else base_config.seed,
        weight_decay=base_config.weight_decay,
        lr_step_size=base_config.lr_step_size,
        lr_gamma=base_config.lr_gamma,
        verbose=args.verbose or base_config.verbose,
        log_every=args.log_every or base_config.log_every,
        log_dir=args.log_dir or base_config.log_dir,
        yolo_save_dir=args.yolo_save_dir or base_config.yolo_save_dir,
        log_every_seconds=args.log_every_seconds or base_config.log_every_seconds,
        pin_memory=base_config.pin_memory,
        persistent_workers=base_config.persistent_workers,
        prefetch_factor=base_config.prefetch_factor,
        drop_last=base_config.drop_last,
    )
    detector = YoloDetector(
        DetectorContext(
            name="YOLOv12n",
            architecture="YOLO",
            recommended_repo="https://github.com/ultralytics/ultralytics",
            target_format="yolo",
        ),
        config=config,
    )
    detector.train(
        dataset_dir=args.data,
        pretrained_weights=args.pretrained_weights,
        weights_out=args.weights_out,
        epochs=args.epochs,
        early_stop=False,
        logger=None,
    )


if __name__ == "__main__":
    _run_cli()
