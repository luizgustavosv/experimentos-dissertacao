from __future__ import annotations

import os
import re
import time
import torch
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from ultralytics import YOLO

from app.detectors.base import DetectionAlgorithm, DetectorContext, Logger
from app.detectors.config import TrainConfig
from app.detectors.utils import resolve_device
from app.metrics import InferencePerformance, Metrics
from app.training.checkpoint_manager import CheckpointManager, get_latest_last_checkpoint
from app.reporting.reports import ReportBuilder


def _prune_yolo_epoch_checkpoints(weights_dir: Path, keep_epoch: Optional[int], logger: Optional[Logger] = None) -> None:
    """Mantém apenas o checkpoint da última época concluída e remove os demais."""

    if not weights_dir.exists():
        return

    epoch_pattern = re.compile(r"^epoch(\d+)\.pt$")
    epoch_checkpoints: List[tuple[int, Path]] = []
    for path in weights_dir.glob("epoch*.pt"):
        match = epoch_pattern.match(path.name)
        if not match:
            continue
        epoch_checkpoints.append((int(match.group(1)), path))

    if not epoch_checkpoints:
        return

    epoch_checkpoints.sort(key=lambda item: item[0])
    if keep_epoch is None:
        keep_epoch = epoch_checkpoints[-1][0]

    for epoch_idx, ckpt_path in epoch_checkpoints:
        if epoch_idx == keep_epoch:
            continue
        try:
            ckpt_path.unlink()
            if logger:
                logger(f"[YOLO][CHECKPOINT] Removido checkpoint antigo: {ckpt_path.name}")
        except FileNotFoundError:
            continue
        except Exception as exc:  # pragma: no cover - proteção defensiva
            if logger:
                logger(f"[YOLO][CHECKPOINT] Falha ao remover {ckpt_path.name}: {exc}")


def train_yolo(
    dataset_yaml: str,
    pretrained_weights: str,
    output_dir: str,
    config: TrainConfig,
    logger: Optional[Logger] = None,
) -> None:
    dataset_yaml = Path(dataset_yaml)
    pretrained_weights = Path(pretrained_weights)
    output_dir = Path(output_dir)

    if not dataset_yaml.is_file():
        raise FileNotFoundError(dataset_yaml)
    if not pretrained_weights.is_file():
        raise FileNotFoundError(pretrained_weights)

    output_dir.mkdir(parents=True, exist_ok=True)

    planned_epochs = config.epochs
    epochs_to_run = planned_epochs if config.max_epochs is None else min(planned_epochs, config.max_epochs)
    if logger and config.max_epochs is not None:
        logger(f"[YOLO][TRAIN] max_epochs ativo={config.max_epochs} -> epochs_to_run={epochs_to_run}")

    run_dir = output_dir / "yolo_visdrone" / "weights"
    checkpoint_manager = CheckpointManager(
        run_dir=run_dir,
        prefix="yolo",
        ext=".pt",
        keep_best=True,
        metric_name="map",
    )

    resume_last = get_latest_last_checkpoint(run_dir, "yolo", ".pt")
    resume_completed_epoch = 0
    if resume_last is not None:
        match = re.search(r"_last_epoch_(\d+)\.pt$", resume_last.name)
        if match:
            resume_completed_epoch = int(match.group(1))
        pretrained_weights = resume_last
        if logger:
            logger(f"[YOLO][RESUME] Retomando de {resume_last} (epoch concluída={resume_completed_epoch})")

    remaining_epochs = max(0, epochs_to_run - resume_completed_epoch)
    if remaining_epochs <= 0:
        if logger:
            logger("[YOLO][RESUME] Treinamento já concluído segundo o checkpoint last.")
        return

    model = YOLO(str(pretrained_weights))
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")

    def _strip_ansi(value: str) -> str:
        """Remove códigos ANSI de uma string para evitar poluir o CLI."""

        return ansi_escape.sub("", value)

    early_enabled = bool(config.early_stop_enabled)
    patience = int(config.early_stop_patience) if early_enabled else 0
    if logger:
        logger(
            f"[YOLO][TRAIN] algoritmo=YOLO epochs={epochs_to_run} early_stopping_enabled={early_enabled} "
            f"patience={patience} monitor=mAP50-95 mode=max min_delta={config.early_stop_min_delta:.4f} "
            f"min_epochs={config.early_stop_min_epochs}"
        )
        if early_enabled:
            logger(
                f"[YOLO][EARLY] Ultralytics usará patience={patience} "
                "(min_delta/min_epochs/ema_alpha ignorados pelo backend)."
            )

    train_kwargs: Dict[str, object] = {
        "data": str(dataset_yaml),
        "epochs": remaining_epochs,
        "project": str(output_dir),
        "name": "yolo_visdrone",
        "verbose": True,
        "patience": patience,
        "save": True,
        "save_period": 1,
    }

    train_kwargs = {
        key: _strip_ansi(str(value)) if isinstance(value, str) else value
        for key, value in train_kwargs.items()
    }

    cli_cmd = ["yolo", "train"] + [f"{key}={value}" for key, value in train_kwargs.items()]
    cli_cmd = [_strip_ansi(str(part)) for part in cli_cmd]

    if logger:
        logger(f"[YOLO][CLI] Comando final: {cli_cmd}")

    def _on_fit_epoch_end(trainer) -> None:  # pragma: no cover - depende do backend Ultralytics
        save_dir = Path(getattr(trainer, "save_dir", output_dir / "yolo_visdrone"))
        weights_dir = save_dir / "weights"
        current_epoch = int(getattr(trainer, "epoch", -1)) + 1 + resume_completed_epoch

        src_last = weights_dir / "last.pt"
        if src_last.exists():
            payload = torch.load(src_last, map_location="cpu")
            checkpoint_manager.save_last(current_epoch, payload)

        metric_value = None
        for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95", "map"):
            try:
                metric_value = float(getattr(trainer, "metrics", {}).get(key))
                break
            except Exception:
                continue

        src_best = weights_dir / "best.pt"
        if src_best.exists() and metric_value is not None:
            payload_best = torch.load(src_best, map_location="cpu")
            checkpoint_manager.maybe_save_best(current_epoch, metric_value, payload_best)

        checkpoint_manager.cleanup()

    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
    model.add_callback("on_train_end", _on_fit_epoch_end)

    model.train(**train_kwargs)


class YoloDetector(DetectionAlgorithm):
    def __init__(self, context: DetectorContext, config: Optional[TrainConfig] = None) -> None:
        super().__init__(context)
        self.config = config or TrainConfig()

    def train(
        self,
        dataset_yaml: Path,
        pretrained_weights: Optional[Path],
        output_dir: Path,
        epochs: int,
        logger: Optional[Logger] = None,
    ) -> Optional[Metrics]:
        if pretrained_weights is None:
            raise FileNotFoundError("Pesos pré-treinados são obrigatórios para treinar o YOLO.")

        train_config = TrainConfig(**vars(self.config))
        train_config.epochs = epochs or self.config.epochs

        if logger:
            logger("Iniciando treinamento YOLO via API Python da Ultralytics...")

        train_yolo(
            dataset_yaml=str(dataset_yaml),
            pretrained_weights=str(pretrained_weights),
            output_dir=str(output_dir),
            config=train_config,
            logger=logger,
        )

        if logger:
            logger("Treinamento finalizado. Pesos salvos pela Ultralytics no diretório de saída informado.")

        return None

    def infer(
        self,
        images_dir: Path,
        weights_path: Optional[Path],
        report_out: Path,
        pedestrian_only: bool = False,
        logger: Optional[Logger] = None,
    ):
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
