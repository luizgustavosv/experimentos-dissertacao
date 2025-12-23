from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import yaml

from app.detectors.base import DetectionAlgorithm, DetectorContext, Logger
from app.detectors.config import TrainConfig
from app.detectors.utils import copy_ultralytics_checkpoint, resolve_device, seed_everything, validate_yolo_dataset
from app.metrics import InferencePerformance, Metrics
from app.reporting.reports import ReportBuilder


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
        from ultralytics import YOLO  # import tardio para evitar dependências pesadas em import
        yaml_path = validate_yolo_dataset(dataset_dir).resolve()
        seed_everything(self.config.seed)
        device_str = resolve_device(self.config.device)
        num_epochs = epochs or self.config.epochs
        if logger:
            logger(f"[TRAIN] {self.context.name} iniciando treino com {num_epochs} épocas em {device_str}")
            logger(f"YOLO train(): usando dataset.yaml = {yaml_path}")

        base_weights = pretrained_weights or "yolov8n.pt"
        model = YOLO(base_weights)
        results = model.train(
            data=str(yaml_path),
            epochs=num_epochs,
            imgsz=640,
            batch=self.config.batch_size,
            device=device_str,
            workers=self.config.num_workers,
            lr0=self.config.lr,
            seed=self.config.seed,
        )

        run_dir = Path(results.save_dir)
        weights_path = copy_ultralytics_checkpoint(run_dir, weights_out)
        if logger:
            logger(f"[TRAIN] Checkpoint copiado de {run_dir} para {weights_path}")

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
        dataset_path: Path,
        weights_path: Optional[Path],
        report_out: Path,
        plots_dir: Path,
        pedestrian_only: bool = False,
        logger: Optional[Logger] = None,
    ):
        from ultralytics import YOLO  # import tardio para evitar dependências pesadas em import

        report_out = report_out.expanduser().resolve()
        plots_dir = plots_dir.expanduser().resolve()

        dataset_yaml_path = self._resolve_dataset_yaml_path(dataset_path).resolve()
        cfg = yaml.safe_load(dataset_yaml_path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError(f"dataset.yaml inválido (esperado objeto mapeável): {dataset_yaml_path}")

        try:
            raw_root = cfg["path"]
            if isinstance(raw_root, str):
                raw_root = raw_root.strip()
            root = Path(raw_root).expanduser()
            if not root.is_absolute():
                root = (dataset_yaml_path.parent / root).resolve()
            else:
                root = root.resolve()
            train_dir = (root / "images" / "train").resolve()
            val_dir = (root / "images" / "val").resolve()
        except KeyError as exc:  # noqa: PERF203
            raise ValueError(f"Chave obrigatória ausente em {dataset_yaml_path}: {exc}") from exc

        test_dir = (root / cfg["test"]).expanduser().resolve() if "test" in cfg else None
        dataset_yaml_resolved = dataset_yaml_path
        cwd_before = Path.cwd()
        if logger:
            logger(f"[VAL] dataset.yaml informado: {dataset_yaml_resolved}")
            logger(f"[VAL] cwd atual: {cwd_before}")
            logger(f"[VAL] cfg.path={cfg.get('path')} | train={cfg.get('train')} | val={cfg.get('val')}")
            logger(f"[VAL] Diretório raiz resolvido: {root}")
            logger(f"[VAL] train_dir: {train_dir}")
            logger(f"[VAL] val_dir: {val_dir}")
            if test_dir:
                logger(f"[VAL] test_dir: {test_dir}")

        images_root = root / "images"
        if logger:
            logger(f"train_dir repr={train_dir!r}")
            logger(f"val_dir repr={val_dir!r}")
            logger(f"root exists={root.exists()} images={images_root.exists()}")
            if images_root.exists():
                subdirs = [p.resolve() for p in images_root.iterdir() if p.is_dir()]
                logger(f"root/images subdirs: {[str(p) for p in subdirs]}")

        missing = []
        for label, path in (("train", train_dir), ("val", val_dir)):
            if not path.exists():
                missing.append((label, path))
                if logger:
                    logger(f"[VAL] Caminho ausente para {label}_dir: {path}")

        if missing:
            images_root = root / "images"
            listing = ""
            if images_root.exists() and images_root.is_dir():
                entries = sorted(images_root.iterdir())
                listing_lines = [str(entry.resolve()) for entry in entries]
                listing = "\n".join(listing_lines)
            missing_str = "; ".join(f"{label}={path}" for label, path in missing)
            debug_message = (
                "dataset.yaml inválido para validação.\n"
                f"YAML usado: {dataset_yaml_resolved}\n"
                f"root calculado: {root}\n"
                f"train_dir: {train_dir}\n"
                f"val_dir: {val_dir}\n"
                f"Ausentes: {missing_str}"
            )
            if listing:
                debug_message += f"\nConteúdo de {images_root.resolve()}:\n{listing}"
            raise FileNotFoundError(debug_message)
        if logger:
            logger(f"[VAL] Diretórios encontrados (train/val): {train_dir} | {val_dir}")

        weights_resolved = weights_path.expanduser().resolve() if weights_path else None
        if weights_resolved is not None and not weights_resolved.exists():
            raise FileNotFoundError(f"Pesos não encontrados: {weights_resolved}")

        base_weights = str(weights_resolved) if weights_resolved else "yolov8n.pt"
        device_str = resolve_device(self.config.device)
        if logger:
            weight_label = weights_resolved if weights_resolved else "pesos padrão (yolov8n.pt)"
            logger(f"[VAL] {self.context.name} validando {dataset_yaml_resolved} em {device_str} usando {weight_label}")
            if pedestrian_only:
                logger("[VAL] Filtrando apenas classe pedestrian (0) durante a validação")

        model = YOLO(base_weights)
        val_name = report_out.stem
        results = model.val(
            data=str(dataset_yaml_resolved),
            imgsz=640,
            device=device_str,
            project=str(plots_dir),
            name=val_name,
            exist_ok=True,
            verbose=False,
            plots=True,
            classes=[0] if pedestrian_only else None,
        )
        if logger:
            logger(f"[VAL] Validação concluída para {dataset_yaml_resolved}")

        run_dir = Path(getattr(results, "save_dir", plots_dir / val_name))
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics = self._build_metrics_from_results(results, device_str, weights_resolved or Path("yolov8n.pt"))

        report_builder = ReportBuilder(self.context.name)
        metrics_plot = run_dir / "metrics_summary.png"
        report_builder.save_plot(metrics_plot, metrics)
        report_builder.save_report(
            report_path=report_out,
            metrics=metrics,
            operation="Validação",
            source_dir=val_dir,
            plot_path=metrics_plot,
            weights_path=weights_resolved or Path("yolov8n.pt"),
        )

        if logger:
            logger(
                f"[VAL] Precisão: {metrics.precision:.3f} | Recall: {metrics.recall:.3f} | mAP@0.50: {metrics.map50:.3f} | mAP@0.50:0.95: {metrics.map50_95:.3f}"
            )
        if logger:
            logger(f"[VAL] Gráficos salvos em {run_dir}")
            logger(f"[VAL] Relatório salvo em {report_out}")
            logger(f"[VAL] cwd final: {Path.cwd()}")

        return metrics

    @staticmethod
    def _resolve_dataset_yaml_path(dataset_input: Path) -> Path:
        dataset_input = dataset_input.expanduser()
        if dataset_input.suffix.lower() in {".yaml", ".yml"}:
            yaml_path = dataset_input
        else:
            yaml_path = dataset_input / "dataset.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"dataset.yaml não encontrado. Caminho informado: {dataset_input.resolve()} | Esperado: {yaml_path.resolve()}"
            )
        return yaml_path.resolve()

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
