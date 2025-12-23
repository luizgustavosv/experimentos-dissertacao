from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

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

    def infer(self, images_dir: Path, weights_path: Optional[Path], report_out: Path, logger: Optional[Logger] = None):
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
        results = model.predict(
            source=str(images_dir),
            imgsz=640,
            device=device_str,
            save=True,
            project=str(predictions_root),
            name=report_out.stem,
            exist_ok=True,
        )
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

    def validate(self, images_dir: Path, report_out: Path, plots_dir: Path, logger: Optional[Logger] = None):
        raise NotImplementedError("Validação via Ultralytics não implementada neste escopo de treino.")

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
