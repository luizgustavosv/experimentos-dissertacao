from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

from app.detectors.base import DetectionAlgorithm, DetectorContext, Logger
from app.metrics import InferencePerformance, Metrics
from app.reporting.reports import ReportBuilder
from PIL import Image, ImageDraw


class StubDetector(DetectionAlgorithm):
    """
    A lightweight placeholder implementation that simulates training, inference,
    validation, and dataset normalization without requiring heavyweight ML
    dependencies. Each operation logs its steps, generates deterministic metrics,
    and leaves artifacts in the requested output locations.
    """

    def __init__(self, context: DetectorContext):
        super().__init__(context)
        self.report_builder = ReportBuilder(context.name)

    def _simulate_metrics(self, salt: str) -> Metrics:
        random.seed(f"{self.context.name}-{salt}")
        return Metrics.from_values(
            [
                0.65 + random.random() * 0.1,
                0.55 + random.random() * 0.15,
                0.50 + random.random() * 0.2,
                0.45 + random.random() * 0.2,
            ]
        )

    def _simulate_latency(self, image_count: int) -> InferencePerformance:
        rng = random.Random(f"{self.context.name}-latency-{image_count}")
        ms_per_image = rng.uniform(30, 120)
        return InferencePerformance(images_per_second=1000 / ms_per_image, milliseconds_per_image=ms_per_image)

    def _collect_images(self, images_dir: Path) -> List[Path]:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in extensions)

    def _generate_detection_previews(self, image_paths: List[Path], preview_dir: Path, logger: Optional[Logger]) -> List[Path]:
        preview_dir.mkdir(parents=True, exist_ok=True)
        previews: List[Path] = []
        for image_path in image_paths[:6]:
            try:
                with Image.open(image_path) as img:
                    rgb_image = img.convert("RGB")
                    rgb_image.thumbnail((960, 960))
                    draw = ImageDraw.Draw(rgb_image)
                    width, height = rgb_image.size
                    rng = random.Random(f"{self.context.name}-{image_path.name}")
                    for _ in range(rng.randint(1, 4)):
                        box_width = rng.uniform(0.15, 0.35) * width
                        box_height = rng.uniform(0.20, 0.45) * height
                        x0 = rng.uniform(0, width - box_width)
                        y0 = rng.uniform(0, height - box_height)
                        x1, y1 = x0 + box_width, y0 + box_height
                        draw.rectangle([x0, y0, x1, y1], outline="#ef6f6c", width=3)
                        draw.text((x0 + 4, y0 + 4), "Pessoa", fill="#ef6f6c")
                    preview_path = preview_dir / f"{image_path.stem}_detected.jpg"
                    rgb_image.save(preview_path)
                    previews.append(preview_path)
            except Exception as exc:  # noqa: BLE001
                self._log(f"[INFER] Não foi possível gerar miniatura para {image_path.name}: {exc}", logger)
        return previews

    def train(self, dataset_dir: Path, weights_out: Path, logger: Optional[Logger] = None) -> Metrics:
        dataset_dir = dataset_dir.expanduser().resolve()
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset de treino não encontrado em {dataset_dir}")
        weights_out = weights_out.expanduser().resolve()
        weights_out.parent.mkdir(parents=True, exist_ok=True)
        self._log(f"[TRAIN] {self.context.name}: usando dataset em {dataset_dir}", logger)

        metrics = self._simulate_metrics("train")
        weights_out.write_text(
            f"Stub pesos para {self.context.name}\nOrigem do dataset: {dataset_dir}\nMétricas: {metrics.to_dict()}\n"
        )
        report_path = weights_out.with_suffix(".training.pdf")
        self.report_builder.save_report(report_path, metrics, operation="Treinamento", source_dir=dataset_dir)
        self._log(f"[TRAIN] Pesos fictícios salvos em {weights_out}", logger)
        self._log(f"[TRAIN] Relatório salvo em {report_path}", logger)
        return metrics

    def infer(self, images_dir: Path, report_out: Path, logger: Optional[Logger] = None) -> InferencePerformance:
        images_dir = images_dir.expanduser().resolve()
        if not images_dir.exists():
            raise FileNotFoundError(f"Imagens para inferência não encontradas em {images_dir}")
        image_paths = self._collect_images(images_dir)
        if not image_paths:
            raise FileNotFoundError(f"Nenhuma imagem suportada encontrada em {images_dir}")

        report_out = report_out.expanduser().resolve()
        report_out.parent.mkdir(parents=True, exist_ok=True)
        preview_dir = report_out.parent / f"{report_out.stem}_previews"
        self._log(f"[INFER] {self.context.name}: inferência em {images_dir}", logger)

        performance = self._simulate_latency(len(image_paths))
        detection_previews = self._generate_detection_previews(image_paths, preview_dir, logger)
        self.report_builder.save_report(
            report_out,
            metrics=None,
            operation="Inferência",
            source_dir=images_dir,
            inference_performance=performance,
            detection_previews=detection_previews,
        )
        self._log(
            f"[INFER] Latência simulada: {performance.images_per_second:.2f} img/s ({performance.milliseconds_per_image:.2f} ms/imagem)",
            logger,
        )
        self._log(f"[INFER] Relatório salvo em {report_out}", logger)
        return performance

    def validate(self, images_dir: Path, report_out: Path, plots_dir: Path, logger: Optional[Logger] = None) -> Metrics:
        images_dir = images_dir.expanduser().resolve()
        if not images_dir.exists():
            raise FileNotFoundError(f"Imagens de validação não encontradas em {images_dir}")
        report_out = report_out.expanduser().resolve()
        plots_dir = plots_dir.expanduser().resolve()
        plots_dir.mkdir(parents=True, exist_ok=True)
        report_out.parent.mkdir(parents=True, exist_ok=True)
        self._log(f"[VAL] {self.context.name}: validação com imagens em {images_dir}", logger)
        metrics = self._simulate_metrics("validate")
        plot_path = plots_dir / f"{self.context.name.lower()}_val.png"
        self.report_builder.save_plot(plot_path, metrics)
        self.report_builder.save_report(report_out, metrics, operation="Validação", source_dir=images_dir, plot_path=plot_path)
        self._log(f"[VAL] Gráfico salvo em {plot_path}", logger)
        self._log(f"[VAL] Relatório salvo em {report_out}", logger)
        return metrics

    def normalize_dataset(
        self,
        dataset_type: str,
        dataset_dir: Path,
        normalized_dir: Path,
        logger: Optional[Logger] = None,
    ):
        from app.datasets.normalizer import normalize_dataset

        return normalize_dataset(
            dataset_type=dataset_type,
            algorithm_key=self.context.architecture,
            dataset_dir=dataset_dir,
            normalized_dir=normalized_dir,
            logger=logger,
            target_format=self.dataset_target(),
        )
