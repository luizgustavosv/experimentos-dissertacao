from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from app.detectors.base import DetectionAlgorithm, DetectorContext, Logger
from app.metrics import Metrics
from app.reporting.reports import ReportBuilder


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

    def infer(self, images_dir: Path, report_out: Path, logger: Optional[Logger] = None) -> Metrics:
        images_dir = images_dir.expanduser().resolve()
        if not images_dir.exists():
            raise FileNotFoundError(f"Imagens para inferência não encontradas em {images_dir}")
        report_out = report_out.expanduser().resolve()
        report_out.parent.mkdir(parents=True, exist_ok=True)
        self._log(f"[INFER] {self.context.name}: inferência em {images_dir}", logger)
        metrics = self._simulate_metrics("infer")
        self.report_builder.save_report(report_out, metrics, operation="Inferência", source_dir=images_dir)
        self._log(f"[INFER] Relatório salvo em {report_out}", logger)
        return metrics

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

    def normalize_dataset(self, dataset_dir: Path, normalized_dir: Path, logger: Optional[Logger] = None) -> None:
        dataset_dir = dataset_dir.expanduser().resolve()
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset bruto não encontrado em {dataset_dir}")
        normalized_dir = normalized_dir.expanduser().resolve()
        normalized_dir.mkdir(parents=True, exist_ok=True)
        info_path = normalized_dir / "normalization_summary.json"
        summary = {
            "algorithm": self.context.name,
            "source_dataset": str(dataset_dir),
            "normalized_dataset": str(normalized_dir),
            "notes": "Normalização simulada para preparação de treinos consistentes.",
        }
        info_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        self._log(
            f"[NORM] Dataset normalizado (stub) em {normalized_dir}. Detalhes em {info_path}",
            logger,
        )
