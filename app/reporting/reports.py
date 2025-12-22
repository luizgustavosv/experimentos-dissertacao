from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from app.metrics import InferencePerformance, Metrics

if TYPE_CHECKING:
    from app.datasets.normalizer import NormalizationResult


@dataclass
class ReportContext:
    algorithm_name: str
    operation: str
    source_dir: Path
    plot_path: Optional[Path] = None


class ReportBuilder:
    def __init__(self, algorithm_name: str) -> None:
        self.algorithm_name = algorithm_name

    def save_plot(self, plot_path: Path, metrics: Metrics) -> Path:
        metrics_pct = metrics.as_percentage()
        plot_path = plot_path.expanduser().resolve()
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = [metrics_pct.precision, metrics_pct.recall, metrics_pct.map50, metrics_pct.map50_95]
        labels = ["Precisão", "Recall", "mAP@0.50", "mAP@0.50:0.95"]
        ax.bar(labels, bars, color=["#1b6ca8", "#17a398", "#ef6f6c", "#f1a208"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentual (%)")
        ax.set_title(f"{self.algorithm_name} — Desempenho")
        for idx, value in enumerate(bars):
            ax.text(idx, value + 1, f"{value:.1f}%", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        return plot_path

    def save_report(
        self,
        report_path: Path,
        metrics: Optional[Metrics],
        operation: str,
        source_dir: Path,
        plot_path: Optional[Path] = None,
        inference_performance: Optional[InferencePerformance] = None,
        detection_previews: Optional[Sequence[Path]] = None,
    ) -> Path:
        report_path = report_path.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(report_path) as pdf:
            fig, ax = plt.subplots(figsize=(8.3, 5.8))
            ax.axis("off")
            lines = [
                f"Relatório de {operation}",
                f"Algoritmo: {self.algorithm_name}",
                f"Origem das imagens: {source_dir}",
            ]
            if inference_performance:
                lines.extend(
                    [
                        "",
                        "Latência:",
                        f" • {inference_performance.images_per_second:.2f} imagens por segundo",
                        f" • {inference_performance.milliseconds_per_image:.2f} ms por imagem",
                    ]
                )
            elif metrics:
                lines.extend(
                    [
                        "",
                        "Métricas:",
                        f" • Precisão: {metrics.precision:.3f}",
                        f" • Recall: {metrics.recall:.3f}",
                        f" • mAP@0.50: {metrics.map50:.3f}",
                        f" • mAP@0.50:0.95: {metrics.map50_95:.3f}",
                    ]
                )
            else:
                lines.extend(
                    [
                        "",
                        "Métricas:",
                        " • Nenhuma métrica calculada para esta operação.",
                    ]
                )
            text = "\n".join(lines)
            ax.text(0.05, 0.95, text, va="top", ha="left", fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)

            if plot_path and plot_path.exists() and metrics:
                plot_fig = plt.figure(figsize=(8.3, 5.8))
                img = plt.imread(plot_path)
                plt.imshow(img)
                plt.axis("off")
                pdf.savefig(plot_fig)
                plt.close(plot_fig)

            if detection_previews:
                for preview in detection_previews:
                    if not preview.exists():
                        continue
                    preview_fig = plt.figure(figsize=(8.3, 5.8))
                    img = plt.imread(preview)
                    plt.imshow(img)
                    plt.axis("off")
                    plt.title(preview.name)
                    pdf.savefig(preview_fig)
                    plt.close(preview_fig)

            detail_fig, detail_ax = plt.subplots(figsize=(8.3, 5.8))
            detail_ax.axis("off")
            detail_ax.text(
                0.05,
                0.95,
                "Este relatório foi gerado automaticamente para apoiar experimentos controlados\n"
                "de detecção de pessoas em imagens aéreas. Substitua as métricas ou latências pelo resultado\n"
                "real assim que treinar e avaliar o modelo. As bounding boxes exibidas são simuladas para fins de exemplo.",
                va="top",
                ha="left",
            )
            pdf.savefig(detail_fig)
        plt.close(detail_fig)
        return report_path


def save_normalization_report(output_dir: Path, result: "NormalizationResult") -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "normalization_report.json"

    def _convert(obj):
        if isinstance(obj, Path):
            return str(obj)
        return obj

    payload = {k: _convert(v) for k, v in result.__dict__.items()}
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report_path
