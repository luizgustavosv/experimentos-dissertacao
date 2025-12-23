from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from app.datasets.exporters import export_coco, export_voc, export_yolo
from app.datasets.ir import DatasetIR
from app.datasets.readers import read_heridal, read_visdrone
from app.datasets.utils import ensure_dir
from app.detectors.base import Logger
from app.reporting.reports import save_normalization_report


TARGET_FORMATS = {
    "YOLO": "yolo",
    "SSD": "voc",
    "Faster R-CNN": "coco",
    "RetinaNet": "coco",
}


@dataclass
class NormalizationResult:
    output_dir: Path
    dataset_type: str
    algorithm_key: str
    target_format: str
    num_images_per_split: Dict[str, int]
    num_annotations_per_split: Dict[str, int]
    discarded_counts: Dict[str, int]
    warnings: List[str]
    is_labelled: bool


def _resolve_target_format(algorithm_key: str, target_format: Optional[str]) -> str:
    if target_format:
        return target_format
    if algorithm_key not in TARGET_FORMATS:
        raise KeyError(f"Algoritmo desconhecido para normalização: {algorithm_key}")
    return TARGET_FORMATS[algorithm_key]


def _timestamped_dir(base: Path, dataset_type: str, algorithm_key: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_algo = algorithm_key.replace(" ", "_").lower()
    return base / f"{dataset_type.lower()}_{safe_algo}_{ts}"


def _log_class_distribution(dataset_ir: DatasetIR, logger: Optional[Logger]) -> None:
    if not logger:
        return
    counts: Dict[int, int] = {idx: 0 for idx in range(len(dataset_ir.classes))}
    for ann in dataset_ir.annotations:
        counts[ann.class_id] = counts.get(ann.class_id, 0) + 1
    logger(
        "Distribuição de classes: "
        + ", ".join(
            f"{idx} ({dataset_ir.classes[idx]}): {count}"
            for idx, count in sorted(counts.items())
            if count > 0
        )
        if any(counts.values())
        else "Distribuição de classes: nenhuma anotação encontrada"
    )


def normalize_dataset(
    dataset_type: str,
    algorithm_key: str,
    dataset_dir: Path,
    normalized_dir: Path,
    split_ratios=(0.8, 0.1, 0.1),
    seed: int = 42,
    logger: Optional[Logger] = None,
    target_format: Optional[str] = None,
) -> NormalizationResult:
    dataset_type = dataset_type.lower()
    dataset_dir = dataset_dir.expanduser().resolve()
    normalized_dir = normalized_dir.expanduser().resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Diretório do dataset não encontrado: {dataset_dir}")

    target_format = _resolve_target_format(algorithm_key, target_format)
    output_dir = _timestamped_dir(normalized_dir, dataset_type, algorithm_key)
    ensure_dir(output_dir)

    if dataset_type == "heridal":
        dataset_ir, discarded, warnings, is_labelled = read_heridal(dataset_dir, split_ratios=split_ratios, seed=seed, logger=logger)
    elif dataset_type == "visdrone":
        dataset_ir, discarded, warnings, is_labelled = read_visdrone(dataset_dir, logger=logger)
    else:
        raise ValueError(f"Tipo de dataset não suportado: {dataset_type}")

    _log_class_distribution(dataset_ir, logger)

    exporter_fn = {"yolo": export_yolo, "voc": export_voc, "coco": export_coco}.get(target_format)
    if exporter_fn is None:
        raise ValueError(f"Formato de exportação não suportado: {target_format}")

    exporter_fn(dataset_ir, output_dir, is_labelled, logger=logger)

    result = NormalizationResult(
        output_dir=output_dir,
        dataset_type=dataset_type,
        algorithm_key=algorithm_key,
        target_format=target_format,
        num_images_per_split=dataset_ir.num_images_per_split(),
        num_annotations_per_split=dataset_ir.num_annotations_per_split(),
        discarded_counts=discarded,
        warnings=warnings,
        is_labelled=is_labelled,
    )
    save_normalization_report(output_dir, result)
    if logger:
        logger(f"[NORM] Dataset salvo em {output_dir}")
        logger(f"[NORM] Relatório salvo em {output_dir / 'normalization_report.json'}")
    return result
