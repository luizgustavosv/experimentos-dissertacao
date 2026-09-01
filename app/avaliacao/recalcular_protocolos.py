from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.avaliacao.config import DEFAULT_CONF_THRESHOLD, DEFAULT_DIAGNOSTIC_MAX_DETECTIONS, DEFAULT_IOU_ASSOCIATION_THRESHOLD
from app.avaliacao.metricas import evaluate_coco_predictions


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_source(result: dict[str, Any], result_path: Path) -> Path | None:
    candidates = [
        result.get("predictions_coco_for_map_ar_curves_json"),
        result.get("prediction_sources", {}).get("map_ar_per_class_and_curves", {}).get("path")
        if isinstance(result.get("prediction_sources"), dict)
        else None,
        result_path.parent / "predictions_coco_for_map_ar_curves.json",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = (result_path.parent / path).resolve()
        if path.exists():
            return path.resolve()
    return None


def _metric_path(metrics: dict[str, Any], key: str) -> Any:
    if isinstance(metrics.get("diagnostic"), dict):
        return metrics["diagnostic"].get(key)
    return metrics.get(key)


def _operating_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "precision_micro",
        "recall_micro",
        "f1_micro",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "confusion_matrix",
    ]
    return {key: metrics.get(key) for key in keys}


def recalculate_result(result_path: Path, output_name: str = "protocol_official_recalc") -> dict[str, Any]:
    result_path = result_path.expanduser().resolve()
    original = _load_json(result_path)
    output_dir = result_path.parent / output_name
    existing_result = output_dir / "results_unified.json"
    if existing_result.exists():
        data = _load_json(existing_result)
        return {
            "status": "already_updated",
            "result_path": str(result_path),
            "updated_results_path": str(existing_result),
            "model": data.get("model"),
            "dataset": data.get("dataset"),
            "split": data.get("split"),
            "official": data.get("metrics", {}).get("official", {}) if isinstance(data.get("metrics"), dict) else {},
            "diagnostic": data.get("metrics", {}).get("diagnostic", {}) if isinstance(data.get("metrics"), dict) else {},
            "saturation": data.get("saturation", {}),
        }
    gt_path = Path(str(original.get("gt_annotations") or "")).expanduser()
    pred_path = _prediction_source(original, result_path)
    if not gt_path.exists() or pred_path is None:
        missing = []
        if not gt_path.exists():
            missing.append("gt_annotations")
        if pred_path is None:
            missing.append("predictions_coco_for_map_ar_curves")
        return {
            "status": "skipped",
            "result_path": str(result_path),
            "reason": f"Artefatos ausentes: {', '.join(missing)}",
        }

    params = original.get("parameters", {}) if isinstance(original.get("parameters"), dict) else {}
    recalculated = evaluate_coco_predictions(
        gt_annotations=gt_path,
        train_annotations=Path(str(original["train_annotations"])) if original.get("train_annotations") else None,
        predictions_json=pred_path,
        output_dir=output_dir,
        model_name=str(original.get("model") or "unknown"),
        dataset_name=str(original.get("dataset") or gt_path),
        split=str(original.get("split") or "val"),
        weights_path=Path(str(original["weights_path"])) if original.get("weights_path") else None,
        conf_threshold=float(params.get("conf_threshold", DEFAULT_CONF_THRESHOLD)),
        iou_threshold=float(params.get("iou_association_threshold", DEFAULT_IOU_ASSOCIATION_THRESHOLD)),
        max_detections=int(params.get("max_detections_per_image", DEFAULT_DIAGNOSTIC_MAX_DETECTIONS)),
        input_size=params.get("input_size"),
        device=params.get("device"),
        epoch_relative=original.get("epoch_relative"),
        epoch_accumulated=original.get("epoch_accumulated"),
        extra={
            "recalculated_from": str(result_path),
            "original_results_preserved": True,
            "original_prediction_source": str(pred_path),
            "original_metrics": original.get("metrics", {}),
        },
    )
    old_operating = _operating_metrics(original.get("metrics", {}) if isinstance(original.get("metrics"), dict) else {})
    new_operating = _operating_metrics(recalculated.get("metrics", {}) if isinstance(recalculated.get("metrics"), dict) else {})
    return {
        "status": "updated",
        "result_path": str(result_path),
        "updated_results_path": recalculated.get("results_path"),
        "model": recalculated.get("model"),
        "dataset": recalculated.get("dataset"),
        "split": recalculated.get("split"),
        "diagnostic_map50_old": _metric_path(original.get("metrics", {}) if isinstance(original.get("metrics"), dict) else {}, "map50"),
        "diagnostic_map50_new": recalculated.get("metrics", {}).get("diagnostic", {}).get("map50"),
        "operating_metrics_identical": old_operating == new_operating,
        "official": recalculated.get("metrics", {}).get("official", {}),
        "diagnostic": recalculated.get("metrics", {}).get("diagnostic", {}),
        "saturation": recalculated.get("saturation", {}),
    }


def recalculate_tree(root: Path, pattern: str | None = None, output_name: str = "protocol_official_recalc") -> dict[str, Any]:
    root = root.expanduser().resolve()
    results = [
        path
        for path in sorted(root.rglob("results_unified.json"))
        if not any(part.startswith("protocol_official_recalc") for part in path.parts)
    ]
    if pattern:
        pattern_lower = pattern.lower()
        results = [path for path in results if pattern_lower in str(path).lower()]
    processed = []
    for path in results:
        try:
            processed.append(recalculate_result(path, output_name=output_name))
        except Exception as exc:  # noqa: BLE001 - relatÃ³rio deve prosseguir nos demais diretÃ³rios
            processed.append({"status": "failed", "result_path": str(path), "reason": str(exc)})
    report = {
        "root": str(root),
        "pattern": pattern,
        "output_name": output_name,
        "processed_count": len(processed),
        "updated_count": sum(1 for item in processed if item.get("status") == "updated"),
        "already_updated_count": sum(1 for item in processed if item.get("status") == "already_updated"),
        "skipped_count": sum(1 for item in processed if item.get("status") == "skipped"),
        "failed_count": sum(1 for item in processed if item.get("status") == "failed"),
        "items": processed,
    }
    report_path = root / f"{output_name}_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalcula results_unified.json com blocos official e diagnostic sem inferÃªncia.")
    parser.add_argument("root", type=Path, help="DiretÃ³rio raiz com resultados de avaliaÃ§Ã£o.")
    parser.add_argument("--pattern", help="Filtra results_unified.json por substring no caminho.")
    parser.add_argument("--output-name", default="protocol_official_recalc", help="SubdiretÃ³rio/relatÃ³rio de saÃ­da.")
    args = parser.parse_args()
    report = recalculate_tree(args.root, pattern=args.pattern, output_name=args.output_name)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
