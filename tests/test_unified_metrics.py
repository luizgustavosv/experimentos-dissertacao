from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.avaliacao.metricas import evaluate_coco_predictions, assert_official_validation_split


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _tiny_coco() -> dict:
    return {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "pedestrian"}],
    }


def test_unified_metrics_schema_and_default_coco_iou_range(tmp_path: Path) -> None:
    gt = tmp_path / "gt.json"
    preds = tmp_path / "predictions_coco.json"
    _write_json(gt, _tiny_coco())
    _write_json(preds, [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}])

    result = evaluate_coco_predictions(
        gt_annotations=gt,
        predictions_json=preds,
        output_dir=tmp_path / "out",
        model_name="unit",
        dataset_name="unit",
        split="val",
        conf_threshold=0.25,
        iou_threshold=0.5,
        max_detections=300,
    )

    metrics = result["metrics"]
    assert set(metrics) == {
        "map50",
        "map50_95",
        "map75",
        "map_small",
        "map_medium",
        "map_large",
        "ar1",
        "ar10",
        "ar100",
        "ar_small",
        "ar_medium",
        "ar_large",
        "precision_micro",
        "recall_micro",
        "f1_micro",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "per_class",
        "confusion_matrix",
    }
    assert metrics["map50"] == pytest.approx(1.0)
    assert metrics["map75"] == pytest.approx(1.0)
    assert metrics["map50_95"] == pytest.approx(1.0)
    assert result["parameters"]["max_detections_per_image"] == 300


def test_visdrone_validation_count_is_asserted(tmp_path: Path) -> None:
    gt = tmp_path / "visdrone_val.json"
    _write_json(gt, _tiny_coco())

    with pytest.raises(AssertionError, match="esperado 548 imagens"):
        assert_official_validation_split(gt, dataset_name="VisDrone")
