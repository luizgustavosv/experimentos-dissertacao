from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.avaliacao.config import DEFAULT_MAX_DETECTIONS
from app.avaliacao.metricas import evaluate_coco_predictions, assert_official_validation_split, check_train_val_disjoint


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
        max_detections=DEFAULT_MAX_DETECTIONS,
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
        "ar_maxdet",
        "ar_maxdet_5000",
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
    assert result["parameters"]["max_detections_per_image"] == DEFAULT_MAX_DETECTIONS
    assert result["parameters"]["recall_metric_maxdet_key"] == "ar_maxdet"
    assert result["parameters"]["recall_metric_maxdet_value"] == DEFAULT_MAX_DETECTIONS
    assert metrics["ar_maxdet_5000"] == metrics["ar_maxdet"]
    assert result["saturation"]["images_at_detection_cap"] == 0
    assert result["saturation"]["max_observed_detections_per_image"] == 1
    assert result["saturation"]["mean_detections_per_image"] == pytest.approx(1.0)
    assert result["prediction_sources"]["map_ar_per_class_and_curves"]["num_detections"] == 1
    assert result["prediction_sources"]["operating_point_precision_recall_f1_and_confusion_matrix"]["num_detections"] == 1
    for figure in result["figures"].values():
        assert Path(figure["png"]).is_file()
        assert Path(figure["pdf"]).is_file()


def test_visdrone_validation_count_is_asserted(tmp_path: Path) -> None:
    gt = tmp_path / "visdrone_val.json"
    _write_json(gt, _tiny_coco())

    with pytest.raises(AssertionError, match="esperado 548 imagens"):
        assert_official_validation_split(gt, dataset_name="VisDrone")


def test_heridal_validation_count_is_asserted(tmp_path: Path) -> None:
    gt = tmp_path / "heridal_val.json"
    _write_json(gt, _tiny_coco())

    with pytest.raises(AssertionError, match="esperado 310 imagens"):
        assert_official_validation_split(gt, dataset_name="HERIDAL")


def test_map_uses_unfiltered_predictions_but_operating_point_uses_conf_threshold(tmp_path: Path) -> None:
    gt = tmp_path / "gt.json"
    preds = tmp_path / "predictions_coco.json"
    _write_json(gt, _tiny_coco())
    _write_json(preds, [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.1}])

    result = evaluate_coco_predictions(
        gt_annotations=gt,
        predictions_json=preds,
        output_dir=tmp_path / "out_thresholds",
        model_name="unit",
        dataset_name="unit",
        split="val",
        conf_threshold=0.25,
        iou_threshold=0.5,
        max_detections=DEFAULT_MAX_DETECTIONS,
    )

    assert result["metrics"]["map50"] == pytest.approx(1.0)
    assert result["metrics"]["recall_micro"] == pytest.approx(0.0)
    assert result["prediction_sources"]["map_ar_per_class_and_curves"]["num_detections"] == 1
    assert result["prediction_sources"]["operating_point_precision_recall_f1_and_confusion_matrix"]["num_detections"] == 0


def test_train_val_disjoint_uses_file_names_when_image_ids_are_split_local(tmp_path: Path) -> None:
    train = tmp_path / "instances_train.json"
    val = tmp_path / "instances_val.json"
    _write_json(
        train,
        {
            "images": [{"id": 1, "file_name": "images/train/a.jpg", "width": 10, "height": 10}],
            "annotations": [],
            "categories": [{"id": 1, "name": "pedestrian"}],
        },
    )
    _write_json(
        val,
        {
            "images": [{"id": 1, "file_name": "images/val/b.jpg", "width": 10, "height": 10}],
            "annotations": [],
            "categories": [{"id": 1, "name": "pedestrian"}],
        },
    )

    result = check_train_val_disjoint(train, val)

    assert result["disjoint"] is True
    assert result["intersection_count"] == 0
    assert result["image_id_disjoint"] is False
    assert result["image_id_intersection_count"] == 1
