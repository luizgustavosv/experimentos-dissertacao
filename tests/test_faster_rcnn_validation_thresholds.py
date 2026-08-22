from __future__ import annotations

import json
from pathlib import Path

from app.controller import ExperimentController


class _FakeFasterRCNNDetector:
    def __init__(self) -> None:
        self.received_kwargs = None

    def validate_trained_weights(self, *args, **kwargs):
        self.received_kwargs = kwargs
        return {"output_dir": str(Path(args[1]).parent)}


def _write_coco(path: Path) -> None:
    payload = {"images": [], "annotations": [], "categories": []}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_faster_rcnn_post_validation_receives_manual_thresholds(tmp_path: Path) -> None:
    images = tmp_path / "images"
    (images / "train").mkdir(parents=True)
    (images / "val").mkdir()
    train_ann = tmp_path / "instances_train.json"
    val_ann = tmp_path / "instances_val.json"
    weights = tmp_path / "last.pth"
    _write_coco(train_ann)
    _write_coco(val_ann)
    weights.write_bytes(b"weights")

    detector = _FakeFasterRCNNDetector()
    controller = ExperimentController.__new__(ExperimentController)
    controller.detectors = {"Faster R-CNN": detector}

    controller.execute_validate_faster_rcnn(
        "Faster R-CNN",
        train_annotations=train_ann,
        images_dir=images,
        weights_path=weights,
        val_annotations=val_ann,
        val_mode="metrics",
        conf_threshold=0.37,
        iou_threshold=0.61,
    )

    assert detector.received_kwargs["conf_threshold"] == 0.37
    assert detector.received_kwargs["iou_threshold"] == 0.61
