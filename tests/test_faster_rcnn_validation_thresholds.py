from __future__ import annotations

import json
from pathlib import Path

import torch

from app.controller import ExperimentController
from app.detectors.config import TrainConfig
from app.detectors import torchvision_train
from app.detectors.utils import resolve_device


class _FakeFasterRCNNDetector:
    def __init__(self) -> None:
        self.received_kwargs = None

    def validate_trained_weights(self, *args, **kwargs):
        self.received_kwargs = kwargs
        return {"output_dir": str(Path(args[1]).parent)}


class _FakeRetinaNetDetector(_FakeFasterRCNNDetector):
    pass


def _write_coco(path: Path) -> None:
    payload = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "pedestrian"}]}
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
        output_dir=tmp_path / "metrics",
        device="cpu",
        conf_threshold=0.37,
        iou_threshold=0.61,
    )

    assert detector.received_kwargs["output_dir"] == tmp_path / "metrics"
    assert detector.received_kwargs["device"] == "cpu"
    assert detector.received_kwargs["conf_threshold"] == 0.37
    assert detector.received_kwargs["iou_threshold"] == 0.61


def test_faster_rcnn_post_validation_loss_mode_has_default_retinanet_flags(monkeypatch, tmp_path: Path) -> None:
    weights = tmp_path / "last.pth"
    weights.write_bytes(b"weights")
    train_ann = tmp_path / "instances_train.json"
    val_ann = tmp_path / "instances_val.json"
    _write_coco(train_ann)
    _write_coco(val_ann)

    seen = {}

    monkeypatch.setattr(
        torchvision_train,
        "_build_val_loader_and_classes",
        lambda *args, **kwargs: ([], 2, 1, 0),
    )
    monkeypatch.setattr(torchvision_train.torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(torchvision_train, "ensure_weights_size", lambda *args, **kwargs: None)
    monkeypatch.setattr(torchvision_train, "_extract_state_dict", lambda loaded: ({}, "fake"))
    monkeypatch.setattr(
        torchvision_train,
        "_load_frcnn_weights_with_head_guard",
        lambda *args, **kwargs: ([], [], None, False, [], True),
    )

    def fake_loss_loop(*args, **kwargs):
        seen["expects_background"] = kwargs["expects_background"]
        seen["label_offset"] = kwargs["label_offset"]
        return {"loss_per_batch": 0.0, "loss_per_image": 0.0, "breakdown": {}}

    monkeypatch.setattr(torchvision_train, "run_val_loss_loop", fake_loss_loop)

    result = torchvision_train.run_post_training_validation(
        model_builder=lambda _num_classes: torch.nn.Linear(1, 1),
        dataset_dir=tmp_path,
        train_ann=train_ann,
        val_ann=val_ann,
        weights_path=weights,
        config=TrainConfig(val_mode="loss", log_dir=tmp_path / "logs"),
        output_dir=tmp_path / "metrics",
        run_tag="faster_rcnn",
    )

    assert seen == {"expects_background": False, "label_offset": 0}
    assert Path(result["output_dir"]).parent == tmp_path / "metrics"


def test_numeric_device_option_resolves_to_torch_cuda_device() -> None:
    assert resolve_device("0") == "cuda:0"


def test_retinanet_post_validation_receives_mode_thresholds_and_device(tmp_path: Path) -> None:
    images = tmp_path / "images"
    (images / "train").mkdir(parents=True)
    (images / "val").mkdir()
    train_ann = tmp_path / "instances_train.json"
    val_ann = tmp_path / "instances_val.json"
    weights = tmp_path / "last.pth"
    _write_coco(train_ann)
    _write_coco(val_ann)
    weights.write_bytes(b"weights")

    detector = _FakeRetinaNetDetector()
    controller = ExperimentController.__new__(ExperimentController)
    controller.detectors = {"RetinaNet": detector}

    controller.execute_validate_retinanet(
        "RetinaNet",
        train_annotations=train_ann,
        images_dir=images,
        weights_path=weights,
        val_annotations=val_ann,
        val_mode="loss",
        output_dir=tmp_path / "metrics",
        device="cpu",
        conf_threshold=0.42,
        iou_threshold=0.7,
    )

    assert detector.received_kwargs["val_mode"] == "loss"
    assert detector.received_kwargs["output_dir"] == tmp_path / "metrics"
    assert detector.received_kwargs["device"] == "cpu"
    assert detector.received_kwargs["conf_threshold"] == 0.42
    assert detector.received_kwargs["iou_threshold"] == 0.7
