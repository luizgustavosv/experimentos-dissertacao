from __future__ import annotations

import logging
import sys
import types

import torch

if "ultralytics" not in sys.modules:
    ultralytics_stub = types.ModuleType("ultralytics")
    ultralytics_stub.YOLO = object
    sys.modules["ultralytics"] = ultralytics_stub

from app.detectors.torchvision_train import _run_smoke_test_val_loss, run_detection_sanity_check


class _DeviceAwareDetectionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_parameter("weight", torch.nn.Parameter(torch.ones(1)))
        self.moved_before_forward = False

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self.moved_before_forward = True
        return result

    def forward(self, images, targets=None):
        assert self.moved_before_forward, "o modelo deve ser movido antes do primeiro forward"
        assert all(image.device == self.weight.device for image in images)
        if targets is not None:
            return {"loss_classifier": self.weight.sum()}
        return [
            {
                "boxes": torch.empty((0, 4), device=self.weight.device),
                "scores": torch.empty(0, device=self.weight.device),
                "labels": torch.empty(0, dtype=torch.int64, device=self.weight.device),
            }
            for _ in images
        ]


def _loader():
    image = torch.zeros((3, 8, 8), dtype=torch.float32)
    target = {
        "boxes": torch.tensor([[1.0, 1.0, 4.0, 4.0]]),
        "labels": torch.tensor([1], dtype=torch.int64),
    }
    return [([image], [target])]


def test_detection_sanity_check_moves_model_before_forward() -> None:
    model = _DeviceAwareDetectionModel()

    run_detection_sanity_check(
        model,
        _loader(),
        torch.device("cpu"),
        logging.getLogger("test.sanity.device"),
        num_classes=2,
    )

    assert model.moved_before_forward


def test_val_loss_smoke_test_moves_model_before_forward() -> None:
    model = _DeviceAwareDetectionModel()

    _run_smoke_test_val_loss(
        model,
        _loader(),
        torch.device("cpu"),
        logging.getLogger("test.smoke.device"),
        max_images=1,
        num_classes=2,
    )

    assert model.moved_before_forward
