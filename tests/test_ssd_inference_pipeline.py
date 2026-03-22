from pathlib import Path
import sys
import types

import torch

if "ultralytics" not in sys.modules:
    ultralytics_stub = types.ModuleType("ultralytics")
    ultralytics_stub.YOLO = object
    sys.modules["ultralytics"] = ultralytics_stub

from app.detectors.base import DetectorContext
from app.detectors.torchvision_detectors import TorchvisionDetector
from app.detectors.utils import filter_torchvision_predictions


def _detector() -> TorchvisionDetector:
    context = DetectorContext(
        name="SSD-300",
        architecture="SSD",
        recommended_repo="https://github.com/pytorch/vision",
        target_format="voc",
    )
    return TorchvisionDetector(context=context, build_model=lambda _: torch.nn.Identity())


def test_filter_torchvision_predictions_applies_score_and_class_filters() -> None:
    output = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [2.0, 2.0, 12.0, 12.0], [3.0, 3.0, 8.0, 8.0]]),
        "scores": torch.tensor([0.02, 0.06, 0.9]),
        "labels": torch.tensor([2, 1, 1]),
    }

    filtered, diag = filter_torchvision_predictions(output, score_threshold=0.05, target_label=1)

    assert filtered["boxes"].shape[0] == 2
    assert filtered["labels"].tolist() == [1, 1]
    assert diag["raw_count"] == 3
    assert diag["after_score"] == 2
    assert diag["after_class"] == 2
    assert diag["unique_labels_before"] == [1, 2]
    assert diag["unique_labels_after"] == [1]


def test_ssd_infer_threshold_uses_score_threshold_and_ignores_conf_threshold(monkeypatch) -> None:
    detector = _detector()
    fake_weights = Path("/tmp/fake_ssd_weights.pth")

    monkeypatch.setattr(
        "app.detectors.torchvision_detectors.resolve_ssd_run_config",
        lambda *_args, **_kwargs: {"conf_threshold": 0.001, "score_threshold": 0.07},
    )
    threshold = detector._resolve_ssd_score_threshold(fake_weights, logger=None)
    assert threshold == 0.07

    monkeypatch.setattr(
        "app.detectors.torchvision_detectors.resolve_ssd_run_config",
        lambda *_args, **_kwargs: {"conf_threshold": 0.001},
    )
    threshold = detector._resolve_ssd_score_threshold(fake_weights, logger=None)
    assert threshold == 0.05
