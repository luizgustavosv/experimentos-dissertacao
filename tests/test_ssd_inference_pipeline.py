from pathlib import Path
import sys
import types

import torch

if "ultralytics" not in sys.modules:
    ultralytics_stub = types.ModuleType("ultralytics")
    ultralytics_stub.YOLO = object
    sys.modules["ultralytics"] = ultralytics_stub

from app.detectors.base import DetectorContext
from app.controller import ExperimentController
from app.detectors.torchvision_detectors import TorchvisionDetector
from app.detectors.utils import filter_torchvision_predictions
from app.metrics import InferencePerformance


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
    assert diag["final_count"] == 2


def test_controller_forwards_ssd_threshold_only_for_ssd() -> None:
    class SSDDummy:
        def __init__(self) -> None:
            self.received_threshold = None
            self.received_benchmark_mode = None

        def infer(self, *_args, ssd_score_threshold=None, benchmark_mode=False, **_kwargs):
            self.received_threshold = ssd_score_threshold
            self.received_benchmark_mode = benchmark_mode
            return InferencePerformance(images_per_second=1.0, milliseconds_per_image=1.0)

    class YOLODummy:
        def __init__(self) -> None:
            self.received_benchmark_mode = None

        def infer(self, *_args, benchmark_mode=False, **_kwargs):
            self.received_benchmark_mode = benchmark_mode
            return InferencePerformance(images_per_second=1.0, milliseconds_per_image=1.0)

    controller = ExperimentController.__new__(ExperimentController)
    ssd_dummy = SSDDummy()
    yolo_dummy = YOLODummy()
    controller.detectors = {"SSD": ssd_dummy, "YOLO": yolo_dummy}

    controller.execute_infer(
        "SSD",
        images_dir=Path("/tmp/images"),
        weights_path=Path("/tmp/weights.pth"),
        report_out=Path("/tmp/report.pdf"),
        ssd_score_threshold=0.33,
        benchmark_mode=True,
    )
    assert ssd_dummy.received_threshold == 0.33
    assert ssd_dummy.received_benchmark_mode is True

    controller.execute_infer(
        "YOLO",
        images_dir=Path("/tmp/images"),
        weights_path=Path("/tmp/weights.pt"),
        report_out=Path("/tmp/report.pdf"),
        ssd_score_threshold=0.88,
        benchmark_mode=False,
    )
    assert yolo_dummy.received_benchmark_mode is False
