from pathlib import Path

from app.detectors.config import TrainConfig
from app.detectors.utils import resolve_ssd_dataset_classes


def _write_labels(tmp_path: Path, labels: list[str]) -> Path:
    labels_path = tmp_path / "labels.txt"
    labels_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    return labels_path


def test_ssd_resolves_single_class_labels(tmp_path: Path) -> None:
    _write_labels(tmp_path, ["human"])

    class_names, dataset_num_classes, _, _, _ = resolve_ssd_dataset_classes(TrainConfig(), tmp_path)

    assert class_names == ["human"]
    assert dataset_num_classes == 1
    assert dataset_num_classes + 1 == 2


def test_ssd_removes_background_from_labels(tmp_path: Path) -> None:
    _write_labels(tmp_path, ["human", "background"])

    class_names, dataset_num_classes, _, _, _ = resolve_ssd_dataset_classes(TrainConfig(), tmp_path)

    assert class_names == ["human"]
    assert dataset_num_classes == 1
    assert dataset_num_classes + 1 == 2


def test_ssd_visdrone_label_count(tmp_path: Path) -> None:
    labels = [
        "pedestrian",
        "people",
        "bicycle",
        "car",
        "van",
        "truck",
        "tricycle",
        "awning-tricycle",
        "bus",
        "motor",
    ]
    _write_labels(tmp_path, labels)

    class_names, dataset_num_classes, _, _, _ = resolve_ssd_dataset_classes(TrainConfig(), tmp_path)

    assert class_names == labels
    assert dataset_num_classes == 10
    assert dataset_num_classes + 1 == 11
