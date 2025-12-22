from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from app.datasets.exporters.coco_exporter import export_coco
from app.datasets.exporters.voc_exporter import export_voc
from app.datasets.exporters.yolo_exporter import export_yolo
from app.datasets.ir import AnnotationRecord, DatasetIR, ImageRecord
from app.datasets.readers.heridal_reader import read_heridal
from app.datasets.readers.visdrone_reader import read_visdrone


def _make_image(path: Path, size=(100, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color="white").save(path)


def test_read_heridal_split_and_annotations(tmp_path: Path) -> None:
    root = tmp_path / "heridal"
    train_dir = root / "train"
    train_dir.mkdir(parents=True)
    image_names = ["img1.jpg", "img2.jpg", "img3.jpg"]
    for name in image_names:
        _make_image(train_dir / name)
    csv_lines = [
        "filename,width,height,class,xmin,ymin,xmax,ymax",
        "img1.jpg,100,100,human,10,10,30,30",
        "img2.jpg,100,100,human,20,20,40,50",
        "img3.jpg,100,100,human,5,5,60,60",
    ]
    (train_dir / "_annotations.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    dataset, discarded, warnings, is_labelled = read_heridal(root, seed=1)

    assert is_labelled is True
    assert discarded["non_human_class"] == 0
    assert warnings == []
    assert dataset.classes == ["human"]
    assert dataset.num_images_per_split()["train"] == 2
    assert dataset.num_images_per_split()["test"] == 1
    assert dataset.num_annotations_per_split()["train"] + dataset.num_annotations_per_split()["test"] == 3


def test_read_visdrone_filters_and_categories(tmp_path: Path) -> None:
    root = tmp_path / "visdrone"
    images_dir = root / "train" / "images"
    annotations_dir = root / "train" / "annotations"
    _make_image(images_dir / "000001.jpg", size=(120, 80))

    lines = [
        "10,10,20,20,1,1,0,0",  # válido
        "0,0,0,10,1,1,0,0",  # largura inválida
        "5,5,10,10,0,1,0,0",  # ignorado por score
        "5,5,10,10,1,3,0,0",  # categoria não humana
    ]
    annotations_dir.mkdir(parents=True, exist_ok=True)
    (annotations_dir / "000001.txt").write_text("\n".join(lines), encoding="utf-8")

    dataset, discarded, warnings, is_labelled = read_visdrone(root, human_categories=[1, 2])

    assert is_labelled is True
    assert discarded["invalid_bbox_size"] == 1
    assert discarded["ignored_by_score"] == 1
    assert discarded["non_human_category"] == 1
    assert dataset.num_annotations_per_split()["train"] == 1
    assert warnings == []


def _build_dataset_ir(tmp_path: Path, split: str = "train") -> DatasetIR:
    image_path = tmp_path / f"{split}_img.jpg"
    _make_image(image_path, size=(100, 100))
    img_record = ImageRecord(
        id=1,
        filename=image_path.name,
        path=image_path,
        width=100,
        height=100,
        split=split,
    )
    ann_record = AnnotationRecord(
        id=1,
        image_id=1,
        class_id=0,
        class_name="human",
        xmin=25,
        ymin=25,
        xmax=75,
        ymax=75,
        area=2500,
    )
    return DatasetIR(classes=["human"], images=[img_record], annotations=[ann_record])


def test_yolo_exporter_outputs_labels_and_yaml(tmp_path: Path) -> None:
    dataset = _build_dataset_ir(tmp_path)
    output_dir = tmp_path / "yolo_out"

    export_yolo(dataset, output_dir, is_labelled=True)

    label_path = output_dir / "labels" / "train" / f"{dataset.images[0].path.stem}.txt"
    yaml_path = output_dir / "dataset.yaml"
    content = label_path.read_text(encoding="utf-8").strip()
    assert content == "0 0.500000 0.500000 0.500000 0.500000"
    yaml_data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert yaml_data["path"] == output_dir.resolve().as_posix()
    assert yaml_data["train"] == "images/train"
    assert "human" in yaml_data["names"].values()


def test_coco_exporter_generates_split_json(tmp_path: Path) -> None:
    dataset = _build_dataset_ir(tmp_path, split="val")
    output_dir = tmp_path / "coco_out"

    export_coco(dataset, output_dir, is_labelled=True)

    ann_path = output_dir / "val.json"
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    assert payload["annotations"][0]["bbox"] == [25, 25, 50, 50]
    assert payload["categories"][0]["id"] == 1


def test_voc_exporter_writes_xml(tmp_path: Path) -> None:
    dataset = _build_dataset_ir(tmp_path)
    output_dir = tmp_path / "voc_out"

    export_voc(dataset, output_dir, is_labelled=True)

    xml_path = output_dir / "Annotations" / f"{dataset.images[0].path.stem}.xml"
    xml_content = xml_path.read_text(encoding="utf-8")
    assert "<xmin>25</xmin>" in xml_content
    split_file = output_dir / "ImageSets" / "Main" / "train.txt"
    assert dataset.images[0].path.stem in split_file.read_text(encoding="utf-8")
