from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.detectors.dataset_voc import PascalVOCDataset


def _write_voc_item(root: Path, image_id: str, objects: list[tuple[str, tuple[int, int, int, int]]]) -> None:
    (root / "JPEGImages").mkdir(parents=True)
    (root / "Annotations").mkdir(parents=True)
    Image.new("RGB", (100, 100), color=(255, 255, 255)).save(root / "JPEGImages" / f"{image_id}.jpg")
    object_xml = "\n".join(
        f"""
  <object>
    <name>{name}</name>
    <bndbox>
      <xmin>{box[0]}</xmin>
      <ymin>{box[1]}</ymin>
      <xmax>{box[2]}</xmax>
      <ymax>{box[3]}</ymax>
    </bndbox>
  </object>"""
        for name, box in objects
    )
    (root / "Annotations" / f"{image_id}.xml").write_text(
        f"""<annotation>
  <filename>{image_id}.jpg</filename>
  <size><width>100</width><height>100</height><depth>3</depth></size>
{object_xml}
</annotation>""",
        encoding="utf-8",
    )


def test_pascal_voc_dataset_preserves_all_configured_classes(tmp_path: Path) -> None:
    _write_voc_item(
        tmp_path,
        "sample",
        [
            ("pedestrian", (1, 2, 20, 30)),
            ("car", (10, 10, 40, 45)),
        ],
    )

    _, target = PascalVOCDataset(
        tmp_path,
        ["sample"],
        {"pedestrian": 1, "car": 4},
    )[0]

    assert target["labels"].tolist() == [1, 4]
    assert target["annotation_balance"]["objects_read"] == 2
    assert target["annotation_balance"]["converted"] == 2
    assert target["annotation_balance"]["discarded"] == 0


def test_pascal_voc_dataset_rejects_unknown_classes(tmp_path: Path) -> None:
    _write_voc_item(tmp_path, "sample", [("ignored_class", (1, 2, 20, 30))])

    dataset = PascalVOCDataset(tmp_path, ["sample"], {"pedestrian": 1})

    with pytest.raises(RuntimeError, match="classe desconhecida"):
        dataset[0]
