from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional

from app.datasets.ir import AnnotationRecord, DatasetIR, ImageRecord
from app.datasets.utils import copy_image, ensure_dir
from app.detectors.base import Logger


def _create_annotation_xml(img: ImageRecord, anns: Dict[int, AnnotationRecord]) -> ET.Element:
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = img.filename

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img.width)
    ET.SubElement(size, "height").text = str(img.height)
    ET.SubElement(size, "depth").text = "3"

    for ann in anns.values():
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = ann.class_name
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(ann.xmin)
        ET.SubElement(bndbox, "ymin").text = str(ann.ymin)
        ET.SubElement(bndbox, "xmax").text = str(ann.xmax)
        ET.SubElement(bndbox, "ymax").text = str(ann.ymax)
    return annotation


def export_voc(dataset: DatasetIR, output_dir: Path, is_labelled: bool, logger: Optional[Logger] = None) -> Path:
    output_dir = output_dir.expanduser().resolve()
    annotations_dir = ensure_dir(output_dir / "Annotations")
    images_dir = ensure_dir(output_dir / "JPEGImages")
    imagesets_dir = ensure_dir(output_dir / "ImageSets" / "Main")

    ann_by_img = dataset.annotations_by_image()
    split_files: Dict[str, Path] = {}

    for img in dataset.images:
        copy_image(img.path, images_dir / img.filename)
        anns = {ann.id: ann for ann in ann_by_img.get(img.id, [])}
        if is_labelled:
            xml_root = _create_annotation_xml(img, anns)
            tree = ET.ElementTree(xml_root)
            tree.write(annotations_dir / f"{Path(img.filename).stem}.xml", encoding="utf-8")
        split_file = split_files.setdefault(img.split, imagesets_dir / f"{img.split}.txt")
        with split_file.open("a", encoding="utf-8") as fh:
            fh.write(f"{Path(img.filename).stem}\n")
        if logger:
            logger(f"[VOC] Exportação concluída para {img.filename} ({img.split})")

    return output_dir
