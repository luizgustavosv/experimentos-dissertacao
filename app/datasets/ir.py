from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence


@dataclass
class ImageRecord:
    id: int
    filename: str
    path: Path
    width: int
    height: int
    split: str


@dataclass
class AnnotationRecord:
    id: int
    image_id: int
    class_id: int
    class_name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    area: int


@dataclass
class DatasetIR:
    classes: List[str]
    images: List[ImageRecord] = field(default_factory=list)
    annotations: List[AnnotationRecord] = field(default_factory=list)

    def images_by_split(self) -> Dict[str, List[ImageRecord]]:
        groups: Dict[str, List[ImageRecord]] = {}
        for img in self.images:
            groups.setdefault(img.split, []).append(img)
        return groups

    def annotations_by_image(self) -> Dict[int, List[AnnotationRecord]]:
        groups: Dict[int, List[AnnotationRecord]] = {}
        for ann in self.annotations:
            groups.setdefault(ann.image_id, []).append(ann)
        return groups

    def annotations_by_split(self) -> Dict[str, List[AnnotationRecord]]:
        ann_by_img = self.annotations_by_image()
        groups: Dict[str, List[AnnotationRecord]] = {}
        for img in self.images:
            anns = ann_by_img.get(img.id, [])
            if not anns:
                continue
            groups.setdefault(img.split, []).extend(anns)
        return groups

    def num_images_per_split(self) -> Dict[str, int]:
        return {split: len(imgs) for split, imgs in self.images_by_split().items()}

    def num_annotations_per_split(self) -> Dict[str, int]:
        return {split: len(anns) for split, anns in self.annotations_by_split().items()}

    def get_class_name(self, class_id: int) -> str:
        return self.classes[class_id]

    def get_class_id(self, class_name: str) -> int:
        return self.classes.index(class_name)

    def add_image(self, record: ImageRecord) -> None:
        self.images.append(record)

    def add_annotation(self, record: AnnotationRecord) -> None:
        self.annotations.append(record)

    def sorted_images(self) -> Sequence[ImageRecord]:
        return sorted(self.images, key=lambda img: img.id)
