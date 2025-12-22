from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        width, height = img.size
    return width, height


def copy_image(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def load_json(path: Path) -> List[int]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def clip_bbox(xmin: int, ymin: int, xmax: int, ymax: int, width: int, height: int) -> Tuple[int, int, int, int]:
    xmin = max(0, min(xmin, width))
    ymin = max(0, min(ymin, height))
    xmax = max(0, min(xmax, width))
    ymax = max(0, min(ymax, height))
    return xmin, ymin, xmax, ymax


def unique_sorted(items: Iterable[str]) -> List[str]:
    return sorted(dict.fromkeys(items))
