from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from app.detectors.base import Logger


def resolve_device(preferred: Optional[str] = None) -> str:
    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_yolo_dataset(dataset_dir: Path) -> Path:
    dataset_dir = dataset_dir.expanduser().resolve()
    yaml_path = dataset_dir / "dataset.yaml"
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    if not yaml_path.exists():
        raise FileNotFoundError(f"dataset.yaml não encontrado em {dataset_dir}")
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Pastas 'images' e 'labels' são obrigatórias em {dataset_dir}")
    return yaml_path


def validate_coco_dataset(dataset_dir: Path) -> Tuple[Path, Path]:
    dataset_dir = dataset_dir.expanduser().resolve()
    images_root = dataset_dir / "images"
    train_images = images_root / "train"
    val_images = images_root / "val"
    ann_candidates = [
        (dataset_dir / "annotations" / "instances_train.json", dataset_dir / "annotations" / "instances_val.json"),
        (dataset_dir / "train.json", dataset_dir / "val.json"),
    ]
    train_ann, val_ann = next(((train, val) for train, val in ann_candidates if train.exists() and val.exists()), (None, None))
    required_paths = [train_images, val_images]
    if train_ann and val_ann:
        required_paths.extend([train_ann, val_ann])
    missing = [p for p in required_paths if not p or not p.exists()]
    if missing or train_ann is None or val_ann is None:
        if train_ann is None or val_ann is None:
            missing.extend([train_ann or dataset_dir / "annotations" / "instances_train.json", val_ann or dataset_dir / "annotations" / "instances_val.json"])
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Dataset COCO incompleto. Esperado: images/train, images/val e anotações (train/val). "
            f"Faltando: {missing_str}"
        )
    assert train_ann is not None and val_ann is not None
    return train_ann, val_ann


def ensure_weights_size(weights_path: Path, min_bytes: int = 1_000_000) -> None:
    if not weights_path.exists():
        raise FileNotFoundError(f"Arquivo de pesos não foi criado: {weights_path}")
    size = weights_path.stat().st_size
    if size < min_bytes:
        raise ValueError(
            f"Arquivo de pesos muito pequeno ({size} bytes). O treinamento pode não ter sido executado corretamente."
        )


def save_state_dict(model: torch.nn.Module, path: Path) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    ensure_weights_size(path)
    return path


def copy_ultralytics_checkpoint(run_dir: Path, weights_out: Path) -> Path:
    for candidate in ["best.pt", "last.pt"]:
        source = run_dir / candidate
        if source.exists():
            weights_out = weights_out.expanduser().resolve()
            weights_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, weights_out)
            ensure_weights_size(weights_out)
            return weights_out
    raise FileNotFoundError(f"Nenhum checkpoint encontrado em {run_dir}")


def coco_collate(batch: Iterable) -> Tuple[List, List]:
    return tuple(zip(*batch))


def log_config(logger: Optional[Logger], message: str) -> None:
    if logger:
        logger(message)


def describe_dataloader(dataset, logger: Optional[Logger]) -> None:
    if logger:
        logger(f"[DATA] Total de imagens: {len(dataset)}")


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))
