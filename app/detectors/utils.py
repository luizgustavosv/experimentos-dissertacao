from __future__ import annotations

import json
import os
import random
import logging
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from torch.utils.data import DataLoader

from app.detectors.base import Logger
from app.detectors.config import TrainConfig

_VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def resolve_device(preferred: Optional[str] = None) -> str:
    if preferred:
        preferred = str(preferred).strip()
        if preferred.isdigit():
            return f"cuda:{preferred}"
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


def validate_voc_dataset(
    dataset_dir: Path,
    *,
    logger: Optional[Logger] = None,
    with_metadata: bool = False,
) -> Tuple[Path, List[str], List[str], List[str]]:
    dataset_dir = dataset_dir.expanduser().resolve()
    candidate_roots = [
        dataset_dir / "VOC2007",
        dataset_dir / "VOCdevkit" / "VOC2007",
        dataset_dir,
    ]
    dataset_root = next((candidate for candidate in candidate_roots if (candidate / "JPEGImages").exists()), None)
    if dataset_root is None:
        raise FileNotFoundError(
            "Estrutura Pascal VOC não encontrada. Certifique-se de apontar para a pasta que contém 'JPEGImages' e 'Annotations'."
        )

    annotations_dir = dataset_root / "Annotations"
    imagesets_dir = dataset_root / "ImageSets" / "Main"
    required_paths = [annotations_dir, imagesets_dir]
    missing = [p for p in required_paths if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Dataset Pascal VOC incompleto. Esperado: Annotations e ImageSets/Main (train.txt e val.txt). "
            f"Faltando: {missing_str}"
        )

    split_files = {"train": imagesets_dir / "train.txt", "val": imagesets_dir / "val.txt"}
    missing_splits = [path for path in split_files.values() if not path.exists()]
    if missing_splits:
        raise FileNotFoundError(
            "Arquivos de split do Pascal VOC ausentes. Esperado pelo menos train.txt e val.txt em ImageSets/Main. "
            f"Faltando: {', '.join(str(p) for p in missing_splits)}"
        )

    train_result = _read_split_file(
        split_files["train"],
        images_dir=dataset_root / "JPEGImages",
        logger=logger,
        return_metadata=with_metadata,
    )
    val_result = _read_split_file(
        split_files["val"],
        images_dir=dataset_root / "JPEGImages",
        logger=logger,
        return_metadata=with_metadata,
    )
    train_ids, train_meta = (train_result if with_metadata else (train_result, []))
    val_ids, val_meta = (val_result if with_metadata else (val_result, []))
    if not train_ids or not val_ids:
        raise ValueError("Arquivos de split do Pascal VOC estão vazios ou inválidos (train.txt / val.txt).")

    class_names = _load_voc_classes(dataset_root)
    if with_metadata:
        return dataset_root, class_names, (train_ids, train_meta), (val_ids, val_meta)
    return dataset_root, class_names, train_ids, val_ids


def atomic_torch_save(obj: Any, path: Path) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)
    return path


def _looks_like_state_dict(obj: Any) -> bool:
    if isinstance(obj, dict):
        if "model" in obj or "state_dict" in obj:
            return True
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return True
    return False


def strip_state_dict_prefixes(state_dict: Dict[str, torch.Tensor], prefixes: Sequence[str]) -> Dict[str, torch.Tensor]:
    filtered: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        filtered[new_key] = value
    return filtered


def extract_checkpoint_state(loaded: Any) -> tuple[Dict[str, torch.Tensor], str]:
    if isinstance(loaded, dict):
        if isinstance(loaded.get("model_state"), dict):
            return loaded["model_state"], "checkpoint.model_state"
        if isinstance(loaded.get("state_dict"), dict):
            return loaded["state_dict"], "checkpoint.state_dict"
        if isinstance(loaded.get("model"), dict):
            return loaded["model"], "checkpoint.model"
        if loaded and all(torch.is_tensor(v) for v in loaded.values()):
            return loaded, "state_dict"
    if isinstance(loaded, dict) and not loaded:
        return {}, "state_dict.empty"
    raise ValueError("Formato de checkpoint não suportado para carregamento de pesos.")


def extract_checkpoint_meta(loaded: Any) -> Dict[str, Any]:
    if isinstance(loaded, dict) and isinstance(loaded.get("meta"), dict):
        return loaded["meta"]
    return {}


def _ssd_priors_per_location() -> Sequence[int]:
    from torchvision.models.detection import ssd300_vgg16

    model = ssd300_vgg16(weights=None, weights_backbone=None, num_classes=91)
    return model.anchor_generator.num_anchors_per_location()


def infer_ssd_num_classes(state_dict: Dict[str, torch.Tensor], logger: Optional[Logger] = None) -> Optional[int]:
    priors = _ssd_priors_per_location()
    for idx, prior in enumerate(priors):
        key = f"head.classification_head.module_list.{idx}.bias"
        if key not in state_dict:
            continue
        bias_len = state_dict[key].numel()
        if bias_len % prior == 0:
            num_classes = bias_len // prior
            if logger:
                logger(f"[SSD][WEIGHTS] Inferido num_classes={num_classes} a partir do state_dict (prior={prior}).")
            return int(num_classes)
    if logger:
        logger("[SSD][WEIGHTS] Falha ao inferir num_classes do state_dict.")
    return None


def load_ssd_weights(
    model: torch.nn.Module,
    weights_path: Path,
    device: torch.device,
    *,
    strict: bool = True,
    loaded: Any = None,
    logger: Optional[Logger] = None,
) -> Dict[str, Any]:
    weights_path = weights_path.expanduser().resolve()
    if loaded is None:
        loaded = torch.load(weights_path, map_location=device)

    top_keys = sorted(list(loaded.keys())) if isinstance(loaded, dict) else []
    if logger and top_keys:
        logger(f"[SSD][WEIGHTS] checkpoint_keys={top_keys}")

    meta = extract_checkpoint_meta(loaded)
    state_dict, checkpoint_format = extract_checkpoint_state(loaded)
    state_dict = strip_state_dict_prefixes(state_dict, ["module.", "model."])

    ckpt_num_classes = infer_ssd_num_classes(state_dict, logger=logger)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))

    if logger:
        logger(
            f"[SSD][WEIGHTS] load_state_dict strict={strict} missing={len(missing)} unexpected={len(unexpected)}"
        )
        if missing:
            logger(f"[SSD][WEIGHTS] missing_keys_sample={missing[:5]}")
        if unexpected:
            logger(f"[SSD][WEIGHTS] unexpected_keys_sample={unexpected[:5]}")

    return {
        "loaded": loaded,
        "meta": meta,
        "state_dict": state_dict,
        "checkpoint_format": checkpoint_format,
        "ckpt_num_classes": ckpt_num_classes,
        "missing": missing,
        "unexpected": unexpected,
        "strict_load": strict,
    }


def filter_torchvision_predictions(
    output: Dict[str, torch.Tensor],
    *,
    score_threshold: float,
    target_label: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    boxes = output.get("boxes", torch.empty((0, 4), dtype=torch.float32))
    scores = output.get("scores", torch.empty((0,), dtype=torch.float32))
    labels = output.get("labels", torch.empty((0,), dtype=torch.int64))

    raw_count = int(boxes.shape[0])
    keep_score = scores >= float(score_threshold)
    boxes = boxes[keep_score]
    scores = scores[keep_score]
    labels = labels[keep_score]
    after_score = int(boxes.shape[0])

    if target_label is not None:
        keep_class = labels == int(target_label)
        boxes = boxes[keep_class]
        scores = scores[keep_class]
        labels = labels[keep_class]
    after_class = int(boxes.shape[0])

    filtered = {
        "boxes": boxes.detach().cpu(),
        "scores": scores.detach().cpu(),
        "labels": labels.detach().cpu(),
    }
    diagnostics = {
        "raw_count": raw_count,
        "after_score": after_score,
        "after_class": after_class,
        "final_count": int(filtered["boxes"].shape[0]),
    }
    return filtered, diagnostics


def ensure_weights_size(
    weights_path: Path | str,
    min_bytes: int = 1_000_000,
    logger: Optional[Logger] = None,
    integrity_check: bool = True,
) -> None:
    resolved_path = Path(weights_path).expanduser().resolve()
    logging_logger = logging.getLogger("weights.integrity")

    def _emit_info(message: str) -> None:
        if logger:
            logger(message)
        else:
            logging_logger.info(message)

    def _emit_warning(message: str) -> None:
        if logger:
            logger(message)
        else:
            logging_logger.warning(message)

    _emit_info(f"[WEIGHTS] Saved weights path={resolved_path}")

    exists = resolved_path.exists()
    is_file = resolved_path.is_file()
    if not exists or not is_file:
        _emit_warning(
            "[WEIGHTS] integrity_check=FAIL reason=missing_or_invalid_path "
            f"path={resolved_path} exists={exists} is_file={is_file}"
        )
        return

    size = 0
    attempts = 0
    last_exception: Optional[Exception] = None
    while attempts < 5:
        attempts += 1
        try:
            size = resolved_path.stat().st_size
        except Exception as exc:  # pragma: no cover - robustez de IO
            last_exception = exc
            size = 0
        if size > 0:
            break
        time.sleep(0.1)

    _emit_info(f"[WEIGHTS] size_bytes={size} min_bytes={min_bytes} attempts={attempts}")

    if size == 0:
        parent = resolved_path.parent
        prefix = resolved_path.stem
        sibling_listing: list[str] = []
        try:
            for candidate in sorted(parent.iterdir()):
                if not candidate.name.startswith(prefix):
                    continue
                try:
                    candidate_size = candidate.stat().st_size
                except Exception:
                    candidate_size = -1
                sibling_listing.append(f"{candidate.name} ({candidate_size} bytes)")
                if len(sibling_listing) >= 10:
                    break
        except Exception:
            sibling_listing = []

        diag = (
            "[WEIGHTS] integrity_check=FAIL reason=size_zero "
            f"path={resolved_path} exists={exists} is_file={is_file} attempts={attempts} "
            f"parent={parent} siblings={sibling_listing}"
        )
        if last_exception:
            diag += f" last_exception={last_exception}"
        _emit_warning(diag)
        return

    if size < min_bytes:
        msg = f"Pesos muito pequenos ({size} bytes). Possível falha ao salvar."
        _emit_warning("[WEIGHTS] integrity_check=FAIL reason=size_below_min")
        raise ValueError(msg)

    if not integrity_check:
        _emit_info("[WEIGHTS] integrity_check=SKIPPED")
        return

    try:
        loaded = torch.load(resolved_path, map_location="cpu")
        if not _looks_like_state_dict(loaded):
            raise ValueError("estrutura inesperada")
    except Exception as exc:  # pragma: no cover - robustez de IO
        _emit_warning(f"[WEIGHTS] integrity_check=FAIL reason={exc}")
        raise ValueError(f"Arquivo de pesos corrompido/incompleto em {resolved_path}: {exc}") from exc

    _emit_info("[WEIGHTS] integrity_check=OK")


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


def ssd_collate_with_diagnostics(batch: Iterable, logger: Optional[Logger] = None) -> Tuple[List, List]:
    batch_list = list(batch)
    diag_logger = logger or logging.getLogger("train.ssd")
    try:
        images, targets = tuple(zip(*batch_list))
    except Exception as exc:  # pragma: no cover - proteção extra
        raise RuntimeError(f"[STAGE=collate] falha ao desempacotar batch: {exc}") from exc

    if diag_logger:
        try:
            diag_logger.debug(
                "[STAGE=collate] batch_size=%d itens=%s",
                len(batch_list),
                [
                    {
                        "shape": getattr(img, "shape", None),
                        "boxes": int(t.get("boxes").shape[0]) if isinstance(t, dict) and "boxes" in t else None,
                        "img_path": t.get("img_path") if isinstance(t, dict) else None,
                    }
                    for img, t in batch_list
                ],
            )
        except Exception:
            diag_logger.debug("[STAGE=collate] falha ao logar shapes")

    return list(images), list(targets)


def log_config(logger: Optional[Logger], message: str) -> None:
    if logger:
        logger(message)


def describe_dataloader(dataset, logger: Optional[Logger], *, label: Optional[str] = None) -> None:
    if logger:
        label_suffix = f" ({label})" if label else ""
        logger(f"[DATA] Total de imagens{label_suffix}: {len(dataset)}")


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_split_file(
    split_file: Path,
    *,
    images_dir: Optional[Path] = None,
    logger: Optional[Logger] = None,
    sample_preview: int = 5,
    return_metadata: bool = False,
) -> List[str]:
    """Lê arquivos de split (train.txt/val.txt) com validação extra.

    Esta função mantém o comportamento padrão quando ``logger`` é ``None`` e
    ``return_metadata`` é ``False`` para não interferir em pipelines que não
    precisam de diagnósticos adicionais (ex.: validação/inferência).
    """

    raw_text = split_file.read_text(encoding="utf-8")
    raw_lines = raw_text.splitlines()
    cleaned: List[str] = []
    metadata: List[dict] = []

    if logger:
        logger(
            f"[STAGE=split_load] arquivo={split_file} encoding=utf-8 linhas_total={len(raw_lines)}"
        )

    for line_no, raw_line in enumerate(raw_lines, start=1):
        normalized = raw_line.replace("\ufeff", "").strip()
        if not normalized:
            if logger and line_no <= sample_preview:
                logger(
                    f"[STAGE=split_load][SKIP] linha_vazia idx={line_no} raw={raw_line!r}"
                )
            continue
        if "," in normalized:
            raise ValueError(
                f"[STAGE=split_load] Linha inesperada com vírgula em {split_file} linha={line_no}: {raw_line!r}"
            )
        resolved_path = None
        if images_dir is not None:
            candidates = [images_dir / f"{normalized}{ext}" for ext in _VALID_IMAGE_EXTENSIONS]
            resolved_path = next((c for c in candidates if c.exists()), candidates[0]) if candidates else None

        entry = {
            "line_no": line_no,
            "raw": raw_line,
            "cleaned": normalized,
            "resolved": str(resolved_path) if resolved_path else None,
        }
        if logger and line_no <= sample_preview:
            logger(
                f"[STAGE=split_load][EX] idx={line_no} raw={raw_line!r} cleaned={normalized!r} resolved={entry['resolved']}"
            )
        cleaned.append(normalized)
        metadata.append(entry)

    if logger:
        logger(
            f"[STAGE=split_load] arquivo={split_file} linhas_validas={len(cleaned)}"
        )

    if return_metadata:
        return cleaned, metadata

    return cleaned


def _load_voc_classes(dataset_root: Path) -> List[str]:
    labels_path = dataset_root / "labels.txt"
    if labels_path.exists():
        classes = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if classes:
            return classes

    annotation_dir = dataset_root / "Annotations"
    found: Set[str] = set()
    for xml_path in annotation_dir.glob("*.xml"):
        tree = ET.parse(xml_path)
        for obj in tree.findall("object"):
            name = obj.findtext("name")
            if name:
                found.add(name.strip())
    classes = sorted(found)
    if not classes:
        raise ValueError(
            "Não foi possível inferir classes do dataset Pascal VOC (labels.txt vazio e anotações sem classes)."
        )
    return classes


def _remove_background_classes(
    class_names: List[str],
) -> tuple[List[str], bool, List[str]]:
    if not class_names:
        return class_names, False, class_names
    background_aliases = {"background", "__background__", "bg"}
    lowered = [name.strip() for name in class_names]
    filtered = [name for name in lowered if name.lower() not in background_aliases]
    return filtered, len(filtered) != len(lowered), lowered


def _read_labels_file(labels_path: Optional[Path]) -> List[str]:
    if not labels_path or not labels_path.exists():
        return []
    return [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _infer_voc_classes_from_annotations(dataset_root: Path) -> List[str]:
    annotation_dir = dataset_root / "Annotations"
    if not annotation_dir.exists():
        return []
    found: Set[str] = set()
    for xml_path in annotation_dir.glob("*.xml"):
        tree = ET.parse(xml_path)
        for obj in tree.findall("object"):
            name = obj.findtext("name")
            if name:
                found.add(name.strip())
    return sorted(found)


def _infer_coco_classes_from_annotations(ann_path: Optional[Path], logging_logger: logging.Logger) -> List[str]:
    if not ann_path:
        return []
    try:
        data = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    except Exception as exc:
        logging_logger.warning("[DATASET] Não foi possível ler %s para inferir classes: %s", ann_path, exc)
        return []

    categories = data.get("categories")
    if not isinstance(categories, list):
        logging_logger.warning("[DATASET] Formato de categorias inválido em %s", ann_path)
        return []

    names = [str(cat.get("name")).strip() for cat in categories if cat.get("name")]
    return [name for name in names if name]


def resolve_ssd_dataset_classes(
    config: TrainConfig,
    dataset_root: Path,
    *,
    train_ann: Optional[Path] = None,
    val_ann: Optional[Path] = None,
    logging_logger: Optional[logging.Logger] = None,
) -> tuple[List[str], int, str, bool, List[str]]:
    """Resolve classes for SSD. dataset_num_classes = foreground; model_num_classes = foreground + background."""
    labels_path = dataset_root / "labels.txt"
    configured_classes = config.dataset_num_classes if config.dataset_num_classes is not None else config.num_classes
    class_names: List[str] = []
    source = ""
    background_removed = False
    background_before: List[str] = []

    if configured_classes is not None:
        labels = _read_labels_file(labels_path)
        labels, removed, before = _remove_background_classes(labels)
        background_removed = background_removed or removed
        background_before = before if removed else background_before
        if labels and len(labels) == configured_classes:
            class_names = labels
        else:
            if labels and len(labels) != configured_classes:
                logging_logger = logging_logger or logging.getLogger("train.ssd")
                logging_logger.warning(
                    "[CLASSES] labels.txt possui %d classes, mas config indica %d; usando placeholders.",
                    len(labels),
                    configured_classes,
                )
            class_names = [f"class_{idx+1}" for idx in range(int(configured_classes))]
        source = "user_override"
    else:
        labels = _read_labels_file(labels_path)
        labels, removed, before = _remove_background_classes(labels)
        background_removed = background_removed or removed
        background_before = before if removed else background_before
        if labels:
            class_names = labels
            source = "labels.txt"
        else:
            voc_classes = _infer_voc_classes_from_annotations(dataset_root)
            if voc_classes:
                class_names, removed, before = _remove_background_classes(voc_classes)
                background_removed = background_removed or removed
                background_before = before if removed else background_before
                source = "voc_xml"
            else:
                ann_path = train_ann or val_ann
                logging_logger = logging_logger or logging.getLogger("train.ssd")
                coco_classes = _infer_coco_classes_from_annotations(ann_path, logging_logger)
                if coco_classes:
                    class_names, removed, before = _remove_background_classes(coco_classes)
                    background_removed = background_removed or removed
                    background_before = before if removed else background_before
                    source = "coco_json"

    if not class_names:
        raise ValueError("Não foi possível inferir classes do dataset; verifique labels.txt/annotations.")

    dataset_num_classes = len(class_names)
    return class_names, dataset_num_classes, source, background_removed, background_before
