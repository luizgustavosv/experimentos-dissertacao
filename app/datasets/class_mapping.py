from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch


def build_coco_class_mapping(categories: List[Dict[str, Any]]) -> tuple[int, List[str], Dict[int, int], Dict[int, int]]:
    """Constroi mapeamentos contíguos (0-based) a partir das categorias COCO.

    Retorna:
        K: número de classes (sem background explícito)
        class_names: nomes das classes na ordem interna
        cat_id_to_label: category_id COCO -> label interno [0, K-1]
        label_to_cat_id: inverso do acima
    """

    categories_sorted = sorted(categories or [], key=lambda c: int(c.get("id", 0)))
    class_names: list[str] = []
    cat_id_to_label: dict[int, int] = {}
    label_to_cat_id: dict[int, int] = {}

    for idx, cat in enumerate(categories_sorted):
        cat_id = int(cat["id"])
        cat_id_to_label[cat_id] = idx
        label_to_cat_id[idx] = cat_id
        class_names.append(str(cat.get("name", cat_id)))

    return len(categories_sorted), class_names, cat_id_to_label, label_to_cat_id


def _ensure_int64(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=torch.int64) if tensor.dtype != torch.int64 else tensor


def map_coco_target_to_internal(target: Dict[str, Any], cat_id_to_label: Dict[int, int]) -> Dict[str, Any]:
    mapped = dict(target)

    raw_labels: Iterable[int]
    if "labels_coco" in mapped and torch.is_tensor(mapped["labels_coco"]):
        raw_labels = mapped["labels_coco"].detach().cpu().tolist()
    elif "labels" in mapped and torch.is_tensor(mapped["labels"]):
        raw_labels = mapped["labels"].detach().cpu().tolist()
    else:
        raw_labels = []

    mapped_labels = [cat_id_to_label[int(cid)] for cid in raw_labels]
    mapped["labels_coco"] = torch.tensor(raw_labels, dtype=torch.int64)
    mapped["labels"] = torch.tensor(mapped_labels, dtype=torch.int64)
    return mapped


def map_internal_predictions_to_coco(
    preds: List[Dict[str, Any]], label_to_cat_id: Dict[int, int]
) -> List[Dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for pred in preds:
        if "labels" not in pred:
            continue
        pred_copy = dict(pred)
        labels_tensor = pred_copy.get("labels")
        if torch.is_tensor(labels_tensor):
            labels_list = labels_tensor.detach().cpu().tolist()
        elif isinstance(labels_tensor, Iterable):
            labels_list = list(labels_tensor)
        else:
            labels_list = []
        category_ids = [label_to_cat_id.get(int(lbl)) for lbl in labels_list]
        pred_copy["category_id"] = category_ids
        mapped.append(pred_copy)
    return mapped


def summarize_class_mapping(
    *,
    dataset_name: str | None,
    k: int,
    class_names: List[str],
    categories: List[Dict[str, Any]],
    cat_id_to_label: Dict[int, int],
    label_to_cat_id: Dict[int, int],
    observed_labels: Iterable[int] | None = None,
) -> str:
    observed_list = list(observed_labels) if observed_labels is not None else []
    min_label = min(observed_list) if observed_list else None
    max_label = max(observed_list) if observed_list else None
    unique = sorted(set(observed_list)) if observed_list else []
    return " | ".join(
        [
            "CLASS_MAPPING_SUMMARY",
            f"dataset_name={dataset_name or '<unknown>'}",
            f"K={k}",
            f"class_names={class_names}",
            f"raw_categories={[{'id': c.get('id'), 'name': c.get('name')} for c in categories]}",
            f"cat_id_to_label={cat_id_to_label}",
            f"label_to_cat_id={label_to_cat_id}",
            f"label_min={min_label}",
            f"label_max={max_label}",
            f"label_unique_sample={unique[:10]}",
        ]
    )
