from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from app.avaliacao.config import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_CURVE_CONF_THRESHOLD,
    DEFAULT_DIAGNOSTIC_MAX_DETECTIONS,
    DEFAULT_IOU_ASSOCIATION_THRESHOLD,
    DEFAULT_MAX_DETECTIONS,
)
from app.avaliacao.figuras import generate_evaluation_figures

Logger = Optional[Callable[[str], None]]

VISDRONE_VAL_IMAGES = 548
HERIDAL_VAL_IMAGES = 310

REFERENCE_EXPECTATIONS = {
    ("visdrone", "val"): {
        "num_images": VISDRONE_VAL_IMAGES,
        "num_categories": 10,
        "num_annotations": 38759,
        "require_all_declared_categories_present": True,
    },
    ("heridal", "val"): {
        "num_images": HERIDAL_VAL_IMAGES,
        "num_categories": 1,
        "num_annotations": 685,
        "require_all_declared_categories_present": True,
    },
}

METRIC_KEYS = [
    "official",
    "diagnostic",
    "precision_micro",
    "recall_micro",
    "f1_micro",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "confusion_matrix",
]

AP_METRIC_KEYS = ["map50", "map50_95", "map75", "map_small", "map_medium", "map_large"]


def expected_validation_images(dataset_name: str | None, annotations_path: Path | None = None) -> int | None:
    name = (dataset_name or "").lower()
    path_text = str(annotations_path or "").lower()
    if "visdrone" in name or "visdrone" in path_text:
        return VISDRONE_VAL_IMAGES
    if "heridal" in name or "heridal" in path_text:
        return HERIDAL_VAL_IMAGES
    return None


def _dataset_family(dataset_name: str | None, annotations_path: Path | None = None) -> str | None:
    text = f"{dataset_name or ''} {annotations_path or ''}".lower()
    if "visdrone" in text:
        return "visdrone"
    if "heridal" in text:
        return "heridal"
    return None


def _reference_summary(gt_data: dict[str, Any]) -> dict[str, Any]:
    categories = gt_data.get("categories", [])
    annotations = gt_data.get("annotations", [])
    category_names = {int(cat["id"]): str(cat.get("name", cat["id"])) for cat in categories if "id" in cat}
    counts = {int(cat_id): 0 for cat_id in category_names}
    for ann in annotations:
        cat_id = int(ann["category_id"])
        counts[cat_id] = counts.get(cat_id, 0) + 1
    per_class = {
        str(cat_id): {"name": category_names.get(cat_id, str(cat_id)), "num_annotations": int(count)}
        for cat_id, count in sorted(counts.items())
    }
    zero_categories = [
        {"id": int(cat_id), "name": category_names.get(cat_id, str(cat_id))}
        for cat_id, count in sorted(counts.items())
        if count == 0
    ]
    return {
        "num_images": len(gt_data.get("images", [])),
        "num_categories": len(categories),
        "num_reference_instances": len(annotations),
        "per_class": per_class,
        "zero_annotation_categories": zero_categories,
    }


def assert_reference_integrity(
    gt_data: dict[str, Any],
    *,
    dataset_name: str | None,
    split: str,
    annotations_path: Path | None = None,
) -> dict[str, Any]:
    summary = _reference_summary(gt_data)
    family = _dataset_family(dataset_name, annotations_path)
    expectation = REFERENCE_EXPECTATIONS.get((family or "", split.lower().strip()))
    failures: list[str] = []
    if expectation:
        if summary["num_images"] != expectation["num_images"]:
            failures.append(f"imagens={summary['num_images']} esperado={expectation['num_images']}")
        if summary["num_categories"] != expectation["num_categories"]:
            failures.append(f"categorias={summary['num_categories']} esperado={expectation['num_categories']}")
        if summary["num_reference_instances"] != expectation["num_annotations"]:
            failures.append(
                f"instancias={summary['num_reference_instances']} esperado={expectation['num_annotations']}"
            )
        if expectation.get("require_all_declared_categories_present") and summary["zero_annotation_categories"]:
            names = ", ".join(str(item["name"]) for item in summary["zero_annotation_categories"])
            failures.append(f"classes declaradas sem anotacoes: {names}")
    if failures:
        raise AssertionError(
            "ReferÃªncia de validaÃ§Ã£o invÃ¡lida; possÃ­vel perda silenciosa de classes/instÃ¢ncias: "
            + "; ".join(failures)
        )
    return {
        "checked": expectation is not None,
        "passed": True,
        "dataset_family": family,
        "expected": expectation,
        "summary": summary,
    }


def _default_annotation_conversion_balance(gt_data: dict[str, Any]) -> dict[str, Any]:
    annotations = len(gt_data.get("annotations", []))
    return {
        "source_format": "COCO",
        "objects_read": annotations,
        "converted": annotations,
        "discarded": 0,
        "discarded_by_cause": {},
        "clamped_boxes": 0,
        "images": len(gt_data.get("images", [])),
        "note": "ReferÃªncia jÃ¡ fornecida em COCO; sem conversÃ£o intermediÃ¡ria neste avaliador.",
    }


def _reference_consistency_check(
    *,
    output_dir: Path,
    dataset_name: str,
    split: str,
    current_summary: dict[str, Any],
) -> dict[str, Any]:
    family = _dataset_family(dataset_name)
    if family is None or output_dir.parent == output_dir:
        return {"checked": False, "consistent": None, "comparison_count": 0, "mismatches": []}
    mismatches: list[dict[str, Any]] = []
    comparisons = 0
    for result_path in output_dir.parent.rglob("results_unified.json"):
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(data.get("split", "")).lower() != split.lower():
            continue
        if _dataset_family(str(data.get("dataset") or ""), Path(str(data.get("gt_annotations") or ""))) != family:
            continue
        other_summary = data.get("reference_summary")
        if not isinstance(other_summary, dict):
            gt_path = data.get("gt_annotations")
            if gt_path and Path(str(gt_path)).exists():
                try:
                    other_summary = _reference_summary(read_coco_json(Path(str(gt_path))))
                except Exception:
                    other_summary = None
        if not isinstance(other_summary, dict):
            continue
        comparisons += 1
        keys = ("num_images", "num_categories", "num_reference_instances")
        diffs = {key: {"current": current_summary.get(key), "other": other_summary.get(key)} for key in keys if current_summary.get(key) != other_summary.get(key)}
        if diffs:
            mismatches.append({"results_path": str(result_path), "differences": diffs})
    return {
        "checked": comparisons > 0,
        "consistent": len(mismatches) == 0 if comparisons > 0 else None,
        "comparison_count": comparisons,
        "mismatches": mismatches[:20],
    }


def read_coco_json(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    required = {"images", "annotations", "categories"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"COCO JSON inválido em {path}. Chaves ausentes: {sorted(missing)}")
    return data


def assert_official_validation_split(
    val_annotations: Path,
    *,
    dataset_name: str | None = None,
    expected_images: int | None = None,
) -> dict[str, Any]:
    val_data = read_coco_json(val_annotations)
    image_count = len(val_data.get("images", []))
    expected = expected_images
    if expected is None:
        expected = expected_validation_images(dataset_name, val_annotations)
    if expected is not None and image_count != expected:
        raise AssertionError(
            f"Partição de validação inválida: esperado {expected} imagens para {dataset_name or val_annotations}, "
            f"mas foram carregadas {image_count}. Use exclusivamente a divisão oficial de validação."
        )
    return {"expected_images": expected, "actual_images": image_count, "passed": expected is None or image_count == expected}


def check_train_val_disjoint(train_annotations: Path | None, val_annotations: Path) -> dict[str, Any]:
    if train_annotations is None:
        return {
            "checked": False,
            "disjoint": None,
            "intersection_count": None,
            "intersection_sample": [],
            "key": "file_name",
        }
    train_data = read_coco_json(train_annotations)
    val_data = read_coco_json(val_annotations)

    def _image_key(img: dict[str, Any]) -> str | None:
        file_name = img.get("file_name")
        if not file_name:
            return None
        return Path(str(file_name)).name.lower()

    train_keys = {key for img in train_data.get("images", []) if (key := _image_key(img))}
    val_keys = {key for img in val_data.get("images", []) if (key := _image_key(img))}
    intersection = sorted(train_keys & val_keys)

    train_ids = {int(img["id"]) for img in train_data.get("images", []) if "id" in img}
    val_ids = {int(img["id"]) for img in val_data.get("images", []) if "id" in img}
    id_intersection = sorted(train_ids & val_ids)
    return {
        "checked": True,
        "disjoint": len(intersection) == 0,
        "key": "file_name",
        "train_images": len(train_keys),
        "val_images": len(val_keys),
        "intersection_count": len(intersection),
        "intersection_sample": intersection[:20],
        "image_id_disjoint": len(id_intersection) == 0,
        "image_id_intersection_count": len(id_intersection),
        "image_id_intersection_sample": id_intersection[:20],
    }


def xywh_to_xyxy(box: Iterable[float]) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _filter_and_cap_predictions(
    predictions: list[dict[str, Any]],
    *,
    conf_threshold: float,
    max_detections: int,
) -> list[dict[str, Any]]:
    # Ordem metodológica: primeiro aplica o limiar de confiança, depois limita o teto por imagem.
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        if float(pred.get("score", 0.0)) >= conf_threshold:
            by_image[int(pred["image_id"])].append(pred)

    capped: list[dict[str, Any]] = []
    for _, items in by_image.items():
        items_sorted = sorted(items, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        capped.extend(items_sorted[:max_detections])
    return capped


def official_protocol_max_dets(dataset_name: str | None, annotations_path: Path | None = None) -> tuple[list[int], str]:
    family = _dataset_family(dataset_name, annotations_path)
    if family == "visdrone":
        return [1, 10, 100, 500], "VisDrone-DET: AR reportado em maxDets 1, 10, 100 e 500."
    if family == "heridal":
        return [1, 10, 100], "HERIDAL sem protocolo de detecÃ§Ã£o publicado; adotado protocolo COCO [1, 10, 100]."
    return [1, 10, 100], "Dataset sem protocolo especÃ­fico configurado; adotado protocolo COCO [1, 10, 100]."


def _weights_policy(weights_path: Path | None) -> str | None:
    if weights_path is None:
        return None
    name = Path(weights_path).name.lower()
    if name.startswith("best"):
        return "best_checkpoint_by_training_metric"
    if name.startswith("last"):
        return "last_checkpoint_at_epoch_budget"
    if "checkpoint_epoch" in name:
        return "explicit_epoch_checkpoint"
    return "explicit_weights_file"


def _compute_pr_and_confusion(
    gt_data: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    conf_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    cat_ids = [int(cat["id"]) for cat in gt_data.get("categories", []) if "id" in cat]
    cat_names = {int(cat["id"]): str(cat.get("name", cat["id"])) for cat in gt_data.get("categories", []) if "id" in cat}
    cat_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
    background_idx = len(cat_ids)

    gts_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in gt_data.get("annotations", []):
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        gts_by_image[int(ann["image_id"])].append(
            {"category_id": int(ann["category_id"]), "bbox": xywh_to_xyxy(ann["bbox"]), "matched": False}
        )

    preds_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        if float(pred.get("score", 0.0)) < conf_threshold:
            continue
        preds_by_image[int(pred["image_id"])].append(
            {
                "category_id": int(pred["category_id"]),
                "bbox": xywh_to_xyxy(pred["bbox"]),
                "score": float(pred.get("score", 0.0)),
            }
        )

    matrix = np.zeros((len(cat_ids) + 1, len(cat_ids) + 1), dtype=np.int64)
    per_class_counts = {cat_id: {"tp": 0, "fp": 0, "fn": 0} for cat_id in cat_ids}
    tp = fp = fn = 0

    all_image_ids = set(gts_by_image.keys()) | set(preds_by_image.keys())
    for image_id in all_image_ids:
        gts = gts_by_image.get(image_id, [])
        preds = sorted(preds_by_image.get(image_id, []), key=lambda item: item["score"], reverse=True)
        matched_gt: set[int] = set()

        for pred in preds:
            best_idx = None
            best_iou = 0.0
            for idx, gt in enumerate(gts):
                if idx in matched_gt:
                    continue
                iou = box_iou(pred["bbox"], gt["bbox"])
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            pred_cat = pred["category_id"]
            pred_idx = cat_to_idx.get(pred_cat, background_idx)
            if best_idx is not None:
                gt = gts[best_idx]
                matched_gt.add(best_idx)
                gt_idx = cat_to_idx.get(gt["category_id"], background_idx)
                matrix[gt_idx, pred_idx] += 1
                if pred_cat == gt["category_id"]:
                    tp += 1
                    per_class_counts.setdefault(pred_cat, {"tp": 0, "fp": 0, "fn": 0})["tp"] += 1
                else:
                    fp += 1
                    fn += 1
                    per_class_counts.setdefault(pred_cat, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
                    per_class_counts.setdefault(gt["category_id"], {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1
            else:
                matrix[background_idx, pred_idx] += 1
                fp += 1
                per_class_counts.setdefault(pred_cat, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1

        for idx, gt in enumerate(gts):
            if idx in matched_gt:
                continue
            gt_cat = gt["category_id"]
            gt_idx = cat_to_idx.get(gt_cat, background_idx)
            matrix[gt_idx, background_idx] += 1
            fn += 1
            per_class_counts.setdefault(gt_cat, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1

    precision_micro = tp / (tp + fp) if tp + fp else 0.0
    recall_micro = tp / (tp + fn) if tp + fn else 0.0
    f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if precision_micro + recall_micro else 0.0

    precisions = []
    recalls = []
    f1s = []
    for cat_id in cat_ids:
        counts = per_class_counts.get(cat_id, {"tp": 0, "fp": 0, "fn": 0})
        p = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
        r = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=np.float64), where=row_sums != 0)
    labels = [cat_names.get(cat_id, str(cat_id)) for cat_id in cat_ids] + ["background"]
    return {
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "f1_micro": float(f1_micro),
        "precision_macro": float(np.mean(precisions)) if precisions else 0.0,
        "recall_macro": float(np.mean(recalls)) if recalls else 0.0,
        "f1_macro": float(np.mean(f1s)) if f1s else 0.0,
        "confusion_matrix": {
            "labels_true": labels,
            "labels_pred": labels,
            "absolute": matrix.tolist(),
            "normalized_by_true": normalized.tolist(),
        },
    }


def _mean_valid(values: np.ndarray) -> float | None:
    valid = values[values > -1]
    return float(np.mean(valid)) if valid.size else None


def _coco_stats_from_eval(evaluator: Any, max_detections: int) -> dict[str, Any]:
    precision = evaluator.eval.get("precision")
    recall = evaluator.eval.get("recall")
    if precision is None or recall is None:
        return {key: None for key in AP_METRIC_KEYS}

    params = evaluator.params
    max_det_idx = list(params.maxDets).index(max_detections)
    area_labels = list(params.areaRngLbl)
    all_idx = area_labels.index("all")
    small_idx = area_labels.index("small")
    medium_idx = area_labels.index("medium")
    large_idx = area_labels.index("large")
    ious = np.asarray(params.iouThrs)
    iou50_idx = int(np.where(np.isclose(ious, 0.5))[0][0])
    iou75_idx = int(np.where(np.isclose(ious, 0.75))[0][0])

    metrics: dict[str, Any] = {
        "map50_95": _mean_valid(precision[:, :, :, all_idx, max_det_idx]),
        "map50": _mean_valid(precision[iou50_idx, :, :, all_idx, max_det_idx]),
        "map75": _mean_valid(precision[iou75_idx, :, :, all_idx, max_det_idx]),
        "map_small": _mean_valid(precision[:, :, :, small_idx, max_det_idx]),
        "map_medium": _mean_valid(precision[:, :, :, medium_idx, max_det_idx]),
        "map_large": _mean_valid(precision[:, :, :, large_idx, max_det_idx]),
        "ar_small": _mean_valid(recall[:, :, small_idx, max_det_idx]),
        "ar_medium": _mean_valid(recall[:, :, medium_idx, max_det_idx]),
        "ar_large": _mean_valid(recall[:, :, large_idx, max_det_idx]),
    }
    for det_idx, det_limit in enumerate(params.maxDets):
        metrics[f"ar_maxdet_{int(det_limit)}"] = _mean_valid(recall[:, :, all_idx, det_idx])
    return metrics


def _log_coco_summary(logger: Logger, metrics: dict[str, Any], max_detections: int, protocol: str) -> None:
    if not logger:
        return
    for key, value in metrics.items():
        if key == "per_class":
            continue
        logger(f"[EVAL][COCO][{protocol}] {key} (maxDets principal={max_detections}) = {value if value is not None else 'null'}")


def _per_class_from_precision(
    precision_array: Any,
    evaluator: Any,
    categories: list[dict[str, Any]],
) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    if precision_array is None:
        for cat in categories:
            per_class[str(cat.get("id"))] = {"name": cat.get("name"), "ap50": 0.0, "ap50_95": 0.0}
        return per_class
    max_det_idx = len(evaluator.params.maxDets) - 1
    for idx, cat_id in enumerate(evaluator.params.catIds):
        cat = next((c for c in categories if int(c.get("id")) == int(cat_id)), {"name": str(cat_id)})
        cls_all = precision_array[:, :, idx, 0, max_det_idx]
        cls_all = cls_all[cls_all > -1]
        cls_50 = precision_array[0, :, idx, 0, max_det_idx]
        cls_50 = cls_50[cls_50 > -1]
        per_class[str(cat_id)] = {
            "name": cat.get("name"),
            "ap50": float(np.mean(cls_50)) if cls_50.size else None,
            "ap50_95": float(np.mean(cls_all)) if cls_all.size else None,
        }
    return per_class


def _run_coco_metric_block(
    coco_gt: Any,
    predictions_path: Path,
    predictions: list[dict[str, Any]],
    *,
    max_dets: list[int],
    protocol_name: str,
    protocol_description: str,
    categories: list[dict[str, Any]],
    logger: Logger,
) -> dict[str, Any]:
    from pycocotools.cocoeval import COCOeval

    primary_max_det = int(max_dets[-1])
    if predictions:
        coco_dt = coco_gt.loadRes(str(predictions_path))
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.params.maxDets = [int(v) for v in max_dets]
        evaluator.evaluate()
        evaluator.accumulate()
        block = _coco_stats_from_eval(evaluator, primary_max_det)
        block["per_class"] = _per_class_from_precision(evaluator.eval.get("precision"), evaluator, categories)
    else:
        block = {key: 0.0 for key in AP_METRIC_KEYS}
        for det_limit in max_dets:
            block[f"ar_maxdet_{int(det_limit)}"] = 0.0
        block.update({"ar_small": 0.0, "ar_medium": 0.0, "ar_large": 0.0, "per_class": _per_class_from_precision(None, None, categories)})
    block["protocol"] = protocol_name
    block["max_dets"] = [int(v) for v in max_dets]
    block["primary_max_det"] = primary_max_det
    block["description"] = protocol_description
    _log_coco_summary(logger, block, primary_max_det, protocol_name)
    return block


def evaluate_coco_predictions(
    *,
    gt_annotations: Path,
    predictions_json: Path,
    output_dir: Path,
    model_name: str,
    dataset_name: str,
    split: str,
    weights_path: Path | None = None,
    train_annotations: Path | None = None,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_ASSOCIATION_THRESHOLD,
    max_detections: int = DEFAULT_MAX_DETECTIONS,
    input_size: int | None = None,
    device: str | None = None,
    epoch_relative: int | None = None,
    epoch_accumulated: int | None = None,
    logger: Logger = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from pycocotools.coco import COCO

    gt_annotations = Path(gt_annotations).expanduser().resolve()
    predictions_json = Path(predictions_json).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_data = read_coco_json(gt_annotations)
    split_check = assert_official_validation_split(gt_annotations, dataset_name=dataset_name)
    reference_integrity_check = assert_reference_integrity(
        gt_data,
        dataset_name=dataset_name,
        split=split,
        annotations_path=gt_annotations,
    )
    reference_summary = reference_integrity_check["summary"]
    disjoint_check = check_train_val_disjoint(train_annotations, gt_annotations)
    if disjoint_check["checked"] and not disjoint_check["disjoint"]:
        raise AssertionError(
            f"Imagens de treino e validação não são disjuntas por file_name: "
            f"interseção={disjoint_check['intersection_count']}"
        )

    raw_predictions = json.loads(predictions_json.read_text(encoding="utf-8"))
    extra_data = extra or {}
    annotation_conversion_balance = extra_data.get("annotation_conversion_balance")
    if not isinstance(annotation_conversion_balance, dict):
        annotation_conversion_balance = _default_annotation_conversion_balance(gt_data)
    representation = extra_data.get("dataset_representation")
    if not isinstance(representation, dict):
        representation = {
            "source_format": "COCO",
            "reference_annotations": str(gt_annotations),
        }
    reference_consistency_check = _reference_consistency_check(
        output_dir=output_dir,
        dataset_name=dataset_name,
        split=split,
        current_summary=reference_summary,
    )
    predictions_for_integrated_metrics = _filter_and_cap_predictions(
        raw_predictions,
        conf_threshold=DEFAULT_CURVE_CONF_THRESHOLD,
        max_detections=max_detections,
    )
    integrated_predictions_path = output_dir / "predictions_coco_for_map_ar_curves.json"
    integrated_predictions_path.write_text(
        json.dumps(predictions_for_integrated_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    predictions_for_operating_point = _filter_and_cap_predictions(
        raw_predictions,
        conf_threshold=conf_threshold,
        max_detections=max_detections,
    )
    filtered_predictions_path = output_dir / "predictions_coco_filtered.json"
    filtered_predictions_path.write_text(
        json.dumps(predictions_for_operating_point, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    coco_gt = COCO(str(gt_annotations))
    categories = gt_data.get("categories", [])
    official_max_dets, official_description = official_protocol_max_dets(dataset_name, gt_annotations)
    diagnostic_max_dets = [1, 10, int(DEFAULT_DIAGNOSTIC_MAX_DETECTIONS)]
    metrics: dict[str, Any] = {
        "official": _run_coco_metric_block(
            coco_gt,
            integrated_predictions_path,
            predictions_for_integrated_metrics,
            max_dets=official_max_dets,
            protocol_name="official",
            protocol_description=official_description,
            categories=categories,
            logger=logger,
        ),
        "diagnostic": _run_coco_metric_block(
            coco_gt,
            integrated_predictions_path,
            predictions_for_integrated_metrics,
            max_dets=diagnostic_max_dets,
            protocol_name="diagnostic",
            protocol_description="DiagnÃ³stico expandido com maxDets [1, 10, 5000], preservando o procedimento usado anteriormente.",
            categories=categories,
            logger=logger,
        ),
    }
    operating_metrics = _compute_pr_and_confusion(
        gt_data,
        predictions_for_operating_point,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
    metrics.update(operating_metrics)

    by_image_counts: dict[int, int] = defaultdict(int)
    for pred in predictions_for_integrated_metrics:
        by_image_counts[int(pred["image_id"])] += 1
    gt_image_count = len(gt_data.get("images", []))
    max_observed_detections = max(by_image_counts.values(), default=0)
    mean_detections_per_image = len(predictions_for_integrated_metrics) / gt_image_count if gt_image_count else 0.0
    saturation_count = sum(1 for count in by_image_counts.values() if count >= max_detections)
    saturation_alert = saturation_count > 0
    metrics_block = {key: metrics.get(key) for key in METRIC_KEYS}
    figure_metrics = dict(metrics_block)
    if isinstance(metrics.get("official"), dict):
        figure_metrics.update(metrics["official"])
    figures = generate_evaluation_figures(
        gt_data=gt_data,
        predictions_for_curves=predictions_for_integrated_metrics,
        metrics=figure_metrics,
        output_dir=output_dir,
        iou_threshold=DEFAULT_IOU_ASSOCIATION_THRESHOLD,
        max_detections_per_image=int(official_max_dets[-1]),
    )

    result = {
        "schema_version": "unified-detection-eval-v1",
        "model": model_name,
        "dataset": dataset_name,
        "split": split,
        "gt_annotations": str(gt_annotations),
        "train_annotations": str(Path(train_annotations).expanduser().resolve()) if train_annotations else None,
        "predictions_coco_json": str(predictions_json),
        "predictions_coco_for_map_ar_curves_json": str(integrated_predictions_path),
        "predictions_coco_filtered_json": str(filtered_predictions_path),
        "output_dir": str(output_dir),
        "weights_path": str(Path(weights_path).expanduser().resolve()) if weights_path is not None else None,
        "epoch_relative": epoch_relative,
        "epoch_accumulated": epoch_accumulated,
        "parameters": {
            "conf_threshold": float(conf_threshold),
            "curve_conf_threshold": DEFAULT_CURVE_CONF_THRESHOLD,
            "iou_association_threshold": float(iou_threshold),
            "max_detections_per_image": int(max_detections),
            "official_max_dets": official_max_dets,
            "diagnostic_max_dets": diagnostic_max_dets,
            "diagnostic_recall_metric_maxdet_key": f"ar_maxdet_{int(DEFAULT_DIAGNOSTIC_MAX_DETECTIONS)}",
            "diagnostic_recall_metric_maxdet_value": int(DEFAULT_DIAGNOSTIC_MAX_DETECTIONS),
            "weights_policy": _weights_policy(weights_path),
            "device": device,
            "input_size": input_size,
            "class_mapping": "COCO category_id preserved; torchvision internal labels reserve/adjust background before export",
        },
        "prediction_sources": {
            "map_ar_per_class_and_curves": {
                "path": str(integrated_predictions_path),
                "confidence_filter_applied_in_evaluator": DEFAULT_CURVE_CONF_THRESHOLD,
                "expected_export_conf_threshold": DEFAULT_CURVE_CONF_THRESHOLD,
                "max_detections_per_image": int(max_detections),
                "num_detections": len(predictions_for_integrated_metrics),
            },
            "operating_point_precision_recall_f1_and_confusion_matrix": {
                "path": str(filtered_predictions_path),
                "confidence_filter": float(conf_threshold),
                "iou_threshold": float(iou_threshold),
                "max_detections_per_image": int(max_detections),
                "num_detections": len(predictions_for_operating_point),
            },
        },
        "validation_split_check": split_check,
        "reference_integrity_check": reference_integrity_check,
        "reference_summary": reference_summary,
        "annotation_conversion_balance": annotation_conversion_balance,
        "dataset_representation": representation,
        "reference_consistency_check": reference_consistency_check,
        "train_val_disjoint_check": disjoint_check,
        "num_images": len(gt_data.get("images", [])),
        "num_detections": len(predictions_for_integrated_metrics),
        "num_detections_filtered": len(predictions_for_operating_point),
        "num_prediction_categories": len({int(p["category_id"]) for p in predictions_for_integrated_metrics}),
        "saturation_alert": saturation_alert,
        "saturation": {
            "images_at_detection_cap": saturation_count,
            "max_detections_per_image": int(max_detections),
            "max_observed_detections_per_image": int(max_observed_detections),
            "mean_detections_per_image": float(mean_detections_per_image),
            "message": (
                f"Há imagens no teto de detecções; eleve max_detections_per_image acima de {int(max_detections)}."
                if saturation_alert
                else None
            ),
        },
        "metrics": metrics_block,
        "figures": figures,
        "extra": extra_data,
        "created_at": datetime.now().isoformat(),
    }

    results_path = output_dir / "results_unified.json"
    result["results_path"] = str(results_path)
    results_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if logger:
        logger(f"[EVAL][UNIFIED] Resultado salvo em {results_path}")
    return result
