# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FIGURE_NAMES = [
    "confusion_matrix",
    "confusion_matrix_normalized",
    "BoxP_curve",
    "BoxR_curve",
    "BoxF1_curve",
    "BoxPR_curve",
]


def _xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def _iou(a: list[float], b: list[float]) -> float:
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


def _save(fig: Any, output_dir: Path, stem: str) -> dict[str, str]:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return {"png": str(png), "pdf": str(pdf)}


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["DejaVu Sans"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    return plt


def _colors(count: int) -> list[Any]:
    plt = _load_pyplot()
    cmap = plt.get_cmap("tab10" if count <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(count)]


def _empty_figure(output_dir: Path, stem: str, title: str, message: str) -> dict[str, str]:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(5.9, 4.2))
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10, wrap=True)
    paths = _save(fig, output_dir, stem)
    plt.close(fig)
    return paths


def _match_predictions(
    gt_data: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    category_id: int,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    gts_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in gt_data.get("annotations", []):
        if int(ann.get("iscrowd", 0)) == 1 or int(ann.get("category_id")) != int(category_id):
            continue
        gts_by_image[int(ann["image_id"])].append({"bbox": _xywh_to_xyxy(ann["bbox"]), "matched": False})

    preds = [
        {
            "image_id": int(pred["image_id"]),
            "bbox": _xywh_to_xyxy(pred["bbox"]),
            "score": float(pred.get("score", 0.0)),
        }
        for pred in predictions
        if int(pred.get("category_id", -1)) == int(category_id)
    ]
    preds.sort(key=lambda item: item["score"], reverse=True)

    scores: list[float] = []
    tp: list[int] = []
    for pred in preds:
        best_idx = None
        best_iou = 0.0
        for idx, gt in enumerate(gts_by_image.get(pred["image_id"], [])):
            if gt["matched"]:
                continue
            iou = _iou(pred["bbox"], gt["bbox"])
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_idx = idx
        scores.append(pred["score"])
        if best_idx is None:
            tp.append(0)
        else:
            gts_by_image[pred["image_id"]][best_idx]["matched"] = True
            tp.append(1)

    return np.asarray(scores, dtype=np.float64), np.asarray(tp, dtype=np.int64), sum(len(v) for v in gts_by_image.values())


def _curve_values(
    scores: np.ndarray,
    tp: np.ndarray,
    total_gt: int,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision = np.zeros_like(thresholds, dtype=np.float64)
    recall = np.zeros_like(thresholds, dtype=np.float64)
    f1 = np.zeros_like(thresholds, dtype=np.float64)
    if scores.size == 0:
        return precision, recall, f1

    fp = 1 - tp
    for idx, threshold in enumerate(thresholds):
        keep = scores >= threshold
        if not np.any(keep):
            continue
        tp_count = int(tp[keep].sum())
        fp_count = int(fp[keep].sum())
        precision[idx] = tp_count / (tp_count + fp_count) if tp_count + fp_count else 0.0
        recall[idx] = tp_count / total_gt if total_gt else 0.0
        if precision[idx] + recall[idx]:
            f1[idx] = 2 * precision[idx] * recall[idx] / (precision[idx] + recall[idx])
    return precision, recall, f1


def _plot_matrix(
    matrix: list[list[float]],
    labels: list[str],
    *,
    title: str,
    normalized: bool,
    output_dir: Path,
    stem: str,
) -> dict[str, str]:
    plt = _load_pyplot()
    values = np.asarray(matrix, dtype=np.float64)
    fig_size = max(5.9, 0.36 * len(labels) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.86))
    if normalized:
        color_max = 1.0
    elif values.ndim == 2 and values.shape[0] > 1:
        color_max = max(1.0, float(values[:-1, :].sum()))
    else:
        color_max = max(1.0, float(values.sum()))
    im = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=color_max)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Classe predita", fontsize=9)
    ax.set_ylabel("Classe verdadeira", fontsize=9)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    threshold = float(values.max()) * 0.55 if values.size else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = f"{values[i, j]:.2f}" if normalized else f"{int(values[i, j])}"
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color="white" if values[i, j] > threshold else "black")
    paths = _save(fig, output_dir, stem)
    plt.close(fig)
    return paths


def generate_evaluation_figures(
    *,
    gt_data: dict[str, Any],
    predictions_for_curves: list[dict[str, Any]],
    metrics: dict[str, Any],
    output_dir: Path,
    iou_threshold: float = 0.5,
    points: int = 1001,
) -> dict[str, dict[str, str]]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = [cat for cat in gt_data.get("categories", []) if "id" in cat]
    categories.sort(key=lambda cat: int(cat["id"]))
    class_ids = [int(cat["id"]) for cat in categories]
    class_names = [str(cat.get("name", cat["id"])) for cat in categories]
    figures: dict[str, dict[str, str]] = {}

    confusion = metrics.get("confusion_matrix") or {}
    labels = ["fundo" if str(label).lower() == "background" else str(label) for label in (confusion.get("labels_true") or class_names + ["background"])]
    figures["confusion_matrix"] = _plot_matrix(
        confusion.get("absolute") or np.zeros((len(labels), len(labels))).tolist(),
        labels,
        title="Matriz de confusão",
        normalized=False,
        output_dir=output_dir,
        stem="confusion_matrix",
    )
    figures["confusion_matrix_normalized"] = _plot_matrix(
        confusion.get("normalized_by_true") or np.zeros((len(labels), len(labels))).tolist(),
        labels,
        title="Matriz de confusão normalizada",
        normalized=True,
        output_dir=output_dir,
        stem="confusion_matrix_normalized",
    )

    map50 = metrics.get("map50")
    if map50 is None:
        map50 = 0.0
    map50_for_legend = float(map50)
    if abs(map50_for_legend - float(metrics.get("map50") or 0.0)) > 0.001:
        raise AssertionError("mAP@0.5 da legenda diverge do arquivo de resultados em mais de 0.001.")

    if not predictions_for_curves:
        msg = "Nenhuma predição foi emitida; curvas definidas como zero."
        for stem, title in [
            ("BoxP_curve", "Precisão x confiança"),
            ("BoxR_curve", "Revocação x confiança"),
            ("BoxF1_curve", "F1 x confiança"),
            ("BoxPR_curve", "Precisão x revocação"),
        ]:
            figures[stem] = _empty_figure(output_dir, stem, title, msg)
        return figures

    thresholds = np.linspace(0.0, 1.0, max(1001, int(points)))
    colors = _colors(max(1, len(class_ids)))
    per_class_metrics = metrics.get("per_class") or {}
    per_class_curves: dict[int, dict[str, np.ndarray | float | str]] = {}
    aggregate_tp: list[np.ndarray] = []
    aggregate_scores: list[np.ndarray] = []
    total_gt_all = 0

    for idx, cat_id in enumerate(class_ids):
        scores, tp, total_gt = _match_predictions(
            gt_data,
            predictions_for_curves,
            category_id=cat_id,
            iou_threshold=iou_threshold,
        )
        precision, recall, f1 = _curve_values(scores, tp, total_gt, thresholds)
        ap50 = per_class_metrics.get(str(cat_id), {}).get("ap50")
        per_class_curves[cat_id] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap50": 0.0 if ap50 is None else float(ap50),
            "name": class_names[idx],
        }
        aggregate_tp.append(tp)
        aggregate_scores.append(scores)
        total_gt_all += total_gt

    all_scores = np.concatenate(aggregate_scores) if aggregate_scores else np.asarray([], dtype=np.float64)
    all_tp = np.concatenate(aggregate_tp) if aggregate_tp else np.asarray([], dtype=np.int64)
    precision_all, recall_all, f1_all = _curve_values(all_scores, all_tp, total_gt_all, thresholds)

    def plot_conf_curve(stem: str, ylabel: str, key: str, aggregate: np.ndarray, show_peak: bool) -> None:
        plt = _load_pyplot()
        fig, ax = plt.subplots(figsize=(5.9, 4.0))
        for idx, cat_id in enumerate(class_ids):
            curve = per_class_curves[cat_id][key]
            ax.plot(thresholds, curve, color=colors[idx], linewidth=0.9, alpha=0.85, label=str(per_class_curves[cat_id]["name"]))
        if show_peak and aggregate.size:
            best_idx = int(np.argmax(aggregate))
            label = f"agregado max={aggregate[best_idx]:.3f} em confiança={thresholds[best_idx]:.3f}"
        else:
            label = "agregado"
        ax.plot(thresholds, aggregate, color="black", linewidth=1.8, label=label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confiança", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        ax.legend(fontsize=7, loc="best")
        figures[stem] = _save(fig, output_dir, stem)
        plt.close(fig)

    plot_conf_curve("BoxP_curve", "Precisão", "precision", precision_all, True)
    plot_conf_curve("BoxR_curve", "Revocação", "recall", recall_all, True)
    plot_conf_curve("BoxF1_curve", "F1", "f1", f1_all, True)

    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(5.9, 4.0))
    for idx, cat_id in enumerate(class_ids):
        item = per_class_curves[cat_id]
        ax.plot(item["recall"], item["precision"], color=colors[idx], linewidth=0.9, alpha=0.85, label=f"{item['name']} AP50={item['ap50']:.3f}")
    ax.plot(recall_all, precision_all, color="black", linewidth=1.8, label=f"agregado mAP50={map50_for_legend:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Revocação", fontsize=9)
    ax.set_ylabel("Precisão", fontsize=9)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=7, loc="best")
    figures["BoxPR_curve"] = _save(fig, output_dir, "BoxPR_curve")
    plt.close(fig)

    return figures
