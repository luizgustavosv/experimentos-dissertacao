from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from ultralytics import YOLO

from app.detectors.base import Logger


def evaluate_yolo(
    data_yaml: str,
    weights_path: str,
    out_dir: str,
    split: str = "val",
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    conf: float = 0.001,
    iou: float = 0.6,
    logger: Optional[Logger] = None,
    log_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    def _emit(message: str) -> None:
        print(message, flush=True)
        if log_cb:
            log_cb(message)
        if logger:
            logger(message)

    data_path = Path(data_yaml).expanduser().resolve()
    weights = Path(weights_path).expanduser().resolve()
    output_dir = Path(out_dir).expanduser().resolve()

    if not data_path.is_file():
        raise FileNotFoundError(f"Arquivo data.yaml não encontrado: {data_path}")
    if not weights.is_file():
        raise FileNotFoundError(f"Pesos YOLO não encontrados: {weights}")

    output_dir.mkdir(parents=True, exist_ok=True)

    _emit("[EVAL][YOLO] Iniciando avaliação via Ultralytics API")
    _emit(
        f"[EVAL][YOLO] Parâmetros: data={data_path}, weights={weights}, out_dir={output_dir}, split={split}, "
        f"imgsz={imgsz}, batch={batch}, device={device}, conf={conf}, iou={iou}"
    )

    model = YOLO(str(weights))
    _emit("[EVAL][YOLO] Executando model.val()...")
    results = model.val(
        data=str(data_path),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        iou=iou,
        project=str(output_dir),
        name="yolo_eval",
        verbose=True,
    )

    box = getattr(results, "box", None)
    precision = float(getattr(box, "mp", 0.0)) if box is not None else 0.0
    recall = float(getattr(box, "mr", 0.0)) if box is not None else 0.0
    map50 = float(getattr(box, "map50", 0.0)) if box is not None else 0.0
    map50_95 = float(getattr(box, "map", 0.0)) if box is not None else 0.0

    metrics_dict = getattr(results, "results_dict", None)
    if isinstance(metrics_dict, dict):
        precision = precision or float(metrics_dict.get("metrics/precision", metrics_dict.get("precision", 0.0)))
        recall = recall or float(metrics_dict.get("metrics/recall", metrics_dict.get("recall", 0.0)))
        map50 = map50 or float(metrics_dict.get("metrics/mAP50(B)", metrics_dict.get("map50", 0.0)))
        map50_95 = map50_95 or float(metrics_dict.get("metrics/mAP50-95(B)", metrics_dict.get("map", 0.0)))

    result = {
        "algorithm": "YOLO",
        "weights": str(weights),
        "data": str(data_path),
        "split": split,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "conf": conf,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "timestamp": datetime.now().isoformat(),
        "out_dir": str(output_dir),
    }

    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"

    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_fields = [
        "timestamp",
        "algorithm",
        "data",
        "weights",
        "split",
        "imgsz",
        "batch",
        "device",
        "conf",
        "iou",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "out_dir",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        if write_header:
            writer.writeheader()
        writer.writerow({field: result.get(field, "") for field in csv_fields})

    _emit(
        f"[EVAL][YOLO] Precisão={precision:.4f} | Recall={recall:.4f} | mAP@0.5={map50:.4f} | mAP@0.5:0.95={map50_95:.4f}"
    )
    _emit(f"[EVAL][YOLO] metrics.json salvo em {json_path}")
    _emit(f"[EVAL][YOLO] metrics.csv atualizado em {csv_path}")

    return result
