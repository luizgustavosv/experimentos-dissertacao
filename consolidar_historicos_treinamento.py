from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


ROOT = Path(os.environ.get("EXPERIMENTOS_CNNS_ROOT", ".")).expanduser().resolve()
OUT = Path(
    os.environ.get("EXPERIMENTOS_HISTORY_OUT", ROOT / "historicos_consolidados")
).expanduser().resolve()


EXPERIMENTS = [
    {
        "id": "yolo12n_visdrone_100epochs",
        "modelo": "YOLOv12n",
        "dataset": "VisDrone",
        "tipo": "principal",
        "segments": [
            ROOT / "YOLO" / "yolo12n_visdrone" / "pesos" / "yolo_visdrone2" / "results.csv",
            ROOT / "YOLO" / "yolo12n_visdrone" / "pesos" / "yolo_visdrone3" / "results.csv",
        ],
    },
    {
        "id": "yolo12n_heridal_100epochs",
        "modelo": "YOLOv12n",
        "dataset": "HERIDAL",
        "tipo": "principal",
        "segments": [],
        "note": "Nao foi localizado results.csv/training_history para este treinamento; apenas best.pt e last.pt foram preservados.",
    },
    {
        "id": "faster_rcnn_heridal_100epochs",
        "modelo": "Faster R-CNN",
        "dataset": "HERIDAL",
        "tipo": "principal",
        "segments": [
            ROOT / "FASTER R-CNN" / "faster_rcnn_heridal_100epochs" / "checkpoints" / "training_history.csv",
            ROOT / "FASTER R-CNN" / "faster_rcnn_heridal_100epochs" / "2-round" / "checkpoints" / "training_history_2.csv",
        ],
    },
    {
        "id": "faster_rcnn_visdrone_100epochs",
        "modelo": "Faster R-CNN",
        "dataset": "VisDrone",
        "tipo": "principal",
        "segments": [
            ROOT / "FASTER R-CNN" / "faster_rcnn_visdrone_100epochs" / "checkpoints" / "training_history.csv",
            ROOT / "FASTER R-CNN" / "faster_rcnn_visdrone_100epochs" / "2run" / "checkpoints" / "training_history.csv",
            ROOT / "FASTER R-CNN" / "faster_rcnn_visdrone_100epochs" / "3run" / "checkpoints" / "training_history.csv",
        ],
    },
    {
        "id": "retinanet_heridal_100epochs",
        "modelo": "RetinaNet",
        "dataset": "HERIDAL",
        "tipo": "principal",
        "segments": [
            ROOT / "RETINANET" / "retinanet_heridal_100epochs" / "checkpoints" / "training_history.csv",
        ],
    },
    {
        "id": "retinanet_visdrone_100epochs",
        "modelo": "RetinaNet",
        "dataset": "VisDrone",
        "tipo": "principal",
        "segments": [
            ROOT / "RETINANET" / "retinanet_visdrone_100epochs" / "checkpoints" / "training_history.csv",
        ],
    },
    {
        "id": "ssd300_heridal_100epochs",
        "modelo": "SSD300",
        "dataset": "HERIDAL",
        "tipo": "principal",
        "segments": [],
        "note": "Nao foi localizado historico por epoca para o treinamento principal; os pesos guardam apenas metadados pontuais.",
    },
    {
        "id": "ssd300_visdrone_100epochs",
        "modelo": "SSD300",
        "dataset": "VisDrone",
        "tipo": "principal",
        "segments": [],
        "note": "Nao foi localizado historico por epoca para o treinamento principal; os pesos guardam apenas metricas pontuais dos checkpoints.",
    },
    {
        "id": "yolo12n_heridal_20epochs",
        "modelo": "YOLOv12n",
        "dataset": "HERIDAL",
        "tipo": "20 epocas",
        "segments": [
            ROOT / "YOLO" / "yolo_heridal_20epochs" / "yolo_visdrone5" / "results.csv",
        ],
    },
    {
        "id": "yolo12n_visdrone_20epochs",
        "modelo": "YOLOv12n",
        "dataset": "VisDrone",
        "tipo": "20 epocas",
        "segments": [
            ROOT / "YOLO" / "yolo_visdrone_20epochs" / "yolo_visdrone2" / "results.csv",
        ],
    },
    {
        "id": "ssd300_heridal_20epochs",
        "modelo": "SSD300",
        "dataset": "HERIDAL",
        "tipo": "20 epocas",
        "segments": [
            ROOT / "SSD300" / "ssd_heridal_20epochs" / "checkpoints" / "training_history.csv",
        ],
    },
    {
        "id": "ssd300_visdrone_20epochs",
        "modelo": "SSD300",
        "dataset": "VisDrone",
        "tipo": "20 epocas",
        "segments": [
            ROOT / "SSD300" / "ssd_visdrone_20epochs" / "checkpoints" / "ssd_visdrone_20epochs_training_history.csv",
        ],
    },
]


CHECKPOINT_VESTIGES = [
    {
        "id": "ssd300_heridal_100epochs",
        "modelo": "SSD300",
        "dataset": "HERIDAL",
        "files": [
            ROOT / "SSD300" / "ssd_heridal" / "best_by_monitor.pth",
            ROOT / "SSD300" / "ssd_heridal" / "epoch_100.pth",
            ROOT / "SSD300" / "ssd_heridal" / "last.pth",
        ],
    },
    {
        "id": "ssd300_visdrone_100epochs",
        "modelo": "SSD300",
        "dataset": "VisDrone",
        "files": [
            ROOT / "SSD300" / "ssd_visdrone" / "pesos" / "weights" / "best.pth",
            ROOT / "SSD300" / "ssd_visdrone" / "pesos" / "weights" / "checkpoints" / "ckpt_epoch_050.pth",
            ROOT
            / "SSD300"
            / "ssd_visdrone"
            / "pesos"
            / "weights"
            / "checkpoints"
            / "checkpoints"
            / "ssd_last_epoch_0003.pth",
            ROOT / "SSD300" / "ssd_visdrone" / "pesos" / "checkpoints" / "ssd_last_epoch_0100.pth",
        ],
    },
]


PORTUGUESE_LABELS = {
    "epoch": "epoca_original",
    "epoca_continua": "Época",
    "train_loss": "Perda de treinamento",
    "val_loss": "Perda de validação",
    "val_loss_per_image": "Perda de validação por imagem",
    "learning_rate": "Taxa de aprendizado",
    "map": "mAP",
    "map50": "mAP@50",
    "map75": "mAP@75",
    "precision": "Precisão",
    "recall": "Revocação",
    "train/box_loss": "Perda de caixa (treino)",
    "train/cls_loss": "Perda de classe (treino)",
    "train/dfl_loss": "Perda DFL (treino)",
    "val/box_loss": "Perda de caixa (validação)",
    "val/cls_loss": "Perda de classe (validação)",
    "val/dfl_loss": "Perda DFL (validação)",
    "metrics/precision(B)": "Precisão",
    "metrics/recall(B)": "Revocação",
    "metrics/mAP50(B)": "mAP@50",
    "metrics/mAP50-95(B)": "mAP@50-95",
    "lr/pg0": "Taxa de aprendizado",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            df[col] = converted
    return df


def consolidate(exp: dict) -> tuple[pd.DataFrame | None, list[dict]]:
    offset = 0
    frames = []
    summary_rows = []

    for index, path in enumerate(exp["segments"], start=1):
        exists = path.exists()
        if not exists:
            summary_rows.append(
                {
                    "experimento": exp["id"],
                    "modelo": exp["modelo"],
                    "dataset": exp["dataset"],
                    "tipo": exp["tipo"],
                    "rodada": index,
                    "arquivo": str(path),
                    "status": "ausente",
                }
            )
            continue

        df = normalize_columns(pd.read_csv(path))
        if "epoch" not in df.columns:
            raise ValueError(f"Arquivo sem coluna epoch: {path}")

        original_epoch = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")
        df.insert(0, "experimento", exp["id"])
        df.insert(1, "modelo", exp["modelo"])
        df.insert(2, "dataset", exp["dataset"])
        df.insert(3, "tipo_experimento", exp["tipo"])
        df.insert(4, "rodada", index)
        df.insert(5, "arquivo_origem", str(path))
        df.insert(6, "epoca_original", original_epoch)
        df.insert(7, "epoca_continua", range(offset + 1, offset + len(df) + 1))
        frames.append(df)

        summary_rows.append(
            {
                "experimento": exp["id"],
                "modelo": exp["modelo"],
                "dataset": exp["dataset"],
                "tipo": exp["tipo"],
                "rodada": index,
                "arquivo": str(path),
                "status": "usado",
                "linhas": len(df),
                "epoca_original_inicial": int(original_epoch.iloc[0]) if len(original_epoch) else "",
                "epoca_original_final": int(original_epoch.iloc[-1]) if len(original_epoch) else "",
                "epoca_continua_inicial": offset + 1,
                "epoca_continua_final": offset + len(df),
            }
        )
        offset += len(df)

    if not frames:
        summary_rows.append(
            {
                "experimento": exp["id"],
                "modelo": exp["modelo"],
                "dataset": exp["dataset"],
                "tipo": exp["tipo"],
                "rodada": "",
                "arquivo": "",
                "status": "sem_historico",
                "observacao": exp.get("note", ""),
            }
        )
        return None, summary_rows

    return pd.concat(frames, ignore_index=True, sort=False), summary_rows


def plot_experiment(df: pd.DataFrame, exp: dict, out_dir: Path) -> list[Path]:
    paths = []
    x = df["epoca_continua"]
    plot_groups = [
        (
            "perdas",
            [
                "train_loss",
                "val_loss",
                "val_loss_per_image",
                "train/box_loss",
                "train/cls_loss",
                "train/dfl_loss",
                "val/box_loss",
                "val/cls_loss",
                "val/dfl_loss",
            ],
            "Perda",
        ),
        (
            "metricas",
            [
                "map",
                "map50",
                "map75",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
                "precision",
                "recall",
                "metrics/precision(B)",
                "metrics/recall(B)",
            ],
            "Valor",
        ),
    ]

    for suffix, columns, ylabel in plot_groups:
        available = [
            c
            for c in columns
            if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()
        ]
        if not available:
            continue

        plt.figure(figsize=(8.0, 4.8), dpi=180)
        ax = plt.gca()
        for col in available:
            y = pd.to_numeric(df[col], errors="coerce")
            ax.plot(x, y, linewidth=1.8, label=PORTUGUESE_LABELS.get(col, col))

        boundaries = df.groupby("rodada")["epoca_continua"].max().tolist()[:-1]
        for boundary in boundaries:
            ax.axvline(boundary + 0.5, color="0.65", linestyle="--", linewidth=0.8)

        ax.set_title(f"{exp['modelo']} - {exp['dataset']} ({exp['tipo']})")
        ax.set_xlabel("Época acumulada")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        plt.tight_layout()

        out_path = out_dir / f"{exp['id']}_{suffix}.png"
        plt.savefig(out_path)
        plt.close()
        paths.append(out_path)

    return paths


def write_markdown(summary_rows: list[dict], outputs: dict[str, dict]) -> None:
    md = [
        "# Históricos de treinamento consolidados",
        "",
        "As épocas foram renumeradas em `epoca_continua` pela ordem cronológica das rodadas preservadas.",
        "A coluna `epoca_original` mantém a numeração registrada no arquivo de origem.",
        "",
        "## Resumo",
        "",
        "| Experimento | Status | Épocas consolidadas | Rodadas usadas | Observação |",
        "|---|---:|---:|---:|---|",
    ]
    for exp_id, data in outputs.items():
        md.append(
            "| {exp} | {status} | {epochs} | {runs} | {obs} |".format(
                exp=exp_id,
                status=data["status"],
                epochs=data.get("epochs", ""),
                runs=data.get("runs", ""),
                obs=data.get("observacao", ""),
            )
        )
    (OUT / "README_historicos_consolidados.md").write_text("\n".join(md), encoding="utf-8")


def checkpoint_vestiges() -> list[dict]:
    rows = []
    for exp in CHECKPOINT_VESTIGES:
        for path in exp["files"]:
            if not path.exists():
                continue
            try:
                obj = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as exc:
                rows.append(
                    {
                        "experimento": exp["id"],
                        "modelo": exp["modelo"],
                        "dataset": exp["dataset"],
                        "arquivo": str(path),
                        "status": f"erro_leitura: {exc}",
                    }
                )
                continue

            metrics = obj.get("metrics", {}) if isinstance(obj, dict) else {}
            early = obj.get("early_stopping", {}) if isinstance(obj, dict) else {}
            row = {
                "experimento": exp["id"],
                "modelo": exp["modelo"],
                "dataset": exp["dataset"],
                "arquivo": str(path),
                "modificado_em": path.stat().st_mtime,
                "epoch": obj.get("epoch") if isinstance(obj, dict) else "",
                "loss": obj.get("loss") if isinstance(obj, dict) else "",
                "train_loss": metrics.get("train_loss") if isinstance(metrics, dict) else "",
                "val_loss": metrics.get("val_loss") if isinstance(metrics, dict) else "",
                "val_loss_per_image": metrics.get("val_loss_per_image") if isinstance(metrics, dict) else "",
                "val_map": metrics.get("val_map") if isinstance(metrics, dict) else "",
                "early_stopping_best_value": early.get("best_value") if isinstance(early, dict) else "",
                "early_stopping_best_epoch": early.get("best_epoch") if isinstance(early, dict) else "",
                "early_stopping_bad_epochs": early.get("num_bad_epochs") if isinstance(early, dict) else "",
            }
            rows.append(row)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    outputs: dict[str, dict] = {}

    for exp in EXPERIMENTS:
        df, rows = consolidate(exp)
        summary_rows.extend(rows)

        if df is None:
            outputs[exp["id"]] = {
                "status": "sem historico",
                "epochs": 0,
                "runs": 0,
                "observacao": exp.get("note", ""),
            }
            continue

        csv_path = OUT / f"{exp['id']}_historico_consolidado.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        plot_paths = plot_experiment(df, exp, OUT)

        outputs[exp["id"]] = {
            "status": "consolidado",
            "epochs": int(df["epoca_continua"].max()),
            "runs": int(df["rodada"].nunique()),
            "csv": str(csv_path),
            "graficos": [str(p) for p in plot_paths],
        }

    summary_path = OUT / "resumo_historicos.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = sorted({k for row in summary_rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    (OUT / "manifest_historicos.json").write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    vestige_rows = checkpoint_vestiges()
    if vestige_rows:
        pd.DataFrame(vestige_rows).to_csv(
            OUT / "vestigios_checkpoints_ssd300_100epochs.csv",
            index=False,
            encoding="utf-8-sig",
        )
    write_markdown(summary_rows, outputs)
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
