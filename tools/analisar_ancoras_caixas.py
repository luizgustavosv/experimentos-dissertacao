from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn, ssd300_vgg16
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import box_iou


DATASETS_PADRAO = {
    "validacao": {
        "HERIDAL": "HERIDAL_VAL_COCO",
        "VisDrone": "VISDRONE_VAL_COCO",
    },
    "treino": {
        "HERIDAL": "HERIDAL_TRAIN_COCO",
        "VisDrone": "VISDRONE_TRAIN_COCO",
    },
}


@dataclass(frozen=True)
class ModeloAuditado:
    nome: str
    transformacao: str
    anchor_generator: object
    tamanhos_entrada: tuple[int, int]
    grade_por_nivel: tuple[tuple[int, int], ...]
    menor_escala: float
    escalas: tuple[float, ...]
    descricao_ancoras: dict[str, object]


@dataclass(frozen=True)
class GrupoAncoras:
    largura: float
    altura: float
    centros_x: np.ndarray
    centros_y: np.ndarray


def _ceil_div(a: int, b: int) -> int:
    return int(math.ceil(float(a) / float(b)))


def _fmt_num(valor: float, casas: int = 4) -> str:
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return ""
    return f"{float(valor):.{casas}f}".replace(".", ",")


def _fmt_lista(valores: Iterable[float]) -> str:
    return "|".join(_fmt_num(v, 2) for v in valores)


def _shape_generalized_rcnn(h: int, w: int, min_size: int, max_size: int) -> tuple[int, int]:
    menor = float(min(h, w))
    maior = float(max(h, w))
    escala = min(float(min_size) / menor, float(max_size) / maior)
    return int(h * escala), int(w * escala)


def _grades_fpn(h: int, w: int, strides: Iterable[int]) -> tuple[tuple[int, int], ...]:
    return tuple((_ceil_div(h, s), _ceil_div(w, s)) for s in strides)


def _escalas_das_ancoras(ancoras: torch.Tensor) -> tuple[float, ...]:
    wh = (ancoras[:, 2:4] - ancoras[:, 0:2]).clamp(min=0)
    lados = torch.sqrt(wh[:, 0] * wh[:, 1]).detach().cpu().numpy()
    return tuple(float(x) for x in np.unique(np.round(lados, 4)))


def _gerar_ancoras(
    anchor_generator: object,
    tamanho_entrada: tuple[int, int],
    grade_por_nivel: tuple[tuple[int, int], ...],
    device: torch.device,
) -> torch.Tensor:
    h, w = tamanho_entrada
    imagens = ImageList(torch.zeros((1, 3, h, w), dtype=torch.float32, device=device), [(h, w)])
    mapas = [
        torch.zeros((1, 1, gh, gw), dtype=torch.float32, device=device)
        for gh, gw in grade_por_nivel
    ]
    with torch.no_grad():
        return anchor_generator(imagens, mapas)[0]


def _preparar_grupos_ancoras(ancoras: torch.Tensor) -> list[GrupoAncoras]:
    a = ancoras.detach().cpu().numpy().astype(np.float64)
    larguras = np.round(a[:, 2] - a[:, 0], 4)
    alturas = np.round(a[:, 3] - a[:, 1], 4)
    centros_x = np.round((a[:, 0] + a[:, 2]) / 2.0, 4)
    centros_y = np.round((a[:, 1] + a[:, 3]) / 2.0, 4)
    grupos: list[GrupoAncoras] = []
    for largura, altura in np.unique(np.column_stack([larguras, alturas]), axis=0):
        mask = (larguras == largura) & (alturas == altura)
        grupos.append(
            GrupoAncoras(
                largura=float(largura),
                altura=float(altura),
                centros_x=np.unique(centros_x[mask]),
                centros_y=np.unique(centros_y[mask]),
            )
        )
    return grupos


def construir_modelos(device: torch.device) -> dict[str, ModeloAuditado]:
    frcnn = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=2)
    retina = retinanet_resnet50_fpn(weights=None, weights_backbone=None, num_classes=1)
    ssd = ssd300_vgg16(weights=None, weights_backbone=None, num_classes=2)

    specs: dict[str, ModeloAuditado] = {}

    frcnn_h, frcnn_w = _shape_generalized_rcnn(640, 640, 800, 1333)
    frcnn_grade = _grades_fpn(frcnn_h, frcnn_w, (4, 8, 16, 32, 64))
    frcnn_ancoras = _gerar_ancoras(frcnn.rpn.anchor_generator, (frcnn_h, frcnn_w), frcnn_grade, device)
    frcnn_escalas = _escalas_das_ancoras(frcnn_ancoras)
    specs["Faster R-CNN"] = ModeloAuditado(
        nome="Faster R-CNN",
        transformacao="DetectionResize(640x640) + GeneralizedRCNNTransform(min_size=(800,), max_size=1333)",
        anchor_generator=frcnn.rpn.anchor_generator,
        tamanhos_entrada=(frcnn_h, frcnn_w),
        grade_por_nivel=frcnn_grade,
        menor_escala=min(frcnn_escalas),
        escalas=frcnn_escalas,
        descricao_ancoras={
            "sizes": frcnn.rpn.anchor_generator.sizes,
            "aspect_ratios": frcnn.rpn.anchor_generator.aspect_ratios,
            "num_anchors_per_location": frcnn.rpn.anchor_generator.num_anchors_per_location(),
        },
    )

    retina_h, retina_w = _shape_generalized_rcnn(640, 640, 800, 1333)
    retina_grade = _grades_fpn(retina_h, retina_w, (8, 16, 32, 64, 128))
    retina_ancoras = _gerar_ancoras(retina.anchor_generator, (retina_h, retina_w), retina_grade, device)
    retina_escalas = _escalas_das_ancoras(retina_ancoras)
    specs["RetinaNet"] = ModeloAuditado(
        nome="RetinaNet",
        transformacao="DetectionResize(640x640) + GeneralizedRCNNTransform(min_size=(800,), max_size=1333)",
        anchor_generator=retina.anchor_generator,
        tamanhos_entrada=(retina_h, retina_w),
        grade_por_nivel=retina_grade,
        menor_escala=min(retina_escalas),
        escalas=retina_escalas,
        descricao_ancoras={
            "sizes": retina.anchor_generator.sizes,
            "aspect_ratios": retina.anchor_generator.aspect_ratios,
            "num_anchors_per_location": retina.anchor_generator.num_anchors_per_location(),
        },
    )

    ssd_grade = ((38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1))
    ssd_ancoras = _gerar_ancoras(ssd.anchor_generator, (300, 300), ssd_grade, device)
    ssd_escalas = _escalas_das_ancoras(ssd_ancoras)
    specs["SSD300"] = ModeloAuditado(
        nome="SSD300",
        transformacao="GeneralizedRCNNTransform(fixed_size=(300, 300), min_size=(300,), max_size=300)",
        anchor_generator=ssd.anchor_generator,
        tamanhos_entrada=(300, 300),
        grade_por_nivel=ssd_grade,
        menor_escala=min(ssd_escalas),
        escalas=ssd_escalas,
        descricao_ancoras={
            "aspect_ratios": ssd.anchor_generator.aspect_ratios,
            "scales": ssd.anchor_generator.scales,
            "steps": ssd.anchor_generator.steps,
            "num_anchors_per_location": ssd.anchor_generator.num_anchors_per_location(),
        },
    )
    return specs


def carregar_coco(caminho: Path) -> dict[str, object]:
    with caminho.open("r", encoding="utf-8") as f:
        return json.load(f)


def caixas_por_imagem(coco: dict[str, object]) -> tuple[dict[int, dict[str, object]], dict[int, list[list[float]]]]:
    imagens = {int(img["id"]): img for img in coco.get("images", [])}
    anns: dict[int, list[list[float]]] = {image_id: [] for image_id in imagens}
    for ann in coco.get("annotations", []):
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        x, y, w, h = [float(v) for v in ann["bbox"]]
        if w <= 0 or h <= 0:
            continue
        anns.setdefault(int(ann["image_id"]), []).append([x, y, x + w, y + h])
    return imagens, anns


def redimensionar_caixas(
    caixas: torch.Tensor,
    largura_original: int,
    altura_original: int,
    modelo: ModeloAuditado,
) -> torch.Tensor:
    h_destino, w_destino = modelo.tamanhos_entrada
    escala_x = float(w_destino) / float(largura_original)
    escala_y = float(h_destino) / float(altura_original)
    saida = caixas.clone()
    saida[:, [0, 2]] *= escala_x
    saida[:, [1, 3]] *= escala_y
    return saida


def _candidatos_mais_proximos(valores: np.ndarray, centro: float) -> np.ndarray:
    pos = int(np.searchsorted(valores, centro))
    idxs = [pos - 1, pos, pos + 1]
    idxs = [i for i in idxs if 0 <= i < len(valores)]
    if not idxs:
        return np.asarray([], dtype=np.float64)
    return valores[np.unique(idxs)]


def _candidatos_mais_proximos_matriz(valores: np.ndarray, centros: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(valores, centros)
    idxs = np.stack([pos - 1, pos, pos + 1], axis=1)
    validos = (idxs >= 0) & (idxs < len(valores))
    idxs_clip = np.clip(idxs, 0, max(len(valores) - 1, 0))
    candidatos = valores[idxs_clip]
    return np.where(validos, candidatos, np.nan)


def max_iou_por_caixa(caixas: torch.Tensor, grupos: list[GrupoAncoras]) -> torch.Tensor:
    if caixas.numel() == 0:
        return torch.empty((0,), dtype=torch.float32)

    caixas_np = caixas.detach().cpu().numpy().astype(np.float64)
    maximos = np.zeros((caixas_np.shape[0],), dtype=np.float64)
    x1 = caixas_np[:, 0]
    y1 = caixas_np[:, 1]
    x2 = caixas_np[:, 2]
    y2 = caixas_np[:, 3]
    area_box = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    validas = area_box > 0
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    for grupo in grupos:
        xs = _candidatos_mais_proximos_matriz(grupo.centros_x, cx)
        ys = _candidatos_mais_proximos_matriz(grupo.centros_y, cy)
        area_anchor = grupo.largura * grupo.altura
        for col_x in range(xs.shape[1]):
            ax = xs[:, col_x]
            valido_x = np.isfinite(ax)
            ax1 = ax - grupo.largura / 2.0
            ax2 = ax + grupo.largura / 2.0
            for col_y in range(ys.shape[1]):
                ay = ys[:, col_y]
                mascara = validas & valido_x & np.isfinite(ay)
                if not np.any(mascara):
                    continue
                ay1 = ay - grupo.altura / 2.0
                ay2 = ay + grupo.altura / 2.0
                inter_w = np.maximum(0.0, np.minimum(x2, ax2) - np.maximum(x1, ax1))
                inter_h = np.maximum(0.0, np.minimum(y2, ay2) - np.maximum(y1, ay1))
                inter = inter_w * inter_h
                iou = inter / np.maximum(area_box + area_anchor - inter, 1e-12)
                maximos[mascara] = np.maximum(maximos[mascara], iou[mascara])

    return torch.from_numpy(maximos.astype(np.float32))


def max_iou_por_caixa_forca_bruta(caixas: torch.Tensor, ancoras: torch.Tensor, chunk_ancoras: int) -> torch.Tensor:
    if caixas.numel() == 0:
        return torch.empty((0,), dtype=torch.float32, device=ancoras.device)
    maximos = torch.zeros((caixas.shape[0],), dtype=torch.float32, device=ancoras.device)
    for inicio in range(0, ancoras.shape[0], chunk_ancoras):
        bloco = ancoras[inicio : inicio + chunk_ancoras]
        maximos = torch.maximum(maximos, box_iou(caixas, bloco).max(dim=1).values)
    return maximos


def validar_busca_local(ancoras_por_modelo: dict[str, torch.Tensor]) -> None:
    rng = np.random.default_rng(42)
    for modelo_nome, ancoras in ancoras_por_modelo.items():
        grupos = _preparar_grupos_ancoras(ancoras)
        limite = 300 if modelo_nome == "SSD300" else 800
        for _ in range(10):
            x1 = float(rng.uniform(0, limite - 40))
            y1 = float(rng.uniform(0, limite - 40))
            w = float(rng.uniform(4, 120))
            h = float(rng.uniform(4, 120))
            caixas = torch.tensor([[x1, y1, min(limite, x1 + w), min(limite, y1 + h)]], dtype=torch.float32)
            rapido = float(max_iou_por_caixa(caixas, grupos)[0].item())
            bruto = float(max_iou_por_caixa_forca_bruta(caixas, ancoras.cpu(), 100000)[0].item())
            if abs(rapido - bruto) > 1e-5:
                raise RuntimeError(
                    f"Validação da busca local falhou para {modelo_nome}: rápido={rapido} bruto={bruto}"
                )


def estatisticas_lado(lados: np.ndarray) -> dict[str, float]:
    if lados.size == 0:
        return {
            "min": float("nan"),
            "p5": float("nan"),
            "p25": float("nan"),
            "mediana": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "min": float(np.min(lados)),
        "p5": float(np.percentile(lados, 5)),
        "p25": float(np.percentile(lados, 25)),
        "mediana": float(np.percentile(lados, 50)),
        "p75": float(np.percentile(lados, 75)),
        "p95": float(np.percentile(lados, 95)),
        "max": float(np.max(lados)),
    }


def auditar_dataset_modelo(
    dataset_nome: str,
    coco_path: Path,
    modelo: ModeloAuditado,
    grupos_ancoras: list[GrupoAncoras],
    device: torch.device,
) -> dict[str, object]:
    coco = carregar_coco(coco_path)
    imagens, anns = caixas_por_imagem(coco)
    lados: list[float] = []
    max_ious: list[float] = []

    for image_id, img in imagens.items():
        caixas_img = anns.get(image_id) or []
        if not caixas_img:
            continue
        largura = int(img["width"])
        altura = int(img["height"])
        caixas = torch.tensor(caixas_img, dtype=torch.float32, device=device)
        caixas_modelo = redimensionar_caixas(caixas, largura, altura, modelo)
        wh = (caixas_modelo[:, 2:4] - caixas_modelo[:, 0:2]).clamp(min=0)
        lados.extend(torch.sqrt(wh[:, 0] * wh[:, 1]).detach().cpu().tolist())
        max_ious.extend(max_iou_por_caixa(caixas_modelo, grupos_ancoras).detach().cpu().tolist())

    lados_np = np.asarray(lados, dtype=np.float64)
    ious_np = np.asarray(max_ious, dtype=np.float64)
    stats = estatisticas_lado(lados_np)
    pct_iou_05 = float(np.mean(ious_np < 0.5) * 100.0) if ious_np.size else float("nan")
    pct_iou_04 = float(np.mean(ious_np < 0.4) * 100.0) if ious_np.size else float("nan")
    pct_iou_03 = float(np.mean(ious_np < 0.3) * 100.0) if ious_np.size else float("nan")
    return {
        "Modelo": modelo.nome,
        "Dataset": dataset_nome,
        "Arquivo de anotações": str(coco_path),
        "Total de caixas": int(lados_np.size),
        "Lado mínimo": stats["min"],
        "Lado p5": stats["p5"],
        "Lado p25": stats["p25"],
        "Lado mediana": stats["mediana"],
        "Lado p75": stats["p75"],
        "Lado p95": stats["p95"],
        "Lado máximo": stats["max"],
        "% IoU máximo < 0,5": pct_iou_05,
        "% IoU máximo < 0,4": pct_iou_04,
        "% IoU máximo < 0,3": pct_iou_03,
        "Menor escala de âncora": modelo.menor_escala,
        "Escalas de âncora": modelo.escalas,
        "Transformação aplicada": modelo.transformacao,
        "_lados": lados_np,
        "_ious": ious_np,
    }


def escrever_csv(modelo_nome: str, linhas: list[dict[str, object]], out_dir: Path, sufixo: str = "") -> Path:
    colunas = [
        "Modelo",
        "Dataset",
        "Arquivo de anotações",
        "Total de caixas",
        "Lado mínimo",
        "Lado p5",
        "Lado p25",
        "Lado mediana",
        "Lado p75",
        "Lado p95",
        "Lado máximo",
        "% IoU máximo < 0,5",
        "% IoU máximo < 0,4",
        "% IoU máximo < 0,3",
        "Menor escala de âncora",
        "Escalas de âncora",
        "Transformação aplicada",
    ]
    path = out_dir / f"auditoria_ancoras_{modelo_nome.lower().replace(' ', '_').replace('-', '')}{sufixo}.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(colunas)
        for linha in linhas:
            writer.writerow([
                _fmt_num(linha[c], 4) if isinstance(linha[c], float) else _fmt_lista(linha[c]) if c == "Escalas de âncora" else linha[c]
                for c in colunas
            ])
    return path


def plotar_histogramas(
    resultados: list[dict[str, object]],
    modelos: dict[str, ModeloAuditado],
    out_dir: Path,
    sufixo: str = "",
) -> list[Path]:
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    datasets = sorted({str(r["Dataset"]) for r in resultados})
    for dataset in datasets:
        linhas = [r for r in resultados if r["Dataset"] == dataset]
        fig, axes = plt.subplots(len(linhas), 1, figsize=(11, 3.8 * len(linhas)), constrained_layout=True)
        if len(linhas) == 1:
            axes = [axes]
        for ax, linha in zip(axes, linhas):
            modelo = modelos[str(linha["Modelo"])]
            lados = linha["_lados"]
            ax.hist(lados, bins=50, color="#4477aa", alpha=0.78)
            for escala in modelo.escalas:
                ax.axvline(escala, color="#cc3311", alpha=0.22, linewidth=1)
            ax.axvline(modelo.menor_escala, color="#000000", linewidth=1.6, linestyle="--")
            ax.set_title(f"{dataset} - {modelo.nome}")
            ax.set_xlabel("Lado equivalente da caixa sqrt(area) em pixels no espaço do modelo")
            ax.set_ylabel("Frequência")
            ax.grid(axis="y", alpha=0.25)
        path = out_dir / f"histograma_lado_caixa_{dataset.lower()}{sufixo}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def _indice_resultados(resultados: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    return {(str(r["Modelo"]), str(r["Dataset"])): r for r in resultados}


def _linha_comparacao_treino_validacao(
    treino: dict[str, object],
    validacao: dict[str, object],
) -> str:
    pct_treino = float(treino["% IoU máximo < 0,5"])
    pct_validacao = float(validacao["% IoU máximo < 0,5"])
    dif = pct_treino - pct_validacao
    med_treino = float(treino["Lado mediana"])
    med_validacao = float(validacao["Lado mediana"])
    dif_med = med_treino - med_validacao
    alerta = " **ALERTA**" if abs(dif) > 5.0 else ""
    return (
        f"| {treino['Modelo']} | {treino['Dataset']} | {pct_treino:.2f} | "
        f"{pct_validacao:.2f} | {dif:+.2f} | {med_treino:.2f} | "
        f"{med_validacao:.2f} | {dif_med:+.2f} |{alerta}"
    )


def _investigar_divergencias(
    resultados_treino: list[dict[str, object]],
    resultados_validacao: list[dict[str, object]],
) -> list[str]:
    linhas: list[str] = []
    idx_validacao = _indice_resultados(resultados_validacao)
    for treino in resultados_treino:
        chave = (str(treino["Modelo"]), str(treino["Dataset"]))
        validacao = idx_validacao.get(chave)
        if validacao is None:
            continue
        dif_pp = float(treino["% IoU máximo < 0,5"]) - float(validacao["% IoU máximo < 0,5"])
        if abs(dif_pp) <= 5.0:
            continue
        med_t = float(treino["Lado mediana"])
        med_v = float(validacao["Lado mediana"])
        p25_t = float(treino["Lado p25"])
        p25_v = float(validacao["Lado p25"])
        p75_t = float(treino["Lado p75"])
        p75_v = float(validacao["Lado p75"])
        linhas.append(
            f"- {treino['Modelo']} / {treino['Dataset']}: diferença {dif_pp:+.2f} pp; "
            f"medianas {med_t:.2f} vs {med_v:.2f}, p25 {p25_t:.2f} vs {p25_v:.2f}, "
            f"p75 {p75_t:.2f} vs {p75_v:.2f}."
        )
    if not linhas:
        linhas.append("- Nenhuma diferença em IoU<0,5 passou de 5 pp; não há sinal forte de divergência entre splits.")
    return linhas


def escrever_resumo(
    resultados: list[dict[str, object]],
    out_dir: Path,
    split: str,
    resultados_validacao: list[dict[str, object]] | None = None,
    sufixo: str = "",
) -> Path:
    linhas = [
        f"Resumo da auditoria de escala caixa-âncora - split {split}",
        "Critério: caixa inalcançável = IoU máximo com qualquer âncora < 0,5.",
    ]
    confirma = any(float(r["% IoU máximo < 0,5"]) >= 50.0 for r in resultados)
    linhas.append("Hipótese confirmada." if confirma else "Hipótese não confirmada como causa dominante.")
    for r in resultados:
        linhas.append(
            f"{r['Modelo']} / {r['Dataset']}: {float(r['% IoU máximo < 0,5']):.2f}% <0,5; "
            f"{float(r['% IoU máximo < 0,4']):.2f}% <0,4; {float(r['% IoU máximo < 0,3']):.2f}% <0,3."
        )

    if split == "treino" and resultados_validacao:
        linhas.extend(
            [
                "",
                "Comparação treino x validação (IoU máximo < 0,5, em %; diferença em pp):",
                "| Modelo | Dataset | Treino | Validação | Dif. pp | Mediana treino | Mediana val. | Dif. mediana |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        idx_validacao = _indice_resultados(resultados_validacao)
        for treino in resultados:
            validacao = idx_validacao.get((str(treino["Modelo"]), str(treino["Dataset"])))
            if validacao is not None:
                linhas.append(_linha_comparacao_treino_validacao(treino, validacao))
        linhas.extend(["", "Investigação de divergência de tamanho:", *_investigar_divergencias(resultados, resultados_validacao)])

    path = out_dir / f"resumo_hipotese{sufixo}.txt"
    path.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita dimensões de caixas GT no espaço visto por SSD300, Faster R-CNN e RetinaNet e mede cobertura por âncoras."
    )
    parser.add_argument("--split", choices=["validacao", "treino"], default="validacao")
    parser.add_argument("--heridal-coco", type=Path, default=None)
    parser.add_argument("--visdrone-coco", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("reports") / "auditoria_ancoras")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--chunk-ancoras", type=int, default=50000)
    parser.add_argument("--validar-busca-local", action="store_true")
    return parser.parse_args()


def _env_path(var_name: str) -> Path | None:
    value = os.environ.get(var_name)
    return Path(value).expanduser().resolve() if value else None


def resolver_datasets(split: str, heridal_coco: Path | None = None, visdrone_coco: Path | None = None) -> dict[str, Path | None]:
    defaults = DATASETS_PADRAO[split]
    return {
        "HERIDAL": heridal_coco or _env_path(defaults["HERIDAL"]),
        "VisDrone": visdrone_coco or _env_path(defaults["VisDrone"]),
    }


def validar_caminhos_datasets(datasets: dict[str, Path | None]) -> None:
    for nome, caminho in datasets.items():
        if caminho is None:
            raise FileNotFoundError(
                f"Informe o arquivo COCO de {nome} por argumento CLI ou variavel de ambiente."
            )
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo COCO de {nome} não encontrado: {caminho}")


def auditar_datasets(
    datasets: dict[str, Path],
    modelos: dict[str, ModeloAuditado],
    grupos_por_modelo: dict[str, list[GrupoAncoras]],
    device: torch.device,
) -> list[dict[str, object]]:
    resultados: list[dict[str, object]] = []
    for modelo_nome, spec in modelos.items():
        print(f"[INFO] Auditando {modelo_nome}...")
        for dataset_nome, coco_path in datasets.items():
            resultado = auditar_dataset_modelo(
                dataset_nome,
                coco_path,
                spec,
                grupos_por_modelo[modelo_nome],
                device,
            )
            resultados.append(resultado)
            print(
                f"  {dataset_nome}: caixas={resultado['Total de caixas']} "
                f"IoU<0,5={float(resultado['% IoU máximo < 0,5']):.2f}%"
            )
    return resultados


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    datasets = resolver_datasets(args.split, args.heridal_coco, args.visdrone_coco)
    validar_caminhos_datasets(datasets)
    datasets_validacao = resolver_datasets("validacao")
    if args.split == "treino":
        validar_caminhos_datasets(datasets_validacao)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    sufixo = "_treino" if args.split == "treino" else ""

    modelos = construir_modelos(device)
    (out_dir / f"configuracao_modelos{sufixo}.json").write_text(
        json.dumps(
            {
                nome: {
                    "transformacao": spec.transformacao,
                    "tamanho_entrada_modelo": spec.tamanhos_entrada,
                    "grade_por_nivel": spec.grade_por_nivel,
                    "menor_escala": spec.menor_escala,
                    "escalas": spec.escalas,
                    "ancoras": spec.descricao_ancoras,
                }
                for nome, spec in modelos.items()
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    ancoras_por_modelo = {
        nome: _gerar_ancoras(spec.anchor_generator, spec.tamanhos_entrada, spec.grade_por_nivel, device)
        for nome, spec in modelos.items()
    }
    if args.validar_busca_local:
        validar_busca_local(ancoras_por_modelo)
    grupos_por_modelo = {
        nome: _preparar_grupos_ancoras(ancoras.cpu())
        for nome, ancoras in ancoras_por_modelo.items()
    }

    print(f"[INFO] Split principal: {args.split}")
    for dataset_nome, coco_path in datasets.items():
        print(f"[INFO] {dataset_nome}: {coco_path}")
    resultados = auditar_datasets(datasets, modelos, grupos_por_modelo, device)

    resultados_validacao = None
    if args.split == "treino":
        print("[INFO] Recalculando validação para comparação treino x validação...")
        resultados_validacao = auditar_datasets(datasets_validacao, modelos, grupos_por_modelo, device)

    for modelo_nome in modelos:
        escrever_csv(modelo_nome, [r for r in resultados if r["Modelo"] == modelo_nome], out_dir, sufixo)
    plotar_histogramas(resultados, modelos, out_dir, sufixo)
    resumo_path = escrever_resumo(resultados, out_dir, args.split, resultados_validacao, sufixo)

    print(f"[OK] Artefatos gravados em: {out_dir}")
    print(resumo_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
