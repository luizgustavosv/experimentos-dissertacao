"""
Utilitários para persistência incremental do histórico de treinamento.

Garante que JSON e CSV sejam salvos atomicamente ao final de cada época,
preservando dados mesmo em caso de reinicialização inesperada do Windows.
"""
from __future__ import annotations

import csv as _csv_mod
import json
import os
from pathlib import Path
from typing import Callable, List, Optional

_DEFAULT_FIELDNAMES: List[str] = [
    "epoch",
    "train_loss",
    "val_loss",
    "map50",
    "precision",
    "recall",
    "epoch_time_sec",
]


def atomic_write_json(path: Path, data: object) -> None:
    """
    Escreve dados como JSON em *path* de forma atômica.

    Grava em arquivo temporário (*.tmp) e faz os.replace() ao final,
    evitando arquivos corrompidos em caso de interrupção durante a escrita.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(str(tmp_path), str(path))
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def atomic_write_csv(
    path: Path,
    data: List[dict],
    fieldnames: Optional[List[str]] = None,
) -> None:
    """
    Escreve *data* como CSV em *path* de forma atômica.

    Grava em arquivo temporário (*.tmp) e faz os.replace() ao final.
    Colunas extras nos dicts são ignoradas (extrasaction="ignore").
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fnames = fieldnames or _DEFAULT_FIELDNAMES
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv_mod.DictWriter(f, fieldnames=fnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)
        os.replace(str(tmp_path), str(path))
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def save_training_history(
    json_path: Path,
    csv_path: Path,
    epoch_history: List[dict],
    fieldnames: Optional[List[str]] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Salva o histórico de treinamento em JSON e CSV atomicamente.

    Deve ser chamado ao final de cada época para garantir persistência
    incremental. Em caso de reinicialização inesperada, o histórico de
    todas as épocas já concluídas estará preservado em disco.

    Args:
        json_path:     Caminho para o arquivo training_history.json.
        csv_path:      Caminho para o arquivo training_history.csv.
        epoch_history: Lista de dicts com os dados de cada época.
        fieldnames:    Colunas do CSV (usa _DEFAULT_FIELDNAMES se None).
        logger:        Callable opcional para logar mensagens (aceita str).
    """
    try:
        atomic_write_json(json_path, epoch_history)
        if logger:
            logger(f"[HISTORY] JSON salvo: {json_path}")
    except Exception as exc:
        if logger:
            logger(f"[HISTORY] Falha ao salvar JSON: {exc}")

    try:
        atomic_write_csv(csv_path, epoch_history, fieldnames)
        if logger:
            logger(f"[HISTORY] CSV salvo: {csv_path}")
    except Exception as exc:
        if logger:
            logger(f"[HISTORY] Falha ao salvar CSV: {exc}")


def load_training_history(json_path: Path) -> List[dict]:
    """
    Carrega histórico de treinamento existente do arquivo JSON.

    Retorna lista vazia se o arquivo não existir, estiver vazio ou
    contiver dados inválidos — nunca levanta exceção.
    """
    try:
        if json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def merge_history(existing: List[dict], new_entries: List[dict]) -> List[dict]:
    """
    Combina histórico existente com novas entradas sem duplicar épocas.

    Mantém a entrada existente em caso de conflito (mesma época).
    Retorna lista ordenada por número de época.
    """
    existing_epochs = {e.get("epoch") for e in existing}
    merged = list(existing)
    for entry in new_entries:
        if entry.get("epoch") not in existing_epochs:
            merged.append(entry)
            existing_epochs.add(entry.get("epoch"))
    merged.sort(key=lambda e: e.get("epoch", 0))
    return merged
