from __future__ import annotations

import json
import math
import re
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


STATE_KEYS = {
    "model",
    "model_state",
    "model_state_dict",
    "state_dict",
    "optimizer",
    "optimizer_state",
    "optimizer_state_dict",
    "scheduler_state",
    "scheduler_state_dict",
    "lr_scheduler",
}

METADATA_KEYS = {
    "epoch",
    "epoch_accumulated",
    "accumulated_epoch",
    "epochs_completed",
    "global_epoch",
    "best_metric",
    "metrics",
    "config",
    "meta",
    "train_args",
    "args",
    "date",
    "timestamp",
    "version",
    "license",
    "docs",
    "ema",
    "updates",
    "optimizer",
    "train_metrics",
    "early_stopping",
    "pretrained_checkpoint",
}

ACCUMULATED_EPOCH_KEYS = (
    "epoch_accumulated",
    "accumulated_epoch",
    "epochs_accumulated",
    "epochs_completed",
    "global_epoch",
    "total_epoch",
)


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _jsonable(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return repr(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return {
            "tipo": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
            "requires_grad": bool(getattr(value, "requires_grad", False)),
        }
    if isinstance(value, dict):
        if value and all(torch.is_tensor(v) for v in value.values()):
            keys = list(value.keys())
            return {
                "tipo": "state_dict",
                "total_tensores": len(keys),
                "chaves_exemplo": keys[:20],
                "total_parametros": int(sum(v.numel() for v in value.values())),
            }
        return {str(k): _jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 100:
            return {
                "tipo": type(value).__name__,
                "total_itens": len(value),
                "itens_exemplo": [_jsonable(v, depth + 1) for v in list(value)[:20]],
            }
        return [_jsonable(v, depth + 1) for v in value]
    if hasattr(value, "__dict__"):
        attrs = {
            k: v
            for k, v in vars(value).items()
            if not k.startswith("_") and k in METADATA_KEYS
        }
        return {
            "tipo": f"{type(value).__module__}.{type(value).__name__}",
            **({ "atributos": _jsonable(attrs, depth + 1) } if attrs else {}),
        }
    return repr(value)


def _summarize_heavy_state(value: Any) -> Any:
    if torch.is_tensor(value):
        return _jsonable(value)
    if isinstance(value, dict):
        keys = list(value.keys())
        tensors = [v for v in value.values() if torch.is_tensor(v)]
        nested_dicts = [v for v in value.values() if isinstance(v, dict)]
        summary: dict[str, Any] = {
            "tipo": "dict",
            "total_chaves": len(keys),
            "chaves_exemplo": [str(k) for k in keys[:20]],
        }
        if tensors:
            summary["tensores_diretos"] = len(tensors)
            summary["parametros_tensores_diretos"] = int(sum(t.numel() for t in tensors))
        if nested_dicts:
            summary["dicts_aninhados"] = len(nested_dicts)
        if value and all(torch.is_tensor(v) for v in value.values()):
            summary["tipo"] = "state_dict"
            summary["total_tensores"] = len(keys)
            summary["total_parametros"] = int(sum(v.numel() for v in value.values()))
        return summary
    if hasattr(value, "state_dict"):
        try:
            return {
                "tipo": f"{type(value).__module__}.{type(value).__name__}",
                "state_dict": _summarize_heavy_state(value.state_dict()),
            }
        except Exception:
            return {"tipo": f"{type(value).__module__}.{type(value).__name__}"}
    return _jsonable(value, depth=1)


def _coerce_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value >= 0 and value.is_integer():
        return int(value)
    return None


def _is_yolo_checkpoint(algorithm: str, loaded: Any) -> bool:
    if algorithm.strip().upper() == "YOLO":
        return True
    return isinstance(loaded, dict) and isinstance(loaded.get("train_args"), dict) and "best_fitness" in loaded


def _yolo_run_dir_from_weight(path: Path) -> Path:
    return path.parent.parent if path.parent.name.lower() == "weights" else path.parent


def _epoch_from_results_csv(run_dir: Path) -> tuple[int | None, str | None]:
    results_csv = run_dir / "results.csv"
    if not results_csv.is_file():
        return None, None
    try:
        with results_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None, None
    epochs = []
    for row in rows:
        normalized = {str(k).strip(): v for k, v in row.items()}
        epoch = _coerce_non_negative_int(_maybe_number(normalized.get("epoch")))
        if epoch is not None:
            epochs.append(epoch)
    if epochs:
        return max(epochs), "results.csv.max_epoch"
    if rows:
        return len(rows), "results.csv.num_linhas"
    return None, None


def _maybe_number(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def _yolo_relative_epoch(path: Path, loaded: Any) -> tuple[int | None, str | None]:
    from_csv, csv_source = _epoch_from_results_csv(_yolo_run_dir_from_weight(path))
    if from_csv is not None:
        return from_csv, csv_source
    if isinstance(loaded, dict):
        raw_epoch = _coerce_non_negative_int(loaded.get("epoch"))
        if raw_epoch is not None:
            return raw_epoch + 1, "checkpoint.epoch+1"
        train_args = loaded.get("train_args")
        if isinstance(train_args, dict):
            epoch = _coerce_non_negative_int(train_args.get("epochs"))
            if epoch is not None:
                return epoch, "checkpoint.train_args.epochs"
    return None, None


def _infer_epoch(path: Path, loaded: Any, sidecar: dict[str, Any] | None) -> tuple[int | None, str | None]:
    if isinstance(loaded, dict):
        epoch = _coerce_non_negative_int(loaded.get("epoch"))
        if epoch is not None:
            return epoch, "checkpoint.epoch"
    if sidecar:
        for key in ("last_epoch", "best_epoch", "epoch"):
            epoch = _coerce_non_negative_int(sidecar.get(key))
            if epoch is not None:
                return epoch, f"sidecar.{key}"
    patterns = [
        r"checkpoint_epoch_(\d+)",
        r"ckpt_epoch_(\d+)",
        r"best_epoch_(\d+)",
        r"epoch[_-]?(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, path.stem, flags=re.IGNORECASE)
        if match:
            return int(match.group(1)), "nome_arquivo"
    epoch, source = _epoch_from_related_files(path)
    if epoch is not None:
        return epoch, source
    if isinstance(loaded, dict):
        train_args = loaded.get("train_args")
        if isinstance(train_args, dict):
            epoch = _coerce_non_negative_int(train_args.get("epochs"))
            if epoch is not None:
                return epoch, "checkpoint.train_args.epochs"
        args = loaded.get("args")
        if isinstance(args, dict):
            epoch = _coerce_non_negative_int(args.get("epochs"))
            if epoch is not None:
                return epoch, "checkpoint.args.epochs"
        for key in ("updates",):
            epoch = _coerce_non_negative_int(loaded.get(key))
            if epoch is not None:
                return epoch, f"checkpoint.{key}"
    return None, None


def _paths_match(path: Path, value: Any) -> bool:
    if not value:
        return False
    try:
        if Path(str(value)).expanduser().resolve() == path.expanduser().resolve():
            return True
    except Exception:
        pass
    return Path(str(value)).name.lower() == path.name.lower()


def _candidate_metadata_dirs(path: Path) -> list[Path]:
    dirs: list[Path] = []
    for base in (path.parent, path.parent.parent):
        dirs.extend(
            [
                base,
                base / "checkpoints",
                base / "weights",
                base / "pesos",
            ]
        )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in dirs:
        try:
            key = str(candidate.expanduser().resolve()).lower()
        except Exception:
            key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _epoch_from_filename(path: Path) -> tuple[int | None, str | None]:
    patterns = [
        r"checkpoint_epoch_(\d+)",
        r"ckpt_epoch_(\d+)",
        r"best_epoch_(\d+)",
        r"epoch[_-]?(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, path.stem, flags=re.IGNORECASE)
        if match:
            return int(match.group(1)), "nome_arquivo"
    return None, None


def _epoch_from_related_files(path: Path) -> tuple[int | None, str | None]:
    candidates: list[tuple[int, Path]] = []
    glob_patterns = (
        "checkpoint_epoch_*.pt",
        "checkpoint_epoch_*.pth",
        "ckpt_epoch_*.pt",
        "ckpt_epoch_*.pth",
        "*_last_epoch_*.pt",
        "*_last_epoch_*.pth",
        "*best_epoch_*.pt",
        "*best_epoch_*.pth",
        "epoch_*.pt",
        "epoch_*.pth",
        "epoch*.pt",
        "epoch*.pth",
    )
    for directory in _candidate_metadata_dirs(path):
        if not directory.is_dir():
            continue
        for pattern in glob_patterns:
            for candidate in directory.glob(pattern):
                if not candidate.is_file():
                    continue
                epoch, _ = _epoch_from_filename(candidate)
                if epoch is not None:
                    candidates.append((epoch, candidate))
    if not candidates:
        return None, None
    epoch, candidate = max(candidates, key=lambda item: item[0])
    return epoch, f"arquivo_relacionado.{candidate.name}"


def _max_epoch_from_history(path: Path) -> tuple[int | None, str | None]:
    candidates = [directory / "training_history.json" for directory in _candidate_metadata_dirs(path)]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            rows = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        epochs = [
            epoch
            for row in rows
            if isinstance(row, dict)
            for epoch in [_coerce_non_negative_int(row.get("epoch"))]
            if epoch is not None
        ]
        if epochs:
            return max(epochs), f"{candidate.name}.max_epoch"
    return None, None


def _run_dir_from_weight(path: Path) -> Path:
    if path.parent.name.lower() in {"checkpoints", "weights", "pesos"}:
        return path.parent.parent
    return path.parent


def _resume_round_index(run_dir: Path) -> int | None:
    name = run_dir.name.lower().replace("_", "-").replace(" ", "-")
    patterns = (
        r"^(\d+)-?round$",
        r"^round-?(\d+)$",
        r"^(\d+)-?run$",
        r"^run-?(\d+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, name)
        if match:
            return int(match.group(1))
    return None


def _max_epoch_in_metadata_dir(directory: Path) -> tuple[int | None, str | None]:
    if not directory.is_dir():
        return None, None

    history_path = directory / "training_history.json"
    if history_path.is_file():
        try:
            rows = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            rows = None
        if isinstance(rows, list):
            epochs = [
                epoch
                for row in rows
                if isinstance(row, dict)
                for epoch in [_coerce_non_negative_int(row.get("epoch"))]
                if epoch is not None
            ]
            if epochs:
                return max(epochs), f"{history_path.name}.max_epoch"

    sidecars = sorted(directory.glob("*_ckpt_metadata.json"))
    sidecar_epochs: list[tuple[int, Path, str]] = []
    for sidecar_path in sidecars:
        try:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("last_epoch", "best_epoch", "epoch"):
            epoch = _coerce_non_negative_int(payload.get(key))
            if epoch is not None:
                sidecar_epochs.append((epoch, sidecar_path, key))
    if sidecar_epochs:
        epoch, sidecar_path, key = max(sidecar_epochs, key=lambda item: item[0])
        return epoch, f"{sidecar_path.name}.{key}"

    file_epochs: list[tuple[int, Path]] = []
    for pattern in ("checkpoint_epoch_*.pt", "checkpoint_epoch_*.pth", "ckpt_epoch_*.pt", "ckpt_epoch_*.pth"):
        for candidate in directory.glob(pattern):
            epoch, _ = _epoch_from_filename(candidate)
            if epoch is not None:
                file_epochs.append((epoch, candidate))
    if file_epochs:
        epoch, candidate = max(file_epochs, key=lambda item: item[0])
        return epoch, f"{candidate.name}"
    return None, None


def _max_epoch_in_run_dir(run_dir: Path) -> tuple[int | None, str | None]:
    for directory in (run_dir / "checkpoints", run_dir / "weights", run_dir / "pesos", run_dir):
        epoch, source = _max_epoch_in_metadata_dir(directory)
        if epoch is not None:
            return epoch, f"{run_dir.name}/{source}"
    return None, None


def _accumulated_epoch_from_resume_dirs(path: Path, local_epoch: int | None) -> tuple[int | None, str | None]:
    if local_epoch is None:
        return None, None

    run_dir = _run_dir_from_weight(path)
    current_round = _resume_round_index(run_dir)
    if current_round is None or current_round <= 1:
        return None, None

    experiment_dir = run_dir.parent
    previous_runs: list[tuple[int, Path]] = []
    base_epoch, _ = _max_epoch_in_run_dir(experiment_dir)
    if base_epoch is not None:
        previous_runs.append((1, experiment_dir))

    try:
        siblings = sorted(p for p in experiment_dir.iterdir() if p.is_dir())
    except Exception:
        siblings = []
    for sibling in siblings:
        sibling_round = _resume_round_index(sibling)
        if sibling_round is None or sibling_round >= current_round:
            continue
        previous_runs.append((sibling_round, sibling))

    total_previous = 0
    parts: list[str] = []
    seen_dirs: set[str] = set()
    for round_index, previous_run in sorted(previous_runs, key=lambda item: item[0]):
        try:
            key = str(previous_run.resolve()).lower()
        except Exception:
            key = str(previous_run).lower()
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        epoch, source = _max_epoch_in_run_dir(previous_run)
        if epoch is None:
            continue
        total_previous += epoch
        parts.append(f"{previous_run.name}:{epoch} ({source})")

    if total_previous <= 0:
        return None, None
    parts.append(f"{run_dir.name}:{local_epoch} (checkpoint atual)")
    return total_previous + local_epoch, "cadeia_diretorios_retomada[" + " + ".join(parts) + "]"


def _explicit_accumulated_epoch(loaded: Any) -> tuple[int | None, str | None]:
    if not isinstance(loaded, dict):
        return None, None
    for key in ACCUMULATED_EPOCH_KEYS:
        epoch = _coerce_non_negative_int(loaded.get(key))
        if epoch is not None:
            return epoch, f"checkpoint.{key}"
    meta = loaded.get("meta")
    if isinstance(meta, dict):
        for key in ACCUMULATED_EPOCH_KEYS:
            epoch = _coerce_non_negative_int(meta.get(key))
            if epoch is not None:
                return epoch, f"checkpoint.meta.{key}"
    config = loaded.get("config")
    if isinstance(config, dict):
        for key in ACCUMULATED_EPOCH_KEYS:
            epoch = _coerce_non_negative_int(config.get(key))
            if epoch is not None:
                return epoch, f"checkpoint.config.{key}"
    return None, None


def _accumulated_epoch_from_sidecar(path: Path, sidecar: dict[str, Any] | None) -> tuple[int | None, str | None]:
    if not sidecar:
        return None, None
    has_path_hints = bool(sidecar.get("best_path") or sidecar.get("last_path"))
    if _paths_match(path, sidecar.get("best_path")):
        epoch = _coerce_non_negative_int(sidecar.get("best_epoch"))
        if epoch is not None:
            return epoch, "sidecar.best_epoch"
    if _paths_match(path, sidecar.get("last_path")):
        epoch = _coerce_non_negative_int(sidecar.get("last_epoch"))
        if epoch is not None:
            return epoch, "sidecar.last_epoch"
    if has_path_hints:
        return None, None
    lower_name = path.name.lower()
    if "best" in lower_name:
        epoch = _coerce_non_negative_int(sidecar.get("best_epoch"))
        if epoch is not None:
            return epoch, "sidecar.best_epoch"
    if "last" in lower_name or "checkpoint" in lower_name or "ckpt" in lower_name:
        epoch = _coerce_non_negative_int(sidecar.get("last_epoch"))
        if epoch is not None:
            return epoch, "sidecar.last_epoch"
    for key in ("last_epoch", "best_epoch", "epoch"):
        epoch = _coerce_non_negative_int(sidecar.get(key))
        if epoch is not None:
            return epoch, f"sidecar.{key}"
    return None, None


def _planned_epochs_from_args(loaded: Any) -> tuple[int | None, str | None]:
    if not isinstance(loaded, dict):
        return None, None
    for container_name in ("train_args", "args", "config"):
        container = loaded.get(container_name)
        if isinstance(container, dict):
            for key in ("epochs_to_run", "epochs"):
                epoch = _coerce_non_negative_int(container.get(key))
                if epoch is not None:
                    return epoch, f"checkpoint.{container_name}.{key}"
    return None, None


def _resolve_existing_reference(reference: Any, current_path: Path) -> Path | None:
    if not reference:
        return None
    raw = Path(str(reference)).expanduser()
    candidates: list[Path] = [raw]
    parts_lower = [part.lower() for part in raw.parts]
    if "cnns" in parts_lower:
        idx = parts_lower.index("cnns")
        suffix = Path(*raw.parts[idx + 1 :])
        for parent in current_path.parents:
            candidates.append(parent / suffix)
    suffixes = []
    raw_parts = raw.parts
    for n_parts in range(min(6, len(raw_parts)), 1, -1):
        suffixes.append(Path(*raw_parts[-n_parts:]))
    for parent in current_path.parents:
        for suffix in suffixes:
            candidates.append(parent / suffix)
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        if resolved.is_file():
            return resolved
    return None


def _previous_yolo_weight_path(path: Path, loaded: Any) -> Path | None:
    if not isinstance(loaded, dict):
        return None
    train_args = loaded.get("train_args")
    if not isinstance(train_args, dict):
        return None
    model_path = train_args.get("model")
    if not model_path:
        return None
    if Path(str(model_path)).suffix.lower() not in {".pt", ".pth"}:
        return None
    return _resolve_existing_reference(model_path, path)


def _infer_yolo_accumulated_epoch(
    path: Path,
    loaded: Any,
    local_epoch: int | None,
    visited: set[Path] | None = None,
) -> tuple[int | None, str | None]:
    if local_epoch is None:
        return None, None
    visited = visited or set()
    resolved = path.expanduser().resolve()
    if resolved in visited:
        return local_epoch, "yolo.cadeia_interrompida_ciclo"
    visited.add(resolved)

    previous_path = _previous_yolo_weight_path(path, loaded)
    if previous_path is None:
        return local_epoch, "yolo.epoca_relativa_sem_checkpoint_anterior"
    previous_resolved = previous_path.expanduser().resolve()
    if previous_resolved in visited:
        return local_epoch, "yolo.cadeia_interrompida_ciclo"

    try:
        previous_loaded = _safe_torch_load(previous_resolved)
    except Exception:
        return local_epoch, "yolo.epoca_relativa_checkpoint_anterior_ilegivel"
    if not _is_yolo_checkpoint("YOLO", previous_loaded):
        return local_epoch, "yolo.epoca_relativa_checkpoint_base"

    previous_local, _ = _yolo_relative_epoch(previous_resolved, previous_loaded)
    if previous_local is None:
        return local_epoch, "yolo.epoca_relativa_checkpoint_base"
    previous_accumulated, previous_source = _infer_yolo_accumulated_epoch(
        previous_resolved,
        previous_loaded,
        previous_local,
        visited,
    )
    if previous_accumulated is None:
        return local_epoch, "yolo.epoca_relativa_checkpoint_anterior_sem_epoca"
    return previous_accumulated + local_epoch, (
        f"yolo.cadeia({previous_resolved.name}:{previous_accumulated}+atual:{local_epoch}; {previous_source})"
    )


def _infer_accumulated_epoch(
    path: Path,
    loaded: Any,
    sidecar: dict[str, Any] | None,
    fallback_epoch: int | None,
    fallback_source: str | None,
) -> tuple[int | None, str | None]:
    def own_checkpoint_epoch() -> tuple[int | None, str | None]:
        if fallback_source == "checkpoint.epoch" and fallback_epoch is not None:
            return fallback_epoch, "checkpoint.epoch"
        return None, None

    for resolver in (
        lambda: _explicit_accumulated_epoch(loaded),
        lambda: _accumulated_epoch_from_resume_dirs(path, fallback_epoch),
        lambda: _accumulated_epoch_from_sidecar(path, sidecar),
        lambda: _max_epoch_from_history(path),
        lambda: _epoch_from_filename(path),
        own_checkpoint_epoch,
        lambda: _epoch_from_related_files(path),
        lambda: _planned_epochs_from_args(loaded),
    ):
        epoch, source = resolver()
        if epoch is not None:
            return epoch, source
    if fallback_epoch is not None:
        return fallback_epoch, fallback_source or "ultima_epoca"
    return None, None


def _load_sidecar(path: Path) -> tuple[Path | None, dict[str, Any] | None]:
    candidates: list[Path] = []
    for directory in _candidate_metadata_dirs(path):
        if directory.is_dir():
            candidates.extend(sorted(directory.glob("*_ckpt_metadata.json")))
    generic_candidates: list[tuple[Path, dict[str, Any]]] = []
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        has_path_hints = bool(payload.get("best_path") or payload.get("last_path"))
        if _paths_match(path, payload.get("best_path")) or _paths_match(path, payload.get("last_path")):
            return candidate, payload
        if not has_path_hints:
            generic_candidates.append((candidate, payload))
    if generic_candidates:
        return generic_candidates[0]
    return None, None


def _extract_timestamp(path: Path, loaded: Any, sidecar: dict[str, Any] | None) -> str:
    if sidecar and sidecar.get("timestamp"):
        return str(sidecar["timestamp"])
    if isinstance(loaded, dict):
        for key in ("timestamp", "date", "time"):
            value = loaded.get(key)
            if value:
                return str(value)
        meta = loaded.get("meta")
        if isinstance(meta, dict):
            for key in ("timestamp", "date", "created_at", "finished_at"):
                value = meta.get(key)
                if value:
                    return str(value)
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _checkpoint_format(loaded: Any) -> str:
    if isinstance(loaded, dict):
        keys = set(loaded.keys())
        if "model_state" in keys or "model_state_dict" in keys:
            return "checkpoint_torchvision_com_estado"
        if "state_dict" in keys:
            return "checkpoint_state_dict"
        model_value = loaded.get("model")
        if isinstance(model_value, dict) and model_value and all(torch.is_tensor(v) for v in model_value.values()):
            return "checkpoint_enxuto_model_state"
        if "model" in keys and not torch.is_tensor(loaded.get("model")):
            return "checkpoint_ultralytics_ou_modelo"
        return "dict_checkpoint"
    if hasattr(loaded, "state_dict"):
        return "objeto_modelo_torch"
    return type(loaded).__name__


def read_weights_metadata(weights_path: Path, algorithm: str) -> dict[str, Any]:
    path = weights_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de pesos não encontrado: {path}")

    sidecar_path, sidecar = _load_sidecar(path)
    loaded = _safe_torch_load(path)
    stat = path.stat()
    inferred_epoch, epoch_source = _infer_epoch(path, loaded, sidecar)
    accumulated_epoch, accumulated_source = _infer_accumulated_epoch(path, loaded, sidecar, inferred_epoch, epoch_source)
    if _is_yolo_checkpoint(algorithm, loaded):
        yolo_epoch, yolo_epoch_source = _yolo_relative_epoch(path, loaded)
        if yolo_epoch is not None:
            inferred_epoch, epoch_source = yolo_epoch, yolo_epoch_source
        yolo_accumulated, yolo_accumulated_source = _infer_yolo_accumulated_epoch(path, loaded, inferred_epoch)
        if yolo_accumulated is not None:
            accumulated_epoch, accumulated_source = yolo_accumulated, yolo_accumulated_source
    raw_epoch = loaded.get("epoch") if isinstance(loaded, dict) else None
    avisos: list[str] = []
    if isinstance(raw_epoch, (int, float)) and raw_epoch < 0:
        avisos.append(
            f"Campo checkpoint.epoch={raw_epoch} ignorado por ser negativo; usando {epoch_source or 'nenhuma fonte alternativa'}."
        )
    if accumulated_source and accumulated_source.endswith((".epochs", ".epochs_to_run")):
        avisos.append(
            "Época acumulada inferida a partir do total de épocas configurado; "
            "não havia contador acumulado explícito no checkpoint/sidecar/histórico."
        )

    planned_epoch, _planned_source = _planned_epochs_from_args(loaded)
    has_confident_accumulated_source = str(accumulated_source or "").startswith(
        ("checkpoint.epoch_accumulated", "checkpoint.accumulated_epoch", "cadeia_")
    )
    if (
        planned_epoch is not None
        and inferred_epoch is not None
        and accumulated_epoch == inferred_epoch
        and planned_epoch > inferred_epoch
        and not has_confident_accumulated_source
    ):
        avisos.append(
            "A configuraÃ§Ã£o previa mais Ã©pocas do que o checkpoint registra, mas nÃ£o hÃ¡ metadado "
            "de checkpoint anterior nem cadeia de diretÃ³rios de retomada. Se este peso veio de retomada "
            "externa, a Ã©poca acumulada real nÃ£o Ã© recuperÃ¡vel apenas deste arquivo."
        )

    payload: dict[str, Any] = {
        "algoritmo_selecionado": algorithm,
        "arquivo": str(path),
        "nome_arquivo": path.name,
        "tamanho_bytes": stat.st_size,
        "modificado_em": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "formato_detectado": _checkpoint_format(loaded),
        "ultima_epoca": inferred_epoch,
        "fonte_ultima_epoca": epoch_source,
        "epoca_acumulada": accumulated_epoch,
        "fonte_epoca_acumulada": accumulated_source,
        "timestamp": _extract_timestamp(path, loaded, sidecar),
        "metadados_auxiliares_path": str(sidecar_path) if sidecar_path else None,
        "metadados_auxiliares": _jsonable(sidecar) if sidecar else None,
        "avisos": avisos,
    }

    if isinstance(loaded, dict):
        payload["chaves_top_level"] = sorted(str(k) for k in loaded.keys())
        payload["metadados_checkpoint"] = {
            str(k): _jsonable(v)
            for k, v in loaded.items()
            if str(k) not in STATE_KEYS
        }
        payload["estados_pesados"] = {
            str(k): _summarize_heavy_state(v)
            for k, v in loaded.items()
            if str(k) in STATE_KEYS
        }
    else:
        payload["metadados_checkpoint"] = _jsonable(loaded)

    return payload


def format_weights_metadata(metadata: dict[str, Any]) -> str:
    linhas = [
        "Metadados dos pesos",
        f"Algoritmo: {metadata.get('algoritmo_selecionado')}",
        f"Arquivo: {metadata.get('arquivo')}",
        f"Formato detectado: {metadata.get('formato_detectado')}",
        f"Última época: {metadata.get('ultima_epoca')} ({metadata.get('fonte_ultima_epoca')})",
        f"Época acumulada: {metadata.get('epoca_acumulada')} ({metadata.get('fonte_epoca_acumulada')})",
        f"Timestamp: {metadata.get('timestamp')}",
        f"Modificado em: {metadata.get('modificado_em')}",
        f"Tamanho: {metadata.get('tamanho_bytes')} bytes",
    ]
    if metadata.get("metadados_auxiliares_path"):
        linhas.append(f"Metadados auxiliares: {metadata.get('metadados_auxiliares_path')}")
    for aviso in metadata.get("avisos") or []:
        linhas.append(f"Aviso: {aviso}")
    linhas.extend(["", json.dumps(metadata, indent=2, ensure_ascii=False)])
    return "\n".join(linhas)
