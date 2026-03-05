from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch


class CheckpointManager:
    def __init__(
        self,
        run_dir: Path,
        prefix: str,
        ext: str,
        keep_best: bool = True,
        metric_name: str = "metric",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.prefix = prefix.strip()
        self.ext = ext if ext.startswith(".") else f".{ext}"
        self.keep_best = keep_best
        self.metric_name = re.sub(r"[^a-zA-Z0-9_]+", "_", metric_name.strip().lower()) or "metric"
        self.logger = logger or logging.getLogger(__name__)

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.run_dir / f"{self.prefix}_ckpt_metadata.json"
        self.best_metric: Optional[float] = None
        self.best_path: Optional[Path] = None
        self.best_epoch: Optional[int] = None

        self._load_metadata()

    def _log(self, msg: str, *args: Any) -> None:
        self.logger.info("[CKPT] " + msg, *args)

    def _write_atomic_bytes(self, target: Path, data: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as tmp_file:
                tmp_file.write(data)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_path, target)
            if not target.exists() or target.stat().st_size <= 0:
                raise IOError(f"Arquivo inválido após salvar: {target}")
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _atomic_torch_save(self, payload: dict[str, Any], target: Path) -> Path:
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=target.parent, prefix=f".{target.name}.", suffix=".tmp") as tmp:
            tmp_path = Path(tmp.name)
            torch.save(payload, tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, target)
        if not target.exists() or target.stat().st_size <= 0:
            raise IOError(f"Arquivo inválido após salvar: {target}")
        return target

    def _load_metadata(self) -> None:
        if not self.metadata_path.exists():
            return
        try:
            meta = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return
        best_metric = meta.get("best_metric")
        if best_metric is not None:
            try:
                self.best_metric = float(best_metric)
            except Exception:
                self.best_metric = None
        best_path = meta.get("best_path")
        if best_path:
            self.best_path = Path(best_path)
        best_epoch = meta.get("best_epoch")
        if isinstance(best_epoch, int):
            self.best_epoch = best_epoch

    def _save_metadata(self, *, last_epoch: Optional[int] = None, last_path: Optional[Path] = None) -> None:
        previous = {}
        if self.metadata_path.exists():
            try:
                previous = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except Exception:
                previous = {}
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prefix": self.prefix,
            "ext": self.ext,
            "last_epoch": last_epoch if last_epoch is not None else previous.get("last_epoch"),
            "last_path": str(last_path) if last_path else previous.get("last_path"),
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "best_path": str(self.best_path) if self.best_path else None,
        }
        self._write_atomic_bytes(self.metadata_path, json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"))

    def save_last(self, epoch: int, payload: dict[str, Any]) -> Path:
        path = self.run_dir / f"{self.prefix}_last_epoch_{epoch:04d}{self.ext}"
        saved = self._atomic_torch_save(payload, path)
        self._save_metadata(last_epoch=epoch, last_path=saved)
        self._log("saved last: %s", saved)
        return saved

    def maybe_save_best(self, epoch: int, metric_value: Optional[float], payload: dict[str, Any]) -> Optional[Path]:
        if not self.keep_best or metric_value is None:
            return None
        try:
            metric_float = float(metric_value)
        except Exception:
            return None
        if not math.isfinite(metric_float):
            return None
        if self.best_metric is not None and metric_float <= self.best_metric:
            return None

        filename = f"{self.prefix}_best_epoch_{epoch:04d}_{self.metric_name}_{metric_float:.4f}{self.ext}"
        path = self.run_dir / filename
        saved = self._atomic_torch_save(payload, path)
        self.best_metric = metric_float
        self.best_epoch = int(epoch)
        self.best_path = saved
        self._save_metadata()
        self._log("saved best: %s", saved)
        return saved

    def cleanup(self) -> None:
        keep_paths: set[Path] = {self.metadata_path}
        latest_last = get_latest_last_checkpoint(self.run_dir, self.prefix, self.ext)
        if latest_last:
            keep_paths.add(latest_last)

        if self.best_path and self.best_path.exists():
            keep_paths.add(self.best_path)
        else:
            discovered_best = sorted(self.run_dir.glob(f"{self.prefix}_best_epoch_*{self.ext}"))
            if discovered_best:
                keep_paths.add(discovered_best[-1])

        deleted = 0
        for path in self.run_dir.iterdir():
            if not path.is_file():
                continue
            if path in keep_paths:
                continue
            if path.suffix not in {".pt", ".pth"}:
                continue
            name = path.name.lower()
            if not (
                name.startswith(f"{self.prefix}_")
                or name.startswith("epoch")
                or name.startswith("checkpoint")
                or name.startswith("model_epoch")
                or name in {"last.pt", "best.pt", "last.pth", "best.pth"}
            ):
                continue
            path.unlink(missing_ok=True)
            deleted += 1
        self._log("deleted %d old files", deleted)


def get_latest_last_checkpoint(run_dir: Path, prefix: str, ext: str) -> Optional[Path]:
    run_dir = Path(run_dir).expanduser().resolve()
    ext = ext if ext.startswith(".") else f".{ext}"
    pattern = re.compile(rf"^{re.escape(prefix)}_last_epoch_(\d+){re.escape(ext)}$")
    latest: Optional[tuple[int, Path]] = None
    for path in run_dir.glob(f"{prefix}_last_epoch_*{ext}"):
        match = pattern.match(path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        if latest is None or epoch > latest[0]:
            latest = (epoch, path)
    return latest[1] if latest else None
