from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from app.detectors.utils import atomic_torch_save


@dataclass
class CheckpointPolicy:
    save_final: bool = True
    save_best: bool = True
    save_every: int = 10
    keep_last_k: int = 3
    monitor_metric: str = "val_map"
    mode: str = "max"


class CheckpointManager:
    def __init__(
        self,
        output_dir: Path,
        policy: CheckpointPolicy,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.output_dir = output_dir.expanduser().resolve()
        self.policy = policy
        self.logger = logger or logging.getLogger(__name__)
        self.weights_dir = self.output_dir / "weights"
        self.checkpoints_dir = self.weights_dir / "checkpoints"
        self.best_path = self.weights_dir / "best.pth"
        self.final_path = self.weights_dir / "final.pth"
        self.best_value: Optional[float] = None
        self._fallback_warned = False

        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "[CHECKPOINT] policy save_every=%s keep_last_k=%s monitor_metric=%s mode=%s",
            self.policy.save_every,
            self.policy.keep_last_k,
            self.policy.monitor_metric,
            self.policy.mode,
        )

        if self.policy.save_best and self.best_path.exists():
            self._load_best_value_from_disk()

    def _load_best_value_from_disk(self) -> None:
        try:
            payload = torch.load(self.best_path, map_location="cpu")
            metrics = payload.get("metrics") if isinstance(payload, dict) else None
            if isinstance(metrics, dict):
                value = metrics.get(self.policy.monitor_metric)
                if value is None:
                    value = metrics.get("val_loss")
                if value is not None and math.isfinite(float(value)):
                    self.best_value = float(value)
        except Exception as exc:  # pragma: no cover - fallback defensivo
            self.logger.warning("[CHECKPOINT] Não foi possível ler best.pth existente: %s", exc)

    def _resolve_monitor_value(self, metrics: dict[str, Any]) -> tuple[str, Optional[float], str]:
        metric_name = self.policy.monitor_metric
        value = metrics.get(metric_name)
        if value is None or not math.isfinite(float(value)):
            metric_name = "val_loss"
            value = metrics.get(metric_name)
            mode = "min"
            if not self._fallback_warned:
                self.logger.warning(
                    "[CHECKPOINT] Métrica %s indisponível. Usando fallback %s com mode=%s.",
                    self.policy.monitor_metric,
                    metric_name,
                    mode,
                )
                self._fallback_warned = True
        else:
            mode = self.policy.mode
        if value is None or not math.isfinite(float(value)):
            return metric_name, None, mode
        return metric_name, float(value), mode

    @staticmethod
    def _is_improved(value: float, best: Optional[float], mode: str) -> bool:
        if best is None:
            return True
        if mode == "min":
            return value < best
        return value > best

    def _build_state(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        metrics: dict[str, Any],
        config: dict[str, Any],
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": config,
        }
        if extra:
            state.update(extra)
        return state

    def save_best_if_improved(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        metrics: dict[str, Any],
        config: dict[str, Any],
        extra: Optional[dict[str, Any]] = None,
    ) -> bool:
        if not self.policy.save_best:
            return False
        metric_name, value, mode = self._resolve_monitor_value(metrics)
        if value is None:
            return False
        if not self._is_improved(value, self.best_value, mode):
            return False
        state = self._build_state(epoch, model, optimizer, scheduler, metrics, config, extra)
        atomic_torch_save(state, self.best_path)
        self.best_value = value
        self.logger.info(
            "[CHECKPOINT] best salvo em %s (monitor=%s value=%.4f)",
            self.best_path,
            metric_name,
            value,
        )
        return True

    def save_periodic(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        metrics: dict[str, Any],
        config: dict[str, Any],
        extra: Optional[dict[str, Any]] = None,
    ) -> Optional[Path]:
        if self.policy.save_every <= 0:
            return None
        if epoch % self.policy.save_every != 0:
            return None
        ckpt_path = self.checkpoints_dir / f"ckpt_epoch_{epoch:03d}.pth"
        state = self._build_state(epoch, model, optimizer, scheduler, metrics, config, extra)
        atomic_torch_save(state, ckpt_path)
        self.logger.info("[CHECKPOINT] ckpt periódico salvo em %s", ckpt_path)
        self.cleanup_old_checkpoints()
        return ckpt_path

    def save_final(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        metrics: dict[str, Any],
        config: dict[str, Any],
        extra: Optional[dict[str, Any]] = None,
    ) -> Optional[Path]:
        if not self.policy.save_final:
            return None
        state = self._build_state(epoch, model, optimizer, scheduler, metrics, config, extra)
        atomic_torch_save(state, self.final_path)
        self.logger.info("[CHECKPOINT] final salvo em %s", self.final_path)
        return self.final_path

    def cleanup_old_checkpoints(self) -> None:
        pattern = re.compile(r"ckpt_epoch_(\d+)\.pth$")
        candidates: list[tuple[int, Path]] = []
        for path in self.checkpoints_dir.glob("ckpt_epoch_*.pth"):
            match = pattern.search(path.name)
            if match:
                candidates.append((int(match.group(1)), path))
        candidates.sort(key=lambda item: item[0])
        if self.policy.keep_last_k <= 0:
            to_delete = candidates
        else:
            to_delete = candidates[:-self.policy.keep_last_k] if len(candidates) > self.policy.keep_last_k else []
        for _, path in to_delete:
            try:
                path.unlink()
                self.logger.info("[CHECKPOINT] ckpt antigo removido: %s", path.name)
            except Exception as exc:  # pragma: no cover - defensivo
                self.logger.warning("[CHECKPOINT] Falha ao remover %s: %s", path.name, exc)
