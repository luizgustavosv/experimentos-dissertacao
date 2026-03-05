from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from app.training.checkpoint_manager import CheckpointManager, get_latest_last_checkpoint


def test_checkpoint_manager_retention_and_resume() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        manager = CheckpointManager(run_dir=run_dir, prefix="torchvision", ext=".pth", keep_best=True, metric_name="map")

        metrics = [0.1, 0.3, 0.2]
        for epoch, metric in enumerate(metrics, start=1):
            payload = {
                "epoch": epoch,
                "model_state": {"w": torch.tensor([epoch], dtype=torch.float32)},
                "optimizer_state": {"lr": 0.01},
                "scheduler_state": {"step": epoch},
            }
            manager.save_last(epoch, payload)
            manager.maybe_save_best(epoch, metric, payload)
            manager.cleanup()

            last_files = sorted(run_dir.glob("torchvision_last_epoch_*.pth"))
            best_files = sorted(run_dir.glob("torchvision_best_epoch_*.pth"))
            assert len(last_files) == 1
            assert len(best_files) <= 1

        latest_last = get_latest_last_checkpoint(run_dir, "torchvision", ".pth")
        assert latest_last is not None
        loaded = torch.load(latest_last, map_location="cpu")
        next_epoch = int(loaded["epoch"]) + 1
        assert next_epoch == 4
