from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from app.training.checkpoint_manager import CheckpointManager, get_latest_last_checkpoint


def test_checkpoint_manager_retention_and_resume() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        manager = CheckpointManager(run_dir=run_dir, prefix="torchvision", ext=".pth", keep_best=True, metric_name="map")

        for epoch in (1, 2, 3):
            payload = {
                "epoch": epoch,
                "model_state_dict": {"w": torch.tensor([epoch], dtype=torch.float32)},
                "optimizer_state_dict": {"lr": 0.01},
                "scheduler_state_dict": {"step": epoch},
                "scaler_state_dict": None,
            }
            manager.save_last(epoch, payload)

            current = run_dir / f"checkpoint_epoch_{epoch}.pth"
            assert current.exists()
            assert current.stat().st_size > 0

            if epoch > 1:
                previous = run_dir / f"checkpoint_epoch_{epoch - 1}.pth"
                assert not previous.exists()

        latest_last = get_latest_last_checkpoint(run_dir, "torchvision", ".pth")
        assert latest_last is not None
        assert latest_last.name == "checkpoint_epoch_3.pth"
        loaded = torch.load(latest_last, map_location="cpu")
        next_epoch = int(loaded["epoch"]) + 1
        assert next_epoch == 4
