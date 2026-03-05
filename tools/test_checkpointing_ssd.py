from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import types
if "ultralytics" not in sys.modules:
    ultralytics_stub = types.ModuleType("ultralytics")
    class _FakeYOLO:  # pragma: no cover - apenas para evitar dependência de OpenCV no smoke test SSD
        def __init__(self, *args, **kwargs):
            raise RuntimeError("YOLO não é usado neste smoke test SSD")
    ultralytics_stub.YOLO = _FakeYOLO
    sys.modules["ultralytics"] = ultralytics_stub

import torch
from torch.utils.data import Dataset
from torchvision.models.detection import ssd300_vgg16

from app.detectors.config import TrainConfig
from app.detectors.torchvision_train import train_torchvision_detector


class TinySSDDataset(Dataset):
    def __init__(self, length: int = 2) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        image = torch.rand(3, 300, 300, dtype=torch.float32)
        boxes = torch.tensor([[30.0, 40.0, 200.0, 250.0]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": torch.tensor([(200.0 - 30.0) * (250.0 - 40.0)], dtype=torch.float32),
            "iscrowd": torch.zeros((1,), dtype=torch.int64),
            "img_path": f"tiny_{idx}.jpg",
        }
        return image, target



def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test de checkpoint para SSD torchvision")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ssd_ckpt_smoke"))
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = ssd300_vgg16(weights=None, weights_backbone=None, num_classes=2)
    train_dataset = TinySSDDataset(length=2)
    val_dataset = TinySSDDataset(length=1)

    config = TrainConfig(
        epochs=3,
        batch_size=1,
        lr=1e-4,
        device="cpu",
        num_workers=0,
        verbose=True,
        log_every=1,
        log_dir=output_dir / "logs",
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        drop_last=False,
        audit_datasets=False,
        val_mode="loss",
        early_stop_enabled=False,
        save_best=True,
    )

    train_torchvision_detector(
        model=model,
        dataset_dir=output_dir,
        train_ann=None,
        val_ann=None,
        weights_out=output_dir,
        config=config,
        logger=print,
        val_ratio=0.0,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        ssd_class_info={
            "class_names": ["human"],
            "dataset_num_classes": 1,
            "class_source": "smoke_test",
            "background_removed": False,
            "background_before": ["human"],
            "logged": False,
        },
    )

    ckpt_dir = (output_dir / "checkpoints").resolve()
    last_files = sorted(ckpt_dir.glob("ssd_last_epoch_*.pth"))
    if not last_files:
        print(f"[SMOKE][ERROR] Nenhum checkpoint last encontrado em {ckpt_dir}", file=sys.stderr)
        return 1

    print(f"[SMOKE] checkpoint encontrado: {last_files[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
