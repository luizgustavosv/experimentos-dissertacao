from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from app.detectors.base import Logger
from app.detectors.config import TrainConfig
from app.detectors.dataset_coco import CocoDetectionDataset
from app.detectors.utils import coco_collate, describe_dataloader, ensure_weights_size, resolve_device, seed_everything
from app.metrics import Metrics


def _split_dataset(dataset: CocoDetectionDataset, val_ratio: float, seed: int) -> Tuple[CocoDetectionDataset, CocoDetectionDataset]:
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))


def train_torchvision_detector(
    model: torch.nn.Module,
    dataset_dir: Path,
    train_ann: Path,
    val_ann: Path,
    weights_out: Path,
    config: TrainConfig,
    logger: Optional[Logger] = None,
    val_ratio: float = 0.1,
) -> Metrics:
    device_str = resolve_device(config.device)
    seed_everything(config.seed)

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds_full = CocoDetectionDataset(dataset_dir / "images" / "train", train_ann, transforms=transform)
    val_ds_full = CocoDetectionDataset(dataset_dir / "images" / "val", val_ann, transforms=transform)

    # Dividir train em train/val adicionais se desejado
    if val_ratio > 0:
        train_ds_full, extra_val = _split_dataset(train_ds_full, val_ratio, config.seed)
        val_ds_full = extra_val

    describe_dataloader(train_ds_full, logger)

    train_loader = DataLoader(
        train_ds_full,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=coco_collate,
    )
    val_loader = DataLoader(
        val_ds_full,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=coco_collate,
    )

    model.to(device_str)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)

    last_loss = None
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device_str) for img in images]
            targets = [{k: v.to(device_str) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
        lr_scheduler.step()
        avg_loss = running_loss / max(1, len(train_loader))
        last_loss = avg_loss
        if logger:
            logger(
                f"[TRAIN] Época {epoch}/{config.epochs} | lr={lr_scheduler.get_last_lr()[0]:.6f} | loss={avg_loss:.4f} | device={device_str}"
            )

        # Avaliação básica de overfit usando loss em validação
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device_str) for img in images]
                targets = [{k: v.to(device_str) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
        val_loss = val_loss / max(1, len(val_loader))
        if logger:
            logger(f"[VAL] Época {epoch} | loss={val_loss:.4f}")

    weights_out = weights_out.expanduser().resolve()
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_out)
    ensure_weights_size(weights_out)

    return Metrics(
        precision=0.0,
        recall=0.0,
        map50=0.0,
        map50_95=0.0,
        loss_final=last_loss,
        epochs=config.epochs,
        train_images=len(train_ds_full),
        device=device_str,
        weights_path=weights_out,
        map_computed=False,
    )

