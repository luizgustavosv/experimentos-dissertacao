from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import torch

from app.datasets.heridal_to_voc import make_split
from app.datasets.visdrone_to_voc import find_visdrone_splits
from app.detectors.base import DetectorContext, Logger
from app.detectors.config import TrainConfig
from app.detectors.dataset_voc import PascalVOCDataset
from app.detectors.torchvision_detectors import TorchvisionDetector
from app.detectors.torchvision_train import train_torchvision_detector
from app.detectors.utils import resolve_device, validate_voc_dataset
from normalize_dataset import build_imagesets_main, normalize_to_voc, validate_voc_dataset as robust_validate


class SSDDetector(TorchvisionDetector):
    def __init__(self, context: DetectorContext):
        from app.detectors.torchvision_models import build_ssd

        super().__init__(context, build_ssd)

    def train(
        self,
        dataset_dir: Path,
        pretrained_weights: Optional[Path],
        output_dir: Path,
        epochs: int,
        logger: Optional[Logger] = None,
    ):
        from torchvision import transforms

        dataset_root, class_names, train_ids, val_ids = validate_voc_dataset(dataset_dir)
        class_to_idx = {name: idx + 1 for idx, name in enumerate(class_names)}
        device_str = resolve_device(self.config.device)
        num_classes = 2 if len(class_names) == 1 else len(class_names) + 1  # +1 para background

        if logger:
            logger(f"[TRAIN] {self.context.name} em {device_str} com {num_classes} classes (Pascal VOC)")
            logger(f"[DATA] Raiz do dataset VOC: {dataset_root}")
            logger(f"[DATA] Splits: train={len(train_ids)}, val={len(val_ids)}")
            logger(f"[DATA] Classes: {', '.join(class_names)}")
            if num_classes == 2:
                logger("[SSD] num_classes_experiment=2 (background+human)")

        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = PascalVOCDataset(dataset_root, train_ids, class_to_idx, transforms=transform)
        val_dataset = PascalVOCDataset(dataset_root, val_ids, class_to_idx, transforms=transform)

        model = self._build_ssd_model(num_classes, pretrained_weights, logger)
        metrics = train_torchvision_detector(
            model,
            dataset_root,
            train_ann=None,
            val_ann=None,
            weights_out=output_dir,
            config=TrainConfig(
                epochs=epochs or self.config.epochs,
                batch_size=self.config.batch_size,
                lr=self.config.lr,
                device=self.config.device,
                num_workers=self.config.num_workers,
                seed=self.config.seed,
                weight_decay=self.config.weight_decay,
                lr_step_size=self.config.lr_step_size,
                lr_gamma=self.config.lr_gamma,
                verbose=self.config.verbose,
                log_every=self.config.log_every,
                debug_dataloader=self.config.debug_dataloader,
                log_dir=self.config.log_dir,
                pin_memory=self.config.pin_memory,
                persistent_workers=self.config.persistent_workers,
                prefetch_factor=self.config.prefetch_factor,
                drop_last=self.config.drop_last,
            ),
            logger=logger,
            val_ratio=0.0,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
        return metrics

    def normalize_dataset(self, dataset_type: str, dataset_dir: Path, normalized_dir: Path, logger: Optional[Logger] = None):
        dataset_type = dataset_type.lower()
        normalized_dir = normalized_dir.expanduser().resolve()
        dataset_dir = dataset_dir.expanduser().resolve()
        normalized_dir.mkdir(parents=True, exist_ok=True)
        processed_any = False

        def _canonical_key(name: str) -> str:
            candidate = name
            while True:
                stem = Path(candidate).stem
                if stem == candidate:
                    break
                candidate = stem
            return candidate.lower()

        if dataset_type == "heridal":
            train_dir = dataset_dir / "train"
            annotations_dir = train_dir
            if not train_dir.exists():
                raise FileNotFoundError(f"Diretório esperado não encontrado para HERIDAL: {train_dir}")
            images = [p for p in train_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
            splits = make_split([p.name for p in images], train_ratio=0.8, val_ratio=0.2, test_ratio=0.0)
            class_map = {"*": "human"}
            for split_name in ("train", "val"):
                selected = {_canonical_key(name) for name, s in splits.items() if s == split_name}
                normalize_to_voc(
                    src_images_dir=str(train_dir),
                    src_annotations_dir=str(annotations_dir),
                    out_dir=str(normalized_dir),
                    split=split_name,
                    class_map=class_map,
                    keep_skipped=True,
                    selected_original_stems=selected,
                    logger=logger,
                )
                processed_any = True
        else:
            visdrone_splits = find_visdrone_splits(dataset_dir)
            has_visdrone_structure = any(visdrone_splits.values())
            if dataset_type == "visdrone" or has_visdrone_structure:
                for split_name in ("train", "val"):
                    split_info = visdrone_splits.get(split_name)
                    if not split_info:
                        continue
                    images_dir, annotations_dir = split_info
                    normalize_to_voc(
                        src_images_dir=str(images_dir),
                        src_annotations_dir=str(annotations_dir),
                        out_dir=str(normalized_dir),
                        split=split_name,
                        class_map={"*": "human"},
                        keep_skipped=True,
                        logger=logger,
                    )
                    processed_any = True
            else:
                raise ValueError("Normalização SSD implementada apenas para os datasets HERIDAL ou VisDrone.")
        if not processed_any:
            raise ValueError("Nenhum split foi processado; verifique a estrutura do dataset de entrada.")
        state_path = normalized_dir / "normalization_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            splits_state = state.get("splits", {})
            if (not splits_state.get("val")) and splits_state.get("train"):
                train_ids = list(splits_state["train"])
                if len(train_ids) == 1:
                    val_ids = [train_ids[0]]
                    train_ids = []
                else:
                    cut = max(1, len(train_ids) // 5)
                    val_ids = train_ids[-cut:]
                    train_ids = train_ids[:-cut]
                splits_state["train"] = train_ids
                splits_state["val"] = val_ids
                state["splits"] = splits_state
                state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
                build_imagesets_main(normalized_dir, splits_state)
        robust_validate(normalized_dir, fix=True, logger=logger)
        if logger:
            logger(f"[SSD][NORM] Dataset pronto em {normalized_dir}")
        return normalized_dir

    def _build_ssd_model(self, num_classes: int, pretrained_weights: Optional[Path], logger: Optional[Logger]):
        from torchvision.models import VGG16_Weights
        from torchvision.models.detection import SSD300_VGG16_Weights, ssd300_vgg16

        mode, checkpoint_path = self._resolve_ssd_pretrained_mode(pretrained_weights)

        weights: Optional[SSD300_VGG16_Weights] = None
        weights_backbone: Optional[VGG16_Weights] = None
        state_dict: Optional[dict] = None
        checkpoint_num_classes: Optional[int] = None
        checkpoint_label = "<none>"

        if mode == "ssd_coco":
            weights = SSD300_VGG16_Weights.DEFAULT
            checkpoint_label = "SSD300_VGG16_Weights.DEFAULT"
        elif mode == "backbone_imagenet":
            weights_backbone = VGG16_Weights.DEFAULT
            checkpoint_label = "VGG16_Weights.DEFAULT"
        elif mode == "from_checkpoint_path":
            state_dict, checkpoint_num_classes, checkpoint_label = self._load_checkpoint_state(checkpoint_path, logger)
        else:  # pragma: no cover - proteção extra para casos não previstos
            raise ValueError(f"Modo de pré-treino SSD desconhecido: {mode}")

        if logger:
            logger(f"[SSD][INIT] mode={mode} weights={weights} weights_backbone={weights_backbone}")
            logger(
                f"[SSD][INIT] weights_type={type(weights).__name__ if weights is not None else 'None'} | "
                f"weights_backbone_type={type(weights_backbone).__name__ if weights_backbone is not None else 'None'}"
            )

        constructor_num_classes = num_classes if weights is None else None
        model = ssd300_vgg16(weights=weights, weights_backbone=weights_backbone, num_classes=constructor_num_classes)

        if logger:
            logger(f"[SSD][WEIGHTS] checkpoint_path={checkpoint_label}")
            logger(
                f"[SSD][WEIGHTS] checkpoint_num_classes={checkpoint_num_classes if checkpoint_num_classes else 'desconhecido'}"
            )
            logger(f"[SSD][WEIGHTS] num_classes_model={num_classes}")

        if state_dict:
            drop_heads = checkpoint_num_classes is None or checkpoint_num_classes != num_classes
            filtered_state_dict, removed_keys = self._filter_ssd_state_dict(state_dict, drop_heads=drop_heads)
            missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
            loaded_keys = [k for k in filtered_state_dict.keys() if k not in unexpected]
            loaded_params = sum(filtered_state_dict[k].numel() for k in loaded_keys if hasattr(filtered_state_dict[k], "numel"))
            if logger:
                logger(
                    "[SSD][WEIGHTS] política de carregamento: "
                    f"{'descartando heads incompatíveis' if drop_heads else 'mantendo heads do checkpoint'}"
                )
                if removed_keys:
                    logger(f"[SSD][WEIGHTS] Heads removidos do checkpoint: {removed_keys}")
                logger(
                    f"[SSD][WEIGHTS] parâmetros carregados={loaded_params} | missing={len(missing)} | unexpected={len(unexpected)}"
                )
                if missing:
                    logger(f"[SSD][WEIGHTS] missing_keys: {missing}")
                if unexpected:
                    logger(f"[SSD][WEIGHTS] unexpected_keys: {unexpected}")
        else:
            if logger:
                logger("[SSD][WEIGHTS] Nenhum checkpoint aplicado; prosseguindo apenas com backbone pré-treinado")

        return model

    @staticmethod
    def _resolve_ssd_pretrained_mode(pretrained_weights: Optional[Path]) -> Tuple[str, Optional[Path]]:
        if pretrained_weights is None:
            return "ssd_coco", None

        lower_value = str(pretrained_weights).strip().lower()
        if lower_value == "ssd_coco":
            return "ssd_coco", None
        if lower_value in {"backbone_imagenet", "imagenet"}:
            return "backbone_imagenet", None

        as_path = Path(pretrained_weights)
        return "from_checkpoint_path", as_path

    def _load_checkpoint_state(self, pretrained_weights: Path, logger: Optional[Logger]) -> tuple[Optional[dict], Optional[int], str]:
        weights_path = pretrained_weights.expanduser().resolve()
        if not weights_path.exists():
            raise FileNotFoundError(f"Pesos não encontrados: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
        checkpoint_num_classes = self._infer_ssd_num_classes(state_dict, logger)
        return state_dict, checkpoint_num_classes, str(weights_path)

    @staticmethod
    def _filter_ssd_state_dict(state_dict: dict, drop_heads: bool):
        head_markers = (
            "head.classification_head",
            "head.regression_head",
            "cls_headers",
            "box_headers",
            ".cls.",
            ".cls_",
            ".conf",
            ".bbox",
            ".box",
            ".reg",
            ".loc",
        )
        filtered = {}
        removed = []
        for key, value in state_dict.items():
            lower_key = key.lower()
            if lower_key.startswith("backbone."):
                filtered[key] = value
                continue
            if drop_heads and any(marker in lower_key for marker in head_markers):
                removed.append(key)
                continue
            filtered[key] = value
        return filtered, removed
