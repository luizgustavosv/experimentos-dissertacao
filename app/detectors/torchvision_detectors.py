from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes

from app.detectors.base import DetectionAlgorithm, DetectorContext, Logger
from app.detectors.config import TrainConfig
from app.detectors.torchvision_train import train_torchvision_detector
from app.detectors.utils import ensure_weights_size, resolve_device, validate_coco_dataset
from app.metrics import Metrics
from app.metrics import InferencePerformance
from app.reporting.reports import ReportBuilder


class TorchvisionDetector(DetectionAlgorithm):
    def __init__(self, context: DetectorContext, build_model: Callable[[int], torch.nn.Module], config: Optional[TrainConfig] = None):
        super().__init__(context)
        self.build_model = build_model
        self.config = config or TrainConfig()

    def train(
        self,
        dataset_dir: Path,
        pretrained_weights: Optional[Path],
        weights_out: Path,
        epochs: int,
        early_stop: bool,
        logger: Optional[Logger] = None,
    ) -> Metrics:
        train_ann, val_ann = validate_coco_dataset(dataset_dir)
        device_str = resolve_device(self.config.device)
        num_classes = self._infer_num_classes(train_ann)
        if logger:
            logger(f"[TRAIN] {self.context.name} em {device_str} com {num_classes} classes")
            logger(f"[DATA] Anotações train: {train_ann}")
            logger(f"[DATA] Anotações val: {val_ann}")

        model = self.build_model(num_classes)
        metrics = train_torchvision_detector(
            model,
            dataset_dir,
            train_ann,
            val_ann,
            weights_out,
            TrainConfig(
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
        )

        ensure_weights_size(weights_out)
        return metrics

    def _infer_num_classes(self, ann_path: Path) -> int:
        import json

        data = json.loads(ann_path.read_text(encoding="utf-8"))
        return len(data.get("categories", [])) + 1  # +1 para background

    def infer(
        self,
        images_dir: Path,
        weights_path: Optional[Path],
        report_out: Path,
        pedestrian_only: bool = False,
        logger: Optional[Logger] = None,
    ):
        images_dir = images_dir.expanduser().resolve()
        report_out = report_out.expanduser().resolve()
        if not images_dir.exists() or not images_dir.is_dir():
            raise FileNotFoundError(f"Pasta de imagens inexistente: {images_dir}")

        image_paths = self._list_images(images_dir)
        if not image_paths:
            raise ValueError(f"Nenhuma imagem encontrada em {images_dir}")

        device_str = resolve_device(self.config.device)
        model, class_names, weights_label = self._load_model_for_inference(weights_path, device_str, logger)

        prediction_root = report_out.parent / "predictions"
        save_dir = prediction_root / report_out.stem
        save_dir.mkdir(parents=True, exist_ok=True)

        transform = transforms.ToTensor()
        previews: List[Path] = []
        start = time.perf_counter()
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            tensor = transform(image)
            with torch.no_grad():
                outputs = model([tensor.to(device_str)])
            output = outputs[0] if outputs else {}
            boxes = output.get("boxes", torch.empty((0, 4))).detach().cpu()
            scores = output.get("scores", torch.empty(0)).detach().cpu()
            labels = output.get("labels", torch.empty(0, dtype=torch.int64)).detach().cpu()

            keep = scores >= 0.5
            if pedestrian_only:
                keep = keep & (labels == 0)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            image_uint8 = (tensor * 255).to(torch.uint8)
            if boxes.numel() > 0:
                label_texts = [self._format_label(l, s, class_names) for l, s in zip(labels, scores)]
                drawn = draw_bounding_boxes(image_uint8, boxes, labels=label_texts, colors="red", width=2)
            else:
                drawn = image_uint8

            save_path = save_dir / image_path.name
            transforms.ToPILImage()(drawn).save(save_path)
            if boxes.numel() > 0:
                previews.append(save_path)
            if logger:
                logger(f"[INFER] {image_path.name}: {len(boxes)} detecções acima de 0.5")

        elapsed = time.perf_counter() - start
        total_images = len(image_paths)
        images_per_second = total_images / elapsed if elapsed > 0 else 0.0
        milliseconds_per_image = (elapsed / total_images * 1000) if total_images else 0.0
        performance = InferencePerformance(
            images_per_second=images_per_second,
            milliseconds_per_image=milliseconds_per_image,
        )

        report_builder = ReportBuilder(self.context.name)
        report_builder.save_report(
            report_path=report_out,
            metrics=None,
            operation="Inferência",
            source_dir=images_dir,
            inference_performance=performance,
            detection_previews=previews,
            weights_path=weights_label,
        )

        if logger:
            logger(
                f"[INFER] Latência: {performance.images_per_second:.2f} img/s ({performance.milliseconds_per_image:.2f} ms/imagem)"
            )
            logger(f"[INFER] Relatório salvo em {report_out}")

        return performance

    def validate(
        self,
        dataset_yaml_path: Path,
        weights_path: Optional[Path],
        report_out: Path,
        plots_dir: Path,
        pedestrian_only: bool = False,
        logger: Optional[Logger] = None,
    ):
        raise NotImplementedError("Validação não implementada neste escopo de treino.")

    def normalize_dataset(self, dataset_type: str, dataset_dir: Path, normalized_dir: Path, logger: Optional[Logger] = None):
        raise NotImplementedError("Normalização delegada ao módulo existente.")

    def _load_model_for_inference(
        self, weights_path: Optional[Path], device_str: str, logger: Optional[Logger]
    ) -> Tuple[torch.nn.Module, Sequence[str], Optional[Path]]:
        if self.context.architecture == "SSD":
            return self._load_ssd_for_inference(weights_path, device_str, logger)
        if self.context.architecture == "Faster R-CNN":
            return self._load_faster_rcnn_for_inference(weights_path, device_str, logger)
        if self.context.architecture == "RetinaNet":
            return self._load_retinanet_for_inference(weights_path, device_str, logger)
        raise NotImplementedError(f"Inferência não implementada para {self.context.architecture} neste módulo.")

    def _load_ssd_for_inference(
        self, weights_path: Optional[Path], device_str: str, logger: Optional[Logger]
    ) -> Tuple[torch.nn.Module, Sequence[str], Optional[Path]]:
        from torchvision.models.detection import SSD300_VGG16_Weights, ssd300_vgg16

        if weights_path is None:
            weights = SSD300_VGG16_Weights.DEFAULT
            model = ssd300_vgg16(weights=weights)
            class_names: Sequence[str] = tuple(weights.meta.get("categories", []))
            weights_label: Optional[Path] = Path("SSD300_VGG16_Weights.DEFAULT")
            if logger:
                logger("[INFER] Usando pesos padrão do SSD (torchvision)")
        else:
            weights_path = weights_path.expanduser().resolve()
            if not weights_path.exists():
                raise FileNotFoundError(f"Pesos não encontrados: {weights_path}")
            state_dict = torch.load(weights_path, map_location="cpu")
            num_classes = self._infer_ssd_num_classes(state_dict, logger)
            model = ssd300_vgg16(weights=None, weights_backbone=None, num_classes=num_classes)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if logger:
                if missing:
                    logger(f"[INFER] Aviso: camadas ausentes ao carregar pesos: {missing}")
                if unexpected:
                    logger(f"[INFER] Aviso: pesos inesperados ignorados: {unexpected}")
            class_names = [str(idx) for idx in range(num_classes)]
            weights_label = weights_path
        model.to(device_str)
        model.eval()
        return model, class_names, weights_label

    def _load_faster_rcnn_for_inference(
        self, weights_path: Optional[Path], device_str: str, logger: Optional[Logger]
    ) -> Tuple[torch.nn.Module, Sequence[str], Optional[Path]]:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

        if weights_path is None:
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model = fasterrcnn_resnet50_fpn(weights=weights)
            class_names: Sequence[str] = tuple(weights.meta.get("categories", []))
            weights_label: Optional[Path] = Path("FasterRCNN_ResNet50_FPN_Weights.DEFAULT")
            if logger:
                logger("[INFER] Usando pesos padrão do Faster R-CNN (torchvision)")
        else:
            weights_path = weights_path.expanduser().resolve()
            if not weights_path.exists():
                raise FileNotFoundError(f"Pesos não encontrados: {weights_path}")
            state_dict = torch.load(weights_path, map_location="cpu")
            num_classes = self._infer_faster_rcnn_num_classes(state_dict, logger)
            model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if logger:
                if missing:
                    logger(f"[INFER] Aviso: camadas ausentes ao carregar pesos: {missing}")
                if unexpected:
                    logger(f"[INFER] Aviso: pesos inesperados ignorados: {unexpected}")
            class_names = [str(idx) for idx in range(num_classes)]
            weights_label = weights_path
        model.to(device_str)
        model.eval()
        return model, class_names, weights_label

    def _infer_ssd_num_classes(self, state_dict: dict, logger: Optional[Logger]) -> int:
        priors = self._ssd_priors_per_location()
        for idx, prior in enumerate(priors):
            key = f"head.classification_head.module_list.{idx}.bias"
            if key not in state_dict:
                continue
            bias_len = state_dict[key].numel()
            if bias_len % prior == 0:
                num_classes = bias_len // prior
                if logger:
                    logger(f"[INFER] Inferido num_classes={num_classes} a partir dos pesos (prior={prior})")
                return int(num_classes)
        if logger:
            logger("[INFER] Falha ao inferir num_classes; usando 91 (padrão COCO)")
        return 91

    @staticmethod
    def _ssd_priors_per_location() -> Sequence[int]:
        from torchvision.models.detection import ssd300_vgg16

        model = ssd300_vgg16(weights=None, weights_backbone=None, num_classes=91)
        return model.anchor_generator.num_anchors_per_location()

    def _infer_faster_rcnn_num_classes(self, state_dict: dict, logger: Optional[Logger]) -> int:
        head_bias = state_dict.get("roi_heads.box_predictor.cls_score.bias")
        if head_bias is not None:
            num_classes = head_bias.numel()
            if logger:
                logger(f"[INFER] Inferido num_classes={num_classes} a partir de roi_heads.box_predictor.cls_score.bias")
            return int(num_classes)
        if logger:
            logger("[INFER] Falha ao inferir num_classes para Faster R-CNN; usando 91 (padrão COCO)")
        return 91

    def _load_retinanet_for_inference(
        self, weights_path: Optional[Path], device_str: str, logger: Optional[Logger]
    ) -> Tuple[torch.nn.Module, Sequence[str], Optional[Path]]:
        from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights, retinanet_resnet50_fpn

        if weights_path is None:
            weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
            model = retinanet_resnet50_fpn(weights=weights)
            class_names: Sequence[str] = tuple(weights.meta.get("categories", []))
            weights_label: Optional[Path] = Path("RetinaNet_ResNet50_FPN_Weights.DEFAULT")
            if logger:
                logger("[INFER] Usando pesos padrão do RetinaNet (torchvision)")
        else:
            weights_path = weights_path.expanduser().resolve()
            if not weights_path.exists():
                raise FileNotFoundError(f"Pesos não encontrados: {weights_path}")
            state_dict = torch.load(weights_path, map_location="cpu")
            num_classes = self._infer_retinanet_num_classes(state_dict, logger)
            model = retinanet_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if logger:
                if missing:
                    logger(f"[INFER] Aviso: camadas ausentes ao carregar pesos: {missing}")
                if unexpected:
                    logger(f"[INFER] Aviso: pesos inesperados ignorados: {unexpected}")
            class_names = [str(idx) for idx in range(num_classes)]
            weights_label = weights_path
        model.to(device_str)
        model.eval()
        return model, class_names, weights_label

    def _infer_retinanet_num_classes(self, state_dict: dict, logger: Optional[Logger]) -> int:
        bias = state_dict.get("head.classification_head.cls_logits.bias")
        if bias is not None:
            anchors_per_location = self._retinanet_anchors_per_location()
            if anchors_per_location > 0 and bias.numel() % anchors_per_location == 0:
                num_classes = bias.numel() // anchors_per_location
                if logger:
                    logger(
                        f"[INFER] Inferido num_classes={num_classes} a partir de head.classification_head.cls_logits.bias "
                        f"(anchors por localização: {anchors_per_location})"
                    )
                return int(num_classes)
        if logger:
            logger("[INFER] Falha ao inferir num_classes para RetinaNet; usando 91 (padrão COCO)")
        return 91

    @staticmethod
    def _retinanet_anchors_per_location() -> int:
        from torchvision.models.detection import retinanet_resnet50_fpn

        model = retinanet_resnet50_fpn(weights=None, weights_backbone=None, num_classes=91)
        anchors_per_location = model.anchor_generator.num_anchors_per_location()
        return anchors_per_location[0] if anchors_per_location else 0

    @staticmethod
    def _format_label(label: torch.Tensor, score: torch.Tensor, class_names: Sequence[str]) -> str:
        idx = int(label.item())
        name = class_names[idx] if idx < len(class_names) else str(idx)
        return f"{name}: {score:.2f}"

    @staticmethod
    def _list_images(root: Path) -> List[Path]:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        return sorted([p for p in root.iterdir() if p.suffix.lower() in extensions and p.is_file()])
