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
from app.detectors.torchvision_models import build_ssd
from app.detectors.utils import (
    ensure_weights_size,
    extract_checkpoint_meta,
    extract_checkpoint_state,
    filter_torchvision_predictions,
    infer_ssd_num_classes,
    load_ssd_weights,
    resolve_device,
    resolve_ssd_run_config,
    validate_coco_dataset,
)
from app.metrics import Metrics
from app.metrics import InferencePerformance
from app.reporting.reports import ReportBuilder

SSD_EVAL_ALIGNED_SCORE_THRESHOLD = 0.05
SSD_PEDESTRIAN_LABEL_ID = 1


class TorchvisionDetector(DetectionAlgorithm):
    def __init__(self, context: DetectorContext, build_model: Callable[[int], torch.nn.Module], config: Optional[TrainConfig] = None):
        super().__init__(context)
        self.build_model = build_model
        self.config = config or TrainConfig()

    def _prepare_model(
        self, num_classes: int, pretrained_weights: Optional[Path], logger: Optional[Logger]
    ) -> torch.nn.Module:
        return self.build_model(num_classes)

    def _infer_dataset_num_classes(self, ann_path: Path, logger: Optional[Logger]) -> int:
        import json

        data = json.loads(ann_path.read_text(encoding="utf-8"))
        dataset_num_classes = len(data.get("categories", []))
        if logger:
            logger(f"[DATASET] COCO categories={dataset_num_classes}")
        return dataset_num_classes

    def _map_model_num_classes(self, dataset_num_classes: int) -> int:
        return dataset_num_classes + 1  # background

    def _log_model_num_classes(self, model_num_classes: int, logger: Optional[Logger]) -> None:
        if logger:
            logger(f"[MODEL] {self.context.name} num_classes={model_num_classes} (inclui background)")

    def train(
        self,
        dataset_dir: Path,
        pretrained_weights: Optional[Path],
        output_dir: Path,
        epochs: int,
        logger: Optional[Logger] = None,
        train_ann: Optional[Path] = None,
        val_ann: Optional[Path] = None,
    ) -> Metrics:
        if train_ann is None or val_ann is None:
            train_ann, val_ann = validate_coco_dataset(dataset_dir)
        device_str = resolve_device(self.config.device)
        dataset_num_classes = self._infer_dataset_num_classes(train_ann, logger)
        model_num_classes = self._map_model_num_classes(dataset_num_classes)
        if logger:
            logger(f"[TRAIN] {self.context.name} em {device_str}")
            logger(f"[DATA] Anotações train: {train_ann}")
            logger(f"[DATA] Anotações val: {val_ann}")
        self._log_model_num_classes(model_num_classes, logger)

        model = self._prepare_model(model_num_classes, pretrained_weights, logger)
        metrics = train_torchvision_detector(
            model,
            dataset_dir,
            train_ann,
            val_ann,
            output_dir,
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
                val_mode=self.config.val_mode,
                dataset_num_classes=dataset_num_classes,
                num_classes=model_num_classes,
                max_epochs=self.config.max_epochs,
                early_stop_enabled=self.config.early_stop_enabled,
                early_stop_patience=self.config.early_stop_patience,
                early_stop_min_delta=self.config.early_stop_min_delta,
                early_stop_min_epochs=self.config.early_stop_min_epochs,
                early_stop_ema_alpha=self.config.early_stop_ema_alpha,
                save_final=self.config.save_final,
                save_best=self.config.save_best,
                save_every=self.config.save_every,
                keep_last_k=self.config.keep_last_k,
                monitor_metric=self.config.monitor_metric,
                mode=self.config.mode,
            ),
            logger=logger,
        )

        ensure_weights_size(output_dir)
        return metrics

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
        ssd_score_threshold = (
            self._resolve_ssd_score_threshold(weights_path=weights_path, logger=logger)
            if self.context.architecture == "SSD"
            else 0.5
        )
        pedestrian_label = (
            self._resolve_pedestrian_label_id(class_names, logger=logger) if self.context.architecture == "SSD" else 0
        )
        if logger and self.context.architecture == "SSD":
            logger(
                f"[SSD][INFER] class_names={list(class_names)} pedestrian_label_id={pedestrian_label} "
                f"pedestrian_only={pedestrian_only} score_threshold={ssd_score_threshold:.4f}"
            )

        prediction_root = report_out.parent / "predictions"
        save_dir = prediction_root / report_out.stem
        save_dir.mkdir(parents=True, exist_ok=True)

        transform = transforms.ToTensor()
        previews: List[Path] = []
        start = time.perf_counter()
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            tensor = transform(image)
            with torch.inference_mode():
                outputs = model([tensor.to(device_str)])
            output = outputs[0] if outputs else {}
            filtered_output, diag = filter_torchvision_predictions(
                output,
                score_threshold=ssd_score_threshold,
                target_label=pedestrian_label if pedestrian_only else None,
            )
            boxes = filtered_output["boxes"]
            scores = filtered_output["scores"]
            labels = filtered_output["labels"]

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
                if self.context.architecture == "SSD":
                    score_range = (
                        f"[{diag['score_min']:.4f}, {diag['score_max']:.4f}]"
                        if diag["score_min"] is not None
                        else "n/a"
                    )
                    logger(
                        f"[SSD][INFER][DIAG] {image_path.name}: raw={diag['raw_count']} "
                        f"after_score={diag['after_score']} after_class={diag['after_class']} final={diag['final_count']} "
                        f"score_threshold={ssd_score_threshold:.4f} pedestrian_only={pedestrian_only} "
                        f"pedestrian_label_id={pedestrian_label} labels_before={diag['unique_labels_before']} "
                        f"labels_after={diag['unique_labels_after']} score_range={score_range}"
                    )
                else:
                    logger(
                        f"[INFER] {image_path.name}: {diag['final_count']} detecções "
                        f"(raw={diag['raw_count']}, score_threshold={ssd_score_threshold}, pedestrian_only={pedestrian_only})"
                    )

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
        from torchvision.models.detection import SSD300_VGG16_Weights

        if weights_path is None:
            weights = SSD300_VGG16_Weights.DEFAULT
            model = build_ssd(num_classes=91)
            model.load_state_dict(weights.get_state_dict(progress=True), strict=True)
            class_names: Sequence[str] = tuple(weights.meta.get("categories", []))
            weights_label: Optional[Path] = Path("SSD300_VGG16_Weights.DEFAULT")
            if logger:
                logger("[INFER] Usando pesos padrão do SSD (torchvision)")
        else:
            weights_path = weights_path.expanduser().resolve()
            if not weights_path.exists():
                raise FileNotFoundError(f"Pesos não encontrados: {weights_path}")

            loaded = torch.load(weights_path, map_location="cpu")
            top_keys = sorted(loaded.keys()) if isinstance(loaded, dict) else []
            if logger:
                logger(f"[SSD][INFER] checkpoint={weights_path}")
                logger(f"[SSD][INFER] checkpoint_type={type(loaded).__name__}")
                if top_keys:
                    logger(f"[SSD][INFER] checkpoint_keys={top_keys}")

            args_info = resolve_ssd_run_config(weights_path, logger=logger)
            meta = extract_checkpoint_meta(loaded)
            state_dict, checkpoint_format = extract_checkpoint_state(loaded)
            ckpt_num_classes = infer_ssd_num_classes(state_dict, logger=logger)
            num_classes = self._resolve_ssd_num_classes(
                args_info=args_info,
                meta=meta,
                ckpt_num_classes=ckpt_num_classes,
                logger=logger,
            )
            model = build_ssd(num_classes=num_classes)
            load_info = load_ssd_weights(
                model,
                weights_path,
                torch.device("cpu"),
                strict=True,
                strict_head=True,
                expected_num_classes=num_classes,
                loaded=loaded,
                logger=logger,
            )
            if logger:
                logger(
                    f"[SSD][INFER] checkpoint_format={checkpoint_format} num_classes_resolved={num_classes} "
                    f"num_classes_ckpt={ckpt_num_classes if ckpt_num_classes is not None else 'desconhecido'}"
                )
                logger(
                    f"[SSD][INFER] strict={load_info.get('strict_load')} "
                    f"missing_keys={len(load_info.get('missing', []))} unexpected_keys={len(load_info.get('unexpected', []))}"
                )
            class_names = self._resolve_ssd_class_names(meta, num_classes)
            weights_label = weights_path
        model.to(device_str)
        model.eval()
        return model, class_names, weights_label

    @staticmethod
    def _resolve_ssd_num_classes(
        *, args_info: dict, meta: dict, ckpt_num_classes: Optional[int], logger: Optional[Logger]
    ) -> int:
        candidates: List[Tuple[str, Optional[int]]] = [
            ("args.model_num_classes", args_info.get("model_num_classes") if isinstance(args_info, dict) else None),
            ("meta.model_num_classes", meta.get("model_num_classes") if isinstance(meta, dict) else None),
            ("meta.dataset_num_classes+1", (meta.get("dataset_num_classes") + 1) if isinstance(meta.get("dataset_num_classes"), int) else None),
            ("checkpoint_state_dict", ckpt_num_classes),
            (
                "args.dataset_num_classes+1",
                (args_info.get("dataset_num_classes") + 1)
                if isinstance(args_info, dict) and isinstance(args_info.get("dataset_num_classes"), int)
                else None,
            ),
        ]
        for source, value in candidates:
            if isinstance(value, int) and value > 1:
                if logger:
                    logger(f"[SSD][INFER] num_classes={value} via {source}")
                return value
        if logger:
            logger("[SSD][INFER][WARN] Não foi possível resolver num_classes a partir do checkpoint/meta; fallback=91")
        return 91

    @staticmethod
    def _resolve_ssd_class_names(meta: dict, num_classes: int) -> Sequence[str]:
        if isinstance(meta, dict):
            class_names = meta.get("class_names")
            if isinstance(class_names, list) and class_names and all(isinstance(name, str) for name in class_names):
                if len(class_names) == num_classes - 1:
                    return ["background", *class_names]
                if len(class_names) == num_classes:
                    return class_names
        return ["background", "human"] if num_classes == 2 else [str(idx) for idx in range(num_classes)]

    @staticmethod
    def _resolve_pedestrian_label_id(class_names: Sequence[str], logger: Optional[Logger] = None) -> int:
        if len(class_names) > SSD_PEDESTRIAN_LABEL_ID:
            if logger:
                logger(
                    f"[SSD][INFER] pedestrian_label_id fixado em {SSD_PEDESTRIAN_LABEL_ID} "
                    "(convenção SSD/VOC com background=0)."
                )
            return SSD_PEDESTRIAN_LABEL_ID

        aliases = {"human", "pedestrian", "person", "people"}
        for idx, name in enumerate(class_names):
            if str(name).strip().lower() in aliases:
                if logger:
                    logger(
                        f"[SSD][INFER][WARN] convenção SSD label=1 indisponível; "
                        f"usando alias '{name}' no índice {idx}."
                    )
                return idx
        fallback = 1 if len(class_names) > 1 else 0
        if logger:
            logger(
                f"[SSD][INFER][WARN] classe de pedestre não encontrada por alias; "
                f"fallback para índice {fallback}."
            )
        return fallback

    def _resolve_ssd_score_threshold(self, weights_path: Optional[Path], logger: Optional[Logger]) -> float:
        args_info: dict = {}
        if weights_path is not None:
            args_info = resolve_ssd_run_config(weights_path.expanduser().resolve(), logger=logger)

        threshold_candidates = [args_info.get("score_threshold") if isinstance(args_info, dict) else None]
        for candidate in threshold_candidates:
            if isinstance(candidate, (int, float)) and 0.0 <= float(candidate) <= 1.0:
                return float(candidate)

        ignored_conf = args_info.get("conf_threshold") if isinstance(args_info, dict) else None
        if logger and isinstance(ignored_conf, (int, float)):
            logger(
                "[SSD][INFER][WARN] conf_threshold encontrado em args.yaml, "
                "mas não será usado na inferência SSD para manter alinhamento com avaliação dedicada."
            )
        if logger:
            logger(
                "[SSD][INFER] score_threshold padrão alinhado à avaliação SSD dedicada: "
                f"{SSD_EVAL_ALIGNED_SCORE_THRESHOLD:.4f}."
            )
        return SSD_EVAL_ALIGNED_SCORE_THRESHOLD

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
            try:
                model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)
            except TypeError:
                model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
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
