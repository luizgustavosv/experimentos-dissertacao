from __future__ import annotations

from app.detectors.base import DetectorContext
from app.detectors.faster_rcnn import FasterRCNNDetector
from app.detectors.retinanet_eval import (
    _load_retinanet_weights_with_head_guard,
    validate_retinanet_post_train,
)
from app.detectors.torchvision_detectors import TorchvisionDetector
from app.detectors.torchvision_models import build_retinanet
from app.detectors.utils import validate_coco_dataset


class RetinaNetDetector(FasterRCNNDetector):
    """Compartilha a normalização COCO com o Faster R-CNN, mas com arquitetura RetinaNet."""

    def __init__(self, context: DetectorContext):
        TorchvisionDetector.__init__(self, context, build_retinanet)

    def validate_trained_weights(
        self,
        dataset_dir,
        weights_path,
        train_ann=None,
        val_ann=None,
        logger=None,
        log_cb=None,
        val_mode=None,
    ) -> dict:
        dataset_dir = dataset_dir.expanduser().resolve()
        if train_ann is None or val_ann is None:
            train_ann, val_ann = validate_coco_dataset(dataset_dir)

        def _build_model(num_classes: int):
            return self._prepare_model(num_classes, None, logger)

        return validate_retinanet_post_train(
            _build_model,
            dataset_dir,
            train_ann=train_ann,
            val_ann=val_ann,
            weights_path=weights_path,
            config=self.config,
            logger=logger,
            log_cb=log_cb,
        )

    def _prepare_model(self, num_classes: int, pretrained_weights, logger):  # type: ignore[override]
        # Usa o construtor configurado para RetinaNet em vez da implementação do Faster R-CNN.
        model = TorchvisionDetector._prepare_model(self, num_classes, pretrained_weights, logger)

        if pretrained_weights:
            import torch
            import logging

            weights_path = pretrained_weights.expanduser().resolve()
            if weights_path.is_file():
                state_dict = torch.load(weights_path, map_location="cpu")
                logging_logger = logging.getLogger("retinanet.weights")
                missing, unexpected, ckpt_num_classes, head_mismatch, ignored_keys, strict_load = (
                    _load_retinanet_weights_with_head_guard(
                        model,
                        state_dict,
                        model_num_classes=num_classes,
                        emit_info=logger or (lambda msg: None),
                        emit_warning=logger or (lambda msg: None),
                        logging_logger=logging_logger,
                    )
                )
                if logger:
                    if missing:
                        logger(f"[WEIGHTS] Chaves ausentes ao carregar RetinaNet: {missing}")
                    if unexpected:
                        logger(f"[WEIGHTS] Chaves ignoradas (head antigo): {unexpected}")
                    if ignored_keys:
                        logger(f"[WEIGHTS] Head cls_logits ignorado: {ignored_keys} (strict={strict_load})")
                    if head_mismatch:
                        logger(
                            "[RETINANET][CLASSES] Checkpoint head mismatch due to num_classes change. Please retrain RetinaNet."
                        )
                    if ckpt_num_classes is not None:
                        logger(f"[RETINANET][CLASSES] ckpt_num_classes={ckpt_num_classes}")
        return model

    def _map_model_num_classes(self, dataset_num_classes: int) -> int:  # type: ignore[override]
        return dataset_num_classes

    def _log_model_num_classes(self, model_num_classes: int, logger):  # type: ignore[override]
        if logger:
            logger(f"[MODEL] RetinaNet num_classes={model_num_classes} (foreground only)")

    def _infer_dataset_num_classes(self, ann_path, logger):  # type: ignore[override]
        dataset_num_classes = super()._infer_dataset_num_classes(ann_path, logger)
        self._audit_annotation_labels(ann_path, logger)
        return dataset_num_classes

    @staticmethod
    def _audit_annotation_labels(ann_path, logger):
        import json

        data = json.loads(ann_path.read_text(encoding="utf-8"))
        labels = [int(ann["category_id"]) for ann in data.get("annotations", []) if "category_id" in ann]
        if not labels:
            return
        min_label, max_label = min(labels), max(labels)
        if logger:
            logger(f"[AUDIT][RETINANET] min_label={min_label} max_label={max_label}")
