from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Dict, Optional

from app.detectors import load_detectors
from app.detectors.base import DetectionAlgorithm, Logger
from app.detectors.torchvision_eval import evaluate_torchvision_ssd_voc
from app.detectors.yolo_eval import evaluate_yolo
from app.metrics import InferencePerformance, Metrics


@dataclass
class OperationResult:
    metrics: Optional[Metrics] = None
    inference_performance: Optional[InferencePerformance] = None
    message: str = ""


class ExperimentController:
    def __init__(self) -> None:
        self.detectors: Dict[str, DetectionAlgorithm] = load_detectors()

    def execute_train(
        self,
        algorithm_key: str,
        dataset_path: Path,
        pretrained_weights: Optional[Path],
        output_dir: Path,
        epochs: int,
        logger: Optional[Logger] = None,
        images_dir: Optional[Path] = None,
        annotations_path: Optional[Path] = None,
        max_epochs: Optional[int] = None,
        early_stop_enabled: bool = False,
        early_stop_patience: int = 10,
        early_stop_min_delta: float = 0.0,
        early_stop_min_epochs: int = 10,
        early_stop_ema_alpha: float = 0.2,
    ) -> OperationResult:
        detector = self._get_detector(algorithm_key)
        metrics: Optional[Metrics]

        if hasattr(detector, "config"):
            detector.config.max_epochs = max_epochs
            detector.config.early_stop_enabled = early_stop_enabled
            detector.config.early_stop_patience = early_stop_patience
            detector.config.early_stop_min_delta = early_stop_min_delta
            detector.config.early_stop_min_epochs = early_stop_min_epochs
            detector.config.early_stop_ema_alpha = early_stop_ema_alpha

        if algorithm_key == "YOLO":
            dataset_yaml = dataset_path.expanduser().resolve()
            if dataset_yaml.suffix.lower() != ".yaml":
                raise ValueError("YOLO exige um arquivo .yaml de dataset.")
            if not dataset_yaml.is_file():
                raise FileNotFoundError(f"Arquivo YAML não encontrado: {dataset_yaml}")
            metrics = detector.train(dataset_yaml, pretrained_weights, output_dir, epochs, logger)

        elif algorithm_key == "SSD":
            voc_root = self._validate_voc_root(dataset_path)
            metrics = detector.train(voc_root, pretrained_weights, output_dir, epochs, logger)

        elif algorithm_key in {"RetinaNet", "Faster R-CNN"}:
            coco_ann = self._validate_coco_annotation_path(annotations_path or dataset_path)
            images_root = self._validate_coco_images_root(images_dir)
            val_ann = self._resolve_coco_val_annotation(coco_ann, images_root)
            dataset_root = images_root.parent if images_root.name.lower() == "images" else images_root
            metrics = detector.train(
                dataset_root,
                pretrained_weights,
                output_dir,
                epochs,
                logger,
                train_ann=coco_ann,
                val_ann=val_ann,
            )

        else:
            raise ValueError(f"Algoritmo desconhecido: {algorithm_key}")

        return OperationResult(metrics=metrics, message="Treinamento concluído.")

    def execute_infer(
        self,
        algorithm_key: str,
        images_dir: Path,
        weights_path: Optional[Path],
        report_out: Path,
        pedestrian_only: bool = False,
        logger: Optional[Logger] = None,
        ssd_score_threshold: Optional[float] = None,
        benchmark_mode: bool = False,
    ) -> OperationResult:
        detector = self._get_detector(algorithm_key)
        if algorithm_key == "SSD":
            performance = detector.infer(
                images_dir,
                weights_path,
                report_out,
                pedestrian_only=pedestrian_only,
                logger=logger,
                ssd_score_threshold=ssd_score_threshold,
                benchmark_mode=benchmark_mode,
            )
        else:
            performance = detector.infer(
                images_dir,
                weights_path,
                report_out,
                pedestrian_only=pedestrian_only,
                logger=logger,
                benchmark_mode=benchmark_mode,
            )
        return OperationResult(inference_performance=performance, message="Inferência concluída.")

    def execute_validate(
        self,
        algorithm_key: str,
        dataset_dir: Path,
        weights_path: Optional[Path],
        report_out: Path,
        plots_dir: Path,
        pedestrian_only: bool = False,
        logger: Optional[Logger] = None,
    ) -> OperationResult:
        detector = self._get_detector(algorithm_key)
        dataset_dir = dataset_dir.expanduser()
        dataset_yaml_path = (dataset_dir / "dataset.yaml").resolve()
        metrics = detector.validate(
            dataset_yaml_path, weights_path, report_out, plots_dir, pedestrian_only=pedestrian_only, logger=logger
        )
        return OperationResult(metrics=metrics, message="Validação concluída.")

    def execute_normalize(
        self,
        algorithm_key: str,
        dataset_type: str,
        dataset_dir: Path,
        normalized_dir: Path,
        logger: Optional[Logger] = None,
    ) -> OperationResult:
        detector = self._get_detector(algorithm_key)
        detector.normalize_dataset(dataset_type, dataset_dir, normalized_dir, logger)
        return OperationResult(metrics=None, message="Normalização concluída.")

    def execute_eval_ssd(
        self,
        algorithm_key: str,
        dataset_dir: Path,
        weights_path: Path,
        split: str = "val",
        out_dir: Optional[Path] = None,
        conf_threshold: float = 0.05,
        iou_threshold: float = 0.5,
        batch_size: int = 1,
        num_workers: int = 2,
        logger: Optional[Logger] = None,
        log_cb: Optional[Callable[[str], None]] = None,
    ) -> OperationResult:
        if algorithm_key != "SSD":
            raise ValueError("A avaliação dedicada está disponível apenas para SSD.")

        voc_root = self._validate_voc_root(dataset_dir)
        result = evaluate_torchvision_ssd_voc(
            voc_root=str(voc_root),
            weights_path=str(weights_path),
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            out_dir=str(out_dir) if out_dir else None,
            logger=logger,
            log_cb=log_cb,
        )

        metrics = Metrics(
            precision=float(result.get("precision", 0.0)),
            recall=float(result.get("recall", 0.0)),
            map50=float(result.get("map50", 0.0)),
            map50_95=float(result.get("map", 0.0)),
            device=result.get("device"),
            weights_path=weights_path,
            map_computed=True,
            extra={"mar_100": float(result.get("mar_100", 0.0))},
        )

        output_dir = Path(out_dir) if out_dir else Path(result["weights_path"]).parent / "eval"
        message = f"Avaliação concluída. Resultados salvos em {output_dir}"
        return OperationResult(metrics=metrics, message=message)

    def execute_eval_yolo(
        self,
        algorithm_key: str,
        data_yaml: Path,
        weights_path: Path,
        out_dir: Path,
        split: str = "val",
        imgsz: int = 640,
        batch: int = 16,
        device: str = "cpu",
        conf: float = 0.001,
        iou: float = 0.6,
        logger: Optional[Logger] = None,
        log_cb: Optional[Callable[[str], None]] = None,
    ) -> OperationResult:
        if algorithm_key != "YOLO":
            raise ValueError("A avaliação dedicada está disponível apenas para YOLO.")

        data_yaml = data_yaml.expanduser().resolve()
        weights_path = weights_path.expanduser().resolve()
        out_dir = out_dir.expanduser().resolve()

        result = evaluate_yolo(
            data_yaml=str(data_yaml),
            weights_path=str(weights_path),
            out_dir=str(out_dir),
            split=split,
            imgsz=imgsz,
            batch=batch,
            device=device,
            conf=conf,
            iou=iou,
            logger=logger,
            log_cb=log_cb,
        )

        message = f"Avaliação concluída. Resultados salvos em {out_dir}"
        return OperationResult(metrics=None, message=message)

    def execute_validate_faster_rcnn(
        self,
        algorithm_key: str,
        train_annotations: Path,
        images_dir: Path,
        weights_path: Path,
        val_annotations: Optional[Path] = None,
        val_mode: Optional[str] = None,
        conf_threshold: float = 0.05,
        iou_threshold: float = 0.5,
        logger: Optional[Logger] = None,
        log_cb: Optional[Callable[[str], None]] = None,
    ) -> OperationResult:
        if algorithm_key != "Faster R-CNN":
            raise ValueError("Selecione o algoritmo Faster R-CNN para executar a validação pós-treinamento.")

        coco_ann = self._validate_coco_annotation_path(train_annotations)
        images_root = self._validate_coco_images_root(images_dir)
        val_ann = self._validate_coco_annotation_path(val_annotations) if val_annotations else self._resolve_coco_val_annotation(coco_ann, images_root)
        dataset_root = images_root.parent if images_root.name.lower() == "images" else images_root

        detector = self._get_detector(algorithm_key)
        result = detector.validate_trained_weights(
            dataset_root,
            weights_path.expanduser().resolve(),
            train_ann=coco_ann,
            val_ann=val_ann,
            val_mode=val_mode,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            logger=logger,
            log_cb=log_cb,
        )

        output_dir = result.get("output_dir") or Path(weights_path).expanduser().resolve().parent
        message = f"Validação concluída. Resultados salvos em {output_dir}"
        return OperationResult(metrics=None, message=message)

    def execute_validate_retinanet(
        self,
        algorithm_key: str,
        train_annotations: Path,
        images_dir: Path,
        weights_path: Path,
        val_annotations: Optional[Path] = None,
        logger: Optional[Logger] = None,
        log_cb: Optional[Callable[[str], None]] = None,
    ) -> OperationResult:
        if algorithm_key != "RetinaNet":
            raise ValueError("Selecione o algoritmo RetinaNet para executar a validação pós-treinamento.")

        coco_ann = self._validate_coco_annotation_path(train_annotations)
        images_root = self._validate_coco_images_root(images_dir)
        val_ann = self._validate_coco_annotation_path(val_annotations) if val_annotations else self._resolve_coco_val_annotation(
            coco_ann, images_root
        )
        dataset_root = images_root.parent if images_root.name.lower() == "images" else images_root

        detector = self._get_detector(algorithm_key)
        result = detector.validate_trained_weights(
            dataset_root,
            weights_path.expanduser().resolve(),
            train_ann=coco_ann,
            val_ann=val_ann,
            logger=logger,
            log_cb=log_cb,
        )

        output_dir = result.get("output_dir") or Path(weights_path).expanduser().resolve().parent
        message = f"Validação concluída. Resultados salvos em {output_dir}"
        return OperationResult(metrics=None, message=message)

    def _get_detector(self, algorithm_key: str) -> DetectionAlgorithm:
        if algorithm_key not in self.detectors:
            raise KeyError(f"Algoritmo desconhecido: {algorithm_key}")
        return self.detectors[algorithm_key]

    def _validate_voc_root(self, dataset_path: Path) -> Path:
        dataset_path = dataset_path.expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Pasta do dataset não encontrada: {dataset_path}")

        candidates = [
            dataset_path,
            dataset_path / "VOC2007",
            dataset_path / "VOCdevkit" / "VOC2007",
        ]
        voc_root = next(
            (candidate for candidate in candidates if (candidate / "JPEGImages").exists() and (candidate / "Annotations").exists()),
            None,
        )
        if voc_root is None:
            raise FileNotFoundError(
                "Estrutura Pascal VOC inválida. São necessárias as pastas 'JPEGImages' e 'Annotations' no diretório informado."
            )
        return voc_root

    def _validate_coco_annotation_path(self, annotations_path: Path) -> Path:
        annotations_path = annotations_path.expanduser().resolve()
        if annotations_path.suffix.lower() != ".json":
            raise ValueError("Anotações COCO devem ser fornecidas em um arquivo .json.")
        if not annotations_path.is_file():
            raise FileNotFoundError(f"Arquivo de anotações COCO não encontrado: {annotations_path}")
        content = json.loads(annotations_path.read_text(encoding="utf-8"))
        required_keys = {"images", "annotations", "categories"}
        if not required_keys.issubset(content.keys()):
            raise ValueError(
                f"Arquivo COCO inválido. Esperado conter as chaves {', '.join(sorted(required_keys))}."
            )
        return annotations_path

    def _validate_coco_images_root(self, images_dir: Optional[Path]) -> Path:
        if images_dir is None:
            raise ValueError("Informe o diretório de imagens para o dataset COCO.")
        images_root = images_dir.expanduser().resolve()
        if not images_root.exists() or not images_root.is_dir():
            raise FileNotFoundError(f"Diretório de imagens inexistente: {images_root}")
        train_dir = images_root / "train"
        val_dir = images_root / "val"
        if not train_dir.exists() or not val_dir.exists():
            raise FileNotFoundError("Estrutura COCO incompleta: pastas train/ e val/ são obrigatórias dentro do diretório de imagens.")
        return images_root

    def _resolve_coco_val_annotation(self, train_ann: Path, images_root: Path) -> Path:
        candidates = []
        name_lower = train_ann.name.lower()
        if "train" in name_lower:
            candidates.append(train_ann.with_name(train_ann.name.replace("train", "val")))
        candidates.append(train_ann.parent / "instances_val.json")
        dataset_root = images_root.parent if images_root.name.lower() == "images" else images_root
        candidates.append(dataset_root / "annotations" / "instances_val.json")
        candidates.append(dataset_root / "val.json")

        for candidate in candidates:
            if candidate.exists():
                return self._validate_coco_annotation_path(candidate)
        raise FileNotFoundError(
            "Arquivo de validação COCO não encontrado. Esperado algo como 'instances_val.json' ao lado do JSON de treino."
        )
