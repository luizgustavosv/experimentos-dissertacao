from __future__ import annotations

import json
import tkinter as tk
import traceback
from pathlib import Path
from queue import Queue
from threading import Thread
from tkinter import filedialog, messagebox, ttk
from typing import Optional, Any

import torch

from app.controller import ExperimentController, OperationResult
from app.detectors.config import TrainConfig
from app.logging_utils import PromptLogForwarder, capture_prompt_output
from app.weights_metadata import format_weights_metadata


def get_weights_filetypes(algo: str) -> list[tuple[str, str]]:
    if algo == "YOLO":
        return [("YOLO weights (*.pt)", "*.pt"), ("All files", "*.*")]
    if algo in ("SSD", "RetinaNet", "Faster R-CNN"):
        return [("PyTorch weights (*.pth, *.pt)", "*.pth *.pt"), ("All files", "*.*")]
    return [("All files", "*.*")]


class DetectorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Laboratório de Detectores — Dissertação")
        self.geometry("820x640")
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self.configure(bg="#0b172a")

        self.controller = ExperimentController()
        self.cuda_available = torch.cuda.is_available()
        self.algorithm_actions: dict[str, list[str]] = {
            "YOLO": ["Treinar", "Inferir", "Inferência Rápida / Benchmark", "Avaliar YOLO", "Ler metadados dos pesos", "Normalizar dataset"],
            "SSD": ["Treinar", "Inferir", "Inferência Rápida / Benchmark", "Avaliar SSD", "Ler metadados dos pesos", "Normalizar dataset"],
            "Faster R-CNN": ["Treinar", "Inferir", "Inferência Rápida / Benchmark", "Validar", "Ler metadados dos pesos", "Normalizar dataset"],
            "RetinaNet": ["Treinar", "Inferir", "Inferência Rápida / Benchmark", "Validar", "Ler metadados dos pesos", "Normalizar dataset"],
        }
        self.algorithm_var = tk.StringVar(value="YOLO")
        self.action_var = tk.StringVar(value=self.algorithm_actions["YOLO"][0])
        self.dataset_type_var = tk.StringVar(value="HERIDAL")
        self.path_vars = {
            "dataset": tk.StringVar(),
            "annotations": tk.StringVar(),
            "weights": tk.StringVar(),
            "pretrained": tk.StringVar(),
            "inference_weights": tk.StringVar(),
            "validation_weights": tk.StringVar(),
            "images": tk.StringVar(),
            "report": tk.StringVar(),
            "plots": tk.StringVar(),
            "normalized": tk.StringVar(),
            "eval_dataset": tk.StringVar(),
            "eval_weights": tk.StringVar(),
            "eval_out": tk.StringVar(),
            "yolo_eval_dataset": tk.StringVar(),
            "yolo_eval_weights": tk.StringVar(),
            "metadata_weights": tk.StringVar(),
            "yolo_eval_out": tk.StringVar(),
            "val_annotations": tk.StringVar(),
            "validation_out": tk.StringVar(),
        }
        self.epochs_var = tk.IntVar(value=10)
        self.eval_split_var = tk.StringVar(value="val")
        self.conf_threshold_var = tk.DoubleVar(value=0.25)
        self.iou_threshold_var = tk.DoubleVar(value=0.5)
        self.ssd_infer_threshold_var = tk.DoubleVar(value=0.05)
        self.yolo_split_var = tk.StringVar(value="val")
        self.yolo_conf_var = tk.DoubleVar(value=0.25)
        self.yolo_iou_var = tk.DoubleVar(value=0.5)
        self.yolo_imgsz_var = tk.IntVar(value=640)
        self.yolo_batch_var = tk.IntVar(value=16)
        self.yolo_device_var = tk.StringVar(value="cpu")
        self.validation_device_var = tk.StringVar(value="cpu")
        self.val_mode_var = tk.StringVar(value="loss")
        self.max_epochs_enabled_var = tk.BooleanVar(value=False)
        self.max_epochs_var = tk.IntVar(value=10)
        self.early_stop_enabled_var = tk.BooleanVar(value=False)
        self.early_patience_var = tk.IntVar(value=10)
        self.early_min_delta_var = tk.DoubleVar(value=0.0)
        self.early_min_epochs_var = tk.IntVar(value=10)
        self.early_ema_alpha_var = tk.DoubleVar(value=0.2)
        default_config = TrainConfig()
        self.train_hparam_vars: dict[str, tk.Variable] = {
            "batch_size": tk.IntVar(value=default_config.batch_size),
            "lr": tk.DoubleVar(value=default_config.lr),
            "momentum": tk.DoubleVar(value=default_config.momentum),
            "device": tk.StringVar(value=default_config.device or "auto"),
            "num_workers": tk.IntVar(value=default_config.num_workers),
            "imgsz": tk.IntVar(value=default_config.imgsz),
            "seed": tk.IntVar(value=default_config.seed),
            "weight_decay": tk.DoubleVar(value=default_config.weight_decay),
            "lr_step_size": tk.IntVar(value=default_config.lr_step_size),
            "lr_gamma": tk.DoubleVar(value=default_config.lr_gamma),
            "verbose": tk.BooleanVar(value=default_config.verbose),
            "log_every": tk.IntVar(value=default_config.log_every),
            "debug_dataloader": tk.BooleanVar(value=default_config.debug_dataloader),
            "log_dir": tk.StringVar(value=str(default_config.log_dir)),
            "yolo_save_dir": tk.StringVar(value=""),
            "log_every_seconds": tk.IntVar(value=default_config.log_every_seconds),
            "pin_memory": tk.BooleanVar(value=default_config.pin_memory),
            "persistent_workers": tk.BooleanVar(value=default_config.persistent_workers),
            "prefetch_factor": tk.IntVar(value=default_config.prefetch_factor or 0),
            "drop_last": tk.BooleanVar(value=default_config.drop_last),
            "smoke_test_val_loss": tk.BooleanVar(value=default_config.smoke_test_val_loss),
            "smoke_test_samples": tk.IntVar(value=default_config.smoke_test_samples),
            "audit_datasets": tk.BooleanVar(value=default_config.audit_datasets),
            "dataset_num_classes": tk.IntVar(value=0),
            "num_classes": tk.IntVar(value=0),
            "val_mode": tk.StringVar(value=default_config.val_mode),
            "val_ratio": tk.DoubleVar(value=default_config.val_ratio),
            "early_stop_enabled": self.early_stop_enabled_var,
            "early_stop_patience": self.early_patience_var,
            "early_stop_min_delta": self.early_min_delta_var,
            "early_stop_min_epochs": self.early_min_epochs_var,
            "early_stop_ema_alpha": self.early_ema_alpha_var,
            "legacy_retinanet_compat": tk.BooleanVar(value=default_config.legacy_retinanet_compat),
            "save_final": tk.BooleanVar(value=default_config.save_final),
            "save_best": tk.BooleanVar(value=default_config.save_best),
            "save_every": tk.IntVar(value=default_config.save_every),
            "keep_last_k": tk.IntVar(value=default_config.keep_last_k),
            "monitor_metric": tk.StringVar(value=default_config.monitor_metric),
            "mode": tk.StringVar(value=default_config.mode),
        }
        self.run_button_text = tk.StringVar(value="Executar")
        self._variable_traces: list[tuple[tk.Variable, str]] = []

        self._build_header()
        self._build_forms()
        self._build_log_area()
        self.log_queue: Queue[str] = Queue()
        self._worker_thread: Optional[Thread] = None
        self._render_fields()
        self.after(100, self._poll_log_queue)

    def _build_header(self) -> None:
        header = tk.Frame(self, bg="#0b172a")
        header.pack(fill="x", padx=20, pady=10)

        title = tk.Label(
            header,
            text="Redes Neurais Profundas para Detecção de Pessoas em Imagens Aéreas",
            fg="white",
            bg="#0b172a",
            font=("Helvetica", 14, "bold"),
            wraplength=760,
            justify="center",
        )
        title.pack(pady=(0, 6))

        subtitle = tk.Label(
            header,
            text="Laboratório de experimentação com YOLO, SSD, Faster R-CNN e RetinaNet",
            fg="#c7d5ed",
            bg="#0b172a",
            font=("Helvetica", 11),
        )
        subtitle.pack()

    def _build_forms(self) -> None:
        container = tk.Frame(self, bg="#0b172a")
        container.pack(fill="x", padx=20, pady=10)

        algo_frame = tk.LabelFrame(container, text="Algoritmo", bg="#12233d", fg="white")
        algo_frame.pack(fill="x", pady=5)
        algo_combo = ttk.Combobox(
            algo_frame,
            textvariable=self.algorithm_var,
            values=list(self.algorithm_actions.keys()),
            state="readonly",
            width=25,
        )
        algo_combo.pack(padx=10, pady=10)
        algo_combo.bind("<<ComboboxSelected>>", self._on_algorithm_change)

        action_frame = tk.LabelFrame(container, text="Ação", bg="#12233d", fg="white")
        action_frame.pack(fill="x", pady=5)
        action_frame.columnconfigure(0, weight=1)

        action_row = ttk.Frame(action_frame, padding=(8, 6))
        action_row.grid(row=0, column=0, sticky="ew")
        action_row.columnconfigure(0, weight=1)
        action_row.columnconfigure(1, weight=0)

        self.action_combo = ttk.Combobox(
            action_row,
            textvariable=self.action_var,
            values=self._available_actions(self.algorithm_var.get()),
            state="readonly",
            width=25,
        )
        self.action_combo.grid(row=0, column=0, sticky="ew")
        self.action_combo.bind("<<ComboboxSelected>>", lambda _: self._render_fields())

        if hasattr(self, "btn_execute"):
            self.btn_execute.destroy()
        self.btn_execute = tk.Button(
            action_row,
            text="Executar",
            command=self.on_execute_clicked,
            relief="raised",
            bd=2,
            padx=14,
            pady=6,
            font=("Segoe UI", 10, "bold"),
        )
        self.btn_execute.grid(row=0, column=1, sticky="e", padx=(12, 0))

        self.dynamic_outer = tk.LabelFrame(container, text="Parâmetros", bg="#12233d", fg="white")
        self.dynamic_outer.pack(fill="both", pady=5)
        self.dynamic_canvas = tk.Canvas(self.dynamic_outer, height=320, bg="#12233d", highlightthickness=0)
        dynamic_scrollbar = ttk.Scrollbar(self.dynamic_outer, orient="vertical", command=self.dynamic_canvas.yview)
        self.dynamic_canvas.configure(yscrollcommand=dynamic_scrollbar.set)
        self.dynamic_canvas.pack(side="left", fill="both", expand=True)
        dynamic_scrollbar.pack(side="right", fill="y")
        self.dynamic_frame = tk.Frame(self.dynamic_canvas, bg="#12233d")
        self._dynamic_window_id = self.dynamic_canvas.create_window((0, 0), window=self.dynamic_frame, anchor="nw")
        self.dynamic_frame.bind(
            "<Configure>",
            lambda _event: self.dynamic_canvas.configure(scrollregion=self.dynamic_canvas.bbox("all")),
        )
        self.dynamic_canvas.bind(
            "<Configure>",
            lambda event: self.dynamic_canvas.itemconfigure(self._dynamic_window_id, width=event.width),
        )

        self._update_action_options()

    def _build_log_area(self) -> None:
        log_frame = tk.LabelFrame(self, text="Log de execução", bg="#12233d", fg="white")
        log_frame.pack(fill="both", expand=True, padx=20, pady=10)
        self.log_widget = tk.Text(log_frame, height=12, wrap="word", bg="#0f1b2d", fg="#c7d5ed")
        self.log_widget.pack(fill="both", expand=True)

    def _clear_dynamic(self) -> None:
        for widget in self.dynamic_frame.winfo_children():
            widget.destroy()
        self._clear_variable_traces()

    def _clear_variable_traces(self) -> None:
        for var, trace_id in self._variable_traces:
            var.trace_remove("write", trace_id)
        self._variable_traces.clear()

    def _render_fields(self) -> None:
        self._clear_dynamic()
        action = self.action_var.get()
        algorithm = self.algorithm_var.get()

        if action == "Treinar":
            if algorithm == "YOLO":
                self._add_path_selector("Dataset (arquivo .yaml)", "dataset", filetypes=[("YAML", "*.yaml")])
            elif algorithm == "SSD":
                self._add_path_selector("Pasta do dataset Pascal VOC", "dataset", is_dir=True)
            elif algorithm in {"RetinaNet", "Faster R-CNN"}:
                self._add_path_selector("Anotações COCO (.json)", "annotations", filetypes=[("COCO JSON", "*.json")])
                self._add_path_selector("Anotações COCO de validação (.json)", "val_annotations", filetypes=[("COCO JSON", "*.json")])
                self._add_path_selector(
                    "Pasta de imagens COCO (deve conter train/ e val/)", "images", is_dir=True
                )
            else:
                self._add_path_selector("Dataset", "dataset")
            self._add_path_selector(
                "Pesos pré-treinados", "pretrained", filetypes=get_weights_filetypes(algorithm)
            )
            self._add_path_selector("Pasta para salvar os pesos treinados", "weights", is_dir=True)
            self._add_epoch_selector()
            self._add_max_epoch_selector()
            self._add_early_stopping_controls()
            self._add_training_hyperparameter_controls()
        elif action in {"Inferir", "Inferência Rápida / Benchmark"}:
            self._add_path_selector("Pesos para inferência", "inference_weights")
            self._add_path_selector("Imagens para inferência", "images", is_dir=True)
            self._add_path_selector("Relatório PDF da inferência", "report", is_file=True, defaultextension=".pdf")
            if algorithm == "SSD":
                self._add_ssd_infer_threshold_selector()
        elif action == "Avaliar SSD":
            if algorithm != "SSD":
                tk.Label(
                    self.dynamic_frame,
                    text="A avaliação pós-treinamento está disponível apenas para o algoritmo SSD.",
                    bg="#12233d",
                    fg="white",
                    wraplength=760,
                ).pack(fill="x", padx=10, pady=6)
            else:
                self._add_path_selector("Pasta do dataset Pascal VOC", "eval_dataset", is_dir=True)
                self._add_path_selector("Pesos do SSD", "eval_weights", filetypes=get_weights_filetypes(algorithm))
                self._add_path_selector("Pasta para salvar métricas", "eval_out", is_dir=True)
                self._add_split_selector()
                self._add_conf_threshold_selector()
                self._add_iou_threshold_selector()
                self._register_eval_traces()
        elif action == "Avaliar YOLO":
            if algorithm != "YOLO":
                tk.Label(
                    self.dynamic_frame,
                    text="A avaliação pós-treinamento está disponível apenas para o algoritmo YOLO.",
                    bg="#12233d",
                    fg="white",
                    wraplength=760,
                ).pack(fill="x", padx=10, pady=6)
            else:
                self._add_path_selector("Dataset (arquivo data.yaml)", "yolo_eval_dataset", filetypes=[("YAML", "*.yaml")])
                self._add_path_selector("Pesos do YOLO (.pt)", "yolo_eval_weights", filetypes=get_weights_filetypes(algorithm))
                self._add_path_selector("Pasta para salvar métricas", "yolo_eval_out", is_dir=True)
                self._add_split_selector(variable=self.yolo_split_var, values=["val", "test"], label="Split para avaliação (YOLO)")
                self._add_conf_threshold_selector(variable=self.yolo_conf_var, label="Limite de confiança (YOLO)")
                self._add_iou_threshold_selector(variable=self.yolo_iou_var, label="Limite de IoU (YOLO)")
                self._add_numeric_selector("Tamanho da imagem (imgsz)", self.yolo_imgsz_var)
                self._add_numeric_selector("Tamanho do batch", self.yolo_batch_var)
                self._add_device_selector()
                self._register_yolo_eval_traces()
        elif action == "Validar":
            if algorithm not in {"Faster R-CNN", "RetinaNet"}:
                tk.Label(
                    self.dynamic_frame,
                    text="A validação pós-treinamento está disponível apenas para os algoritmos Faster R-CNN e RetinaNet.",
                    bg="#12233d",
                    fg="white",
                    wraplength=760,
                ).pack(fill="x", padx=10, pady=6)
            else:
                self._add_path_selector("Anotações COCO de treino (.json)", "annotations", filetypes=[("COCO JSON", "*.json")])
                self._add_path_selector(
                    "Anotações COCO de validação (.json)",
                    "val_annotations",
                    filetypes=[("COCO JSON", "*.json")],
                )
                self._add_path_selector("Pasta de imagens COCO (deve conter train/ e val/)", "images", is_dir=True)
                self._add_path_selector(
                    f"Pesos do {algorithm}",
                    "validation_weights",
                    filetypes=get_weights_filetypes(algorithm),
                )
                self._add_path_selector("Pasta para salvar métricas", "validation_out", is_dir=True)
                self._add_device_selector(variable=self.validation_device_var)
                self._add_val_mode_selector()
                self._add_conf_threshold_selector(label=f"Limite de confiança ({algorithm})")
                self._add_iou_threshold_selector(label=f"Limite de IoU ({algorithm})")
        elif action == "Ler metadados dos pesos":
            self._add_path_selector(
                f"Pesos do {algorithm}",
                "metadata_weights",
                filetypes=get_weights_filetypes(algorithm),
            )
        elif action == "Normalizar dataset":
            self._add_dataset_type_selector()
            self._add_path_selector("Dataset bruto", "dataset", is_dir=True)
            self._add_path_selector("Destino do dataset normalizado", "normalized", is_dir=True)

        self._update_run_button_text()

    def _available_actions(self, algorithm: str) -> list[str]:
        return self.algorithm_actions.get(
            algorithm, ["Treinar", "Inferir", "Inferência Rápida / Benchmark", "Normalizar dataset"]
        )

    def _on_algorithm_change(self, _event: object | None = None) -> None:
        self._update_action_options()
        self._render_fields()

    def _update_action_options(self) -> None:
        actions = self._available_actions(self.algorithm_var.get())
        self.action_combo.configure(values=actions)
        if self.action_var.get() not in actions:
            self.action_var.set(actions[0] if actions else "")

    def _add_path_selector(
        self,
        label: str,
        key: str,
        is_dir: bool = False,
        is_file: bool = False,
        defaultextension: str = "",
        filetypes: Optional[list[tuple[str, str]]] = None,
    ) -> None:
        frame = tk.Frame(self.dynamic_frame, bg="#12233d")
        frame.pack(fill="x", padx=10, pady=6)
        tk.Label(frame, text=label, bg="#12233d", fg="white").pack(anchor="w")
        entry = ttk.Entry(frame, textvariable=self.path_vars[key], width=80)
        entry.pack(side="left", padx=(0, 8))

        def browse() -> None:
            if is_dir:
                path = filedialog.askdirectory()
            elif is_file:
                path = filedialog.asksaveasfilename(defaultextension=defaultextension)
            else:
                if filetypes:
                    path = filedialog.askopenfilename(filetypes=filetypes)
                else:
                    path = filedialog.askopenfilename()
            if path:
                self.path_vars[key].set(path)

        ttk.Button(frame, text="Selecionar", command=browse).pack(side="left")

    def _add_dataset_type_selector(self) -> None:
        frame = tk.Frame(self.dynamic_frame, bg="#12233d")
        frame.pack(fill="x", padx=10, pady=6)
        tk.Label(frame, text="Tipo de dataset", bg="#12233d", fg="white").pack(anchor="w")
        combo = ttk.Combobox(frame, textvariable=self.dataset_type_var, values=["HERIDAL", "VisDrone"], state="readonly", width=30)
        combo.pack(anchor="w")

    def _add_split_selector(self, variable: Optional[tk.Variable] = None, values: Optional[list[str]] = None, label: str = "Split para avaliação (VOC)") -> None:
        frame = tk.Frame(self.dynamic_frame, bg="#12233d")
        frame.pack(fill="x", padx=10, pady=6)
        tk.Label(frame, text=label, bg="#12233d", fg="white").pack(anchor="w")
        combo = ttk.Combobox(
            frame,
            textvariable=variable or self.eval_split_var,
            values=values or ["train", "val", "test"],
            state="readonly",
            width=20,
        )
        combo.pack(anchor="w")

    def _add_conf_threshold_selector(self, variable: Optional[tk.Variable] = None, label: str = "Limite de confiança (score)") -> None:
        frame = tk.Frame(self.dynamic_frame, bg="#12233d")
        frame.pack(fill="x", padx=10, pady=6)
        tk.Label(frame, text=label, bg="#12233d", fg="white").pack(anchor="w")
        spinbox = tk.Spinbox(frame, from_=0.0, to=1.0, increment=0.01, textvariable=variable or self.conf_threshold_var, width=8)
        spinbox.pack(anchor="w")

    def _add_iou_threshold_selector(self, variable: Optional[tk.Variable] = None, label: str = "Limite de IoU para precision/recall") -> None:
        frame = tk.Frame(self.dynamic_frame, bg="#12233d")
        frame.pack(fill="x", padx=10, pady=6)
        tk.Label(frame, text=label, bg="#12233d", fg="white").pack(anchor="w")
        spinbox = tk.Spinbox(frame, from_=0.1, to=1.0, increment=0.05, textvariable=variable or self.iou_threshold_var, width=8)
        spinbox.pack(anchor="w")

    def _add_ssd_infer_threshold_selector(self) -> None:
        frame = tk.Frame(self.dynamic_frame, bg="#12233d")
        frame.pack(fill="x", padx=10, pady=6)
        tk.Label(frame, text="Threshold de Inferência SSD (score)", bg="#12233d", fg="white").pack(anchor="w")
        spinbox = tk.Spinbox(
            frame,
            from_=0.01,
            to=0.99,
            increment=0.01,
            format="%.2f",
            textvariable=self.ssd_infer_threshold_var,
            width=8,
        )
        spinbox.pack(anchor="w")

    def _add_val_mode_selector(self) -> None:
        frame = tk.Frame(self.dynamic_frame, bg="#12233d")
        frame.pack(fill="x", padx=10, pady=6)
        tk.Label(frame, text="Modo de validação", bg="#12233d", fg="white").pack(anchor="w")
        combo = ttk.Combobox(
            frame,
            textvariable=self.val_mode_var,
            values=["loss", "metrics"],
            state="readonly",
            width=12,
        )
        combo.pack(anchor="w")

    def _add_numeric_selector(self, label: str, variable: tk.Variable) -> None:
        frame = tk.Frame(self.dynamic_frame, bg="#12233d")
        frame.pack(fill="x", padx=10, pady=6)
        tk.Label(frame, text=label, bg="#12233d", fg="white").pack(anchor="w")
        entry = ttk.Entry(frame, textvariable=variable, width=12)
        entry.pack(anchor="w")

    def _add_text_selector(self, label: str, variable: tk.Variable, placeholder: str = "") -> None:
        frame = tk.Frame(self.dynamic_frame, bg="#12233d")
        frame.pack(fill="x", padx=10, pady=6)
        tk.Label(frame, text=label, bg="#12233d", fg="white").pack(anchor="w")
        entry = ttk.Entry(frame, textvariable=variable, width=20)
        entry.pack(anchor="w")

    def _build_device_options(self) -> list[str]:
        if not self.cuda_available:
            return ["cpu"]
        gpu_indices = [str(idx) for idx in range(torch.cuda.device_count())]
        return ["cpu", *gpu_indices]

    def _add_device_selector(self, variable: Optional[tk.Variable] = None) -> None:
        device_var = variable or self.yolo_device_var
        options = self._build_device_options()
        if device_var.get() not in options:
            device_var.set("cpu")

        frame = tk.Frame(self.dynamic_frame, bg="#12233d")
        frame.pack(fill="x", padx=10, pady=6)
        tk.Label(frame, text="Dispositivo", bg="#12233d", fg="white").pack(anchor="w")
        combo = ttk.Combobox(
            frame,
            textvariable=device_var,
            values=options,
            state="readonly",
            width=20,
        )
        combo.pack(anchor="w")

    def _register_eval_traces(self) -> None:
        variables: list[tk.Variable] = [
            self.path_vars["eval_dataset"],
            self.path_vars["eval_weights"],
            self.path_vars["eval_out"],
            self.eval_split_var,
            self.conf_threshold_var,
            self.iou_threshold_var,
        ]

        for var in variables:
            trace_id = var.trace_add("write", self._update_run_button_text)
            self._variable_traces.append((var, trace_id))

    def _register_yolo_eval_traces(self) -> None:
        variables: list[tk.Variable] = [
            self.path_vars["yolo_eval_dataset"],
            self.path_vars["yolo_eval_weights"],
            self.path_vars["yolo_eval_out"],
            self.yolo_split_var,
            self.yolo_conf_var,
            self.yolo_iou_var,
            self.yolo_imgsz_var,
            self.yolo_batch_var,
            self.yolo_device_var,
        ]

        for var in variables:
            trace_id = var.trace_add("write", self._update_run_button_text)
            self._variable_traces.append((var, trace_id))

    def _add_epoch_selector(self) -> None:
        frame = tk.Frame(self.dynamic_frame, bg="#12233d")
        frame.pack(fill="x", padx=10, pady=6)
        tk.Label(frame, text="Número de épocas", bg="#12233d", fg="white").pack(anchor="w")
        spinbox = tk.Spinbox(frame, from_=1, to=500, textvariable=self.epochs_var, width=8)
        spinbox.pack(anchor="w")

    def _add_max_epoch_selector(self) -> None:
        frame = tk.Frame(self.dynamic_frame, bg="#12233d")
        frame.pack(fill="x", padx=10, pady=6)
        row = tk.Frame(frame, bg="#12233d")
        row.pack(anchor="w")
        check = tk.Checkbutton(
            row,
            text="Ativar limite superior de épocas (max_epochs)",
            variable=self.max_epochs_enabled_var,
            onvalue=True,
            offvalue=False,
            bg="#12233d",
            fg="white",
            selectcolor="#0b172a",
            activebackground="#12233d",
            activeforeground="white",
            command=lambda: spin.config(state="normal" if self.max_epochs_enabled_var.get() else "disabled"),
        )
        check.pack(side="left")
        spin = tk.Spinbox(
            row,
            from_=1,
            to=1000,
            textvariable=self.max_epochs_var,
            width=8,
            state="disabled",
        )
        spin.pack(side="left", padx=(10, 0))

    def _add_early_stopping_controls(self) -> None:
        frame = tk.LabelFrame(self.dynamic_frame, text="Early stopping (loss + EMA)", bg="#12233d", fg="white")
        frame.pack(fill="x", padx=10, pady=6)

        def _toggle(state: bool) -> None:
            new_state = "normal" if state else "disabled"
            for widget in controls:
                widget.configure(state=new_state)

        header = tk.Checkbutton(
            frame,
            text="Habilitar early stopping",
            variable=self.early_stop_enabled_var,
            onvalue=True,
            offvalue=False,
            bg="#12233d",
            fg="white",
            selectcolor="#0b172a",
            activebackground="#12233d",
            activeforeground="white",
            command=lambda: _toggle(self.early_stop_enabled_var.get()),
        )
        header.pack(anchor="w", padx=6, pady=(4, 6))

        controls: list[tk.Widget] = []

        def _spinbox(label: str, variable: tk.Variable, from_: float, to: float, increment: float = 1.0) -> None:
            row = tk.Frame(frame, bg="#12233d")
            row.pack(fill="x", padx=10, pady=2)
            tk.Label(row, text=label, bg="#12233d", fg="white").pack(side="left")
            spin = tk.Spinbox(row, from_=from_, to=to, increment=increment, textvariable=variable, width=8)
            spin.pack(side="left", padx=(6, 0))
            controls.append(spin)

        _spinbox("Paciência (épocas)", self.early_patience_var, 1, 500, 1)
        _spinbox("min_delta", self.early_min_delta_var, 0.0, 10.0, 0.1)
        _spinbox("min_epochs", self.early_min_epochs_var, 0, 500, 1)
        _spinbox("ema_alpha", self.early_ema_alpha_var, 0.01, 1.0, 0.01)

        _toggle(self.early_stop_enabled_var.get())

    def _add_training_hyperparameter_controls(self) -> None:
        frame = tk.LabelFrame(self.dynamic_frame, text="Hiperparâmetros de treinamento", bg="#12233d", fg="white")
        frame.pack(fill="x", padx=10, pady=6)

        numeric_fields = [
            ("batch_size", "Batch size"),
            ("lr", "Learning rate"),
            ("momentum", "Momentum"),
            ("num_workers", "Num workers"),
            ("imgsz", "Tamanho da imagem"),
            ("seed", "Seed"),
            ("weight_decay", "Weight decay"),
            ("lr_step_size", "LR step size"),
            ("lr_gamma", "LR gamma"),
            ("log_every", "Log a cada N batches"),
            ("log_every_seconds", "Heartbeat em segundos"),
            ("yolo_save_dir", "Diretório da run YOLO (vazio = padrão)"),
            ("prefetch_factor", "Prefetch factor (0 = automático/desativado)"),
            ("smoke_test_samples", "Amostras do smoke test"),
            ("dataset_num_classes", "Classes do dataset (0 = inferir)"),
            ("num_classes", "Classes do modelo (0 = inferir)"),
            ("val_ratio", "Val ratio"),
            ("save_every", "Salvar checkpoint a cada N épocas"),
            ("keep_last_k", "Manter últimos K checkpoints"),
        ]
        for key, label in numeric_fields:
            self._add_hparam_entry(frame, label, key)

        self._add_hparam_combo(frame, "Dispositivo de treino", "device", ["auto", "cpu", *self._cuda_device_options()])
        self._add_hparam_combo(frame, "Modo de validação no treino", "val_mode", ["loss", "metrics", "both"])
        self._add_hparam_combo(frame, "Métrica monitorada", "monitor_metric", ["val_map", "val_loss", "train_loss"])
        self._add_hparam_combo(frame, "Modo do monitor", "mode", ["max", "min"])
        self._add_hparam_entry(frame, "Diretório de logs", "log_dir")

        for key, label in [
            ("verbose", "Verbose"),
            ("debug_dataloader", "Debug dataloader"),
            ("pin_memory", "Pin memory"),
            ("persistent_workers", "Persistent workers"),
            ("drop_last", "Drop last batch"),
            ("smoke_test_val_loss", "Smoke test de val loss"),
            ("audit_datasets", "Auditar datasets"),
            ("legacy_retinanet_compat", "Compatibilidade legada RetinaNet"),
            ("save_final", "Salvar checkpoint final"),
            ("save_best", "Salvar melhor checkpoint"),
        ]:
            self._add_hparam_check(frame, label, key)

    def _add_hparam_entry(self, parent: tk.Widget, label: str, key: str) -> None:
        row = tk.Frame(parent, bg="#12233d")
        row.pack(fill="x", padx=10, pady=2)
        tk.Label(row, text=label, bg="#12233d", fg="white", width=34, anchor="w").pack(side="left")
        ttk.Entry(row, textvariable=self.train_hparam_vars[key], width=18).pack(side="left")

    def _add_hparam_combo(self, parent: tk.Widget, label: str, key: str, values: list[str]) -> None:
        row = tk.Frame(parent, bg="#12233d")
        row.pack(fill="x", padx=10, pady=2)
        tk.Label(row, text=label, bg="#12233d", fg="white", width=34, anchor="w").pack(side="left")
        ttk.Combobox(row, textvariable=self.train_hparam_vars[key], values=values, state="readonly", width=18).pack(side="left")

    def _add_hparam_check(self, parent: tk.Widget, label: str, key: str) -> None:
        check = tk.Checkbutton(
            parent,
            text=label,
            variable=self.train_hparam_vars[key],
            onvalue=True,
            offvalue=False,
            bg="#12233d",
            fg="white",
            selectcolor="#0b172a",
            activebackground="#12233d",
            activeforeground="white",
        )
        check.pack(anchor="w", padx=10, pady=1)

    def _training_config_overrides(self) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        for key, variable in self.train_hparam_vars.items():
            value = variable.get()
            if key in {"dataset_num_classes", "num_classes", "prefetch_factor"} and int(value) <= 0:
                overrides[key] = None
            elif key == "device" and str(value).strip().lower() == "auto":
                overrides[key] = None
            elif key in {"log_dir", "yolo_save_dir"}:
                if key == "yolo_save_dir" and not str(value).strip():
                    overrides[key] = None
                    continue
                overrides[key] = Path(str(value).strip() or "logs")
            else:
                overrides[key] = value
        return overrides

    def _cuda_device_options(self) -> list[str]:
        if not self.cuda_available:
            return []
        return [str(idx) for idx in range(torch.cuda.device_count())]

    def append_log(self, message: str) -> None:
        self.log_widget.insert("end", message + "\n")
        self.log_widget.see("end")

    def _queue_log(self, message: str) -> None:
        self.log_queue.put(message)

    def _poll_log_queue(self) -> None:
        while not self.log_queue.empty():
            self.append_log(self.log_queue.get())
        self.after(100, self._poll_log_queue)

    def _build_prompt_command(self, action: str, algorithm: str, args: dict[str, object]) -> str:
        parts = [action.lower(), algorithm]
        for key, value in args.items():
            if value is None or value == "":
                continue
            if isinstance(value, Path):
                path_str = str(value.expanduser())
                if path_str in {"", "."}:
                    continue
                value_to_append = path_str
            elif isinstance(value, bool):
                value_to_append = "sim" if value else "não"
            else:
                value_to_append = value
            parts.append(f"{key}={value_to_append}")
        return "$ " + " ".join(str(part) for part in parts)

    def _update_run_button_text(self, *_args: object) -> None:
        self.run_button_text.set("Executar")

    def _emit_gui(self, message: str, *, stdout: bool = True) -> None:
        if stdout:
            print(message, flush=True)
        self.after(0, self.append_log, message)

    def on_execute_clicked(self) -> None:
        print("[UI] Clique em Executar recebido.", flush=True)
        self._emit_gui("[UI] Clique em Executar recebido.", stdout=False)
        if self.action_var.get() == "Avaliar SSD":
            self._execute_eval_ssd()
        else:
            self._execute_action()

    def _execute_action(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showinfo("Em execução", "Aguarde o término do comando atual antes de iniciar outro.")
            return

        action = self.action_var.get()
        algorithm_key = self.algorithm_var.get()
        prompt_logger = PromptLogForwarder(self._queue_log, Path("app.log"))

        try:
            if action == "Treinar":
                weights = Path(self.path_vars["weights"].get())
                pretrained_raw = self.path_vars["pretrained"].get().strip()
                pretrained = Path(pretrained_raw) if pretrained_raw else None
                epochs = int(self.epochs_var.get())
                max_epochs = int(self.max_epochs_var.get()) if self.max_epochs_enabled_var.get() else None
                early_stop_enabled = bool(self.early_stop_enabled_var.get())
                patience = int(self.early_patience_var.get())
                min_delta = float(self.early_min_delta_var.get())
                min_epochs = int(self.early_min_epochs_var.get())
                ema_alpha = float(self.early_ema_alpha_var.get())
                config_overrides = self._training_config_overrides()

                if pretrained and not pretrained.exists():
                    messagebox.showerror("Erro", "O arquivo de pesos pré-treinados não existe.")
                    return

                if algorithm_key == "YOLO":
                    dataset_arg = Path(self.path_vars["dataset"].get())
                    prompt_dataset = dataset_arg
                    exec_kwargs = {"dataset_path": dataset_arg}
                elif algorithm_key == "SSD":
                    dataset_arg = Path(self.path_vars["dataset"].get())
                    prompt_dataset = dataset_arg
                    exec_kwargs = {"dataset_path": dataset_arg}
                elif algorithm_key in {"RetinaNet", "Faster R-CNN"}:
                    annotations = Path(self.path_vars["annotations"].get())
                    val_annotations_raw = self.path_vars["val_annotations"].get().strip()
                    if not val_annotations_raw:
                        raise ValueError("Informe as anotações COCO de validação (.json) para o treinamento.")
                    val_annotations = Path(val_annotations_raw)
                    images = Path(self.path_vars["images"].get())
                    prompt_dataset = annotations
                    exec_kwargs = {
                        "dataset_path": annotations,
                        "images_dir": images,
                        "annotations_path": annotations,
                        "val_annotations_path": val_annotations,
                    }
                else:
                    raise ValueError(f"Algoritmo desconhecido para treinamento: {algorithm_key}")

                prompt_logger(
                    self._build_prompt_command(
                        "treinar",
                        algorithm_key,
                        {
                            "dataset": prompt_dataset,
                            "pesos_out": weights,
                            "pretreinados": pretrained or "padrão",
                            "epocas": epochs,
                            **({"max_epochs": max_epochs} if max_epochs else {}),
                            **({"early_stop": "on"} if early_stop_enabled else {}),
                            "batch_size": config_overrides["batch_size"],
                            "lr": config_overrides["lr"],
                            "momentum": config_overrides["momentum"],
                            "device": config_overrides["device"] or "auto",
                            "imgsz": config_overrides["imgsz"],
                            "seed": config_overrides["seed"],
                            **({"imagens": exec_kwargs.get("images_dir")} if exec_kwargs.get("images_dir") else {}),
                            **({"val_annotations": exec_kwargs.get("val_annotations_path")} if exec_kwargs.get("val_annotations_path") else {}),
                        },
                    )
                )

                def run_action() -> OperationResult:
                    return self.controller.execute_train(
                        algorithm_key,
                        pretrained_weights=pretrained,
                        output_dir=weights,
                        epochs=epochs,
                        logger=prompt_logger,
                        **exec_kwargs,
                        max_epochs=max_epochs,
                        early_stop_enabled=early_stop_enabled,
                        early_stop_patience=patience,
                        early_stop_min_delta=min_delta,
                        early_stop_min_epochs=min_epochs,
                        early_stop_ema_alpha=ema_alpha,
                        config_overrides=config_overrides,
                    )

            elif action in {"Inferir", "Inferência Rápida / Benchmark"}:
                images = Path(self.path_vars["images"].get())
                weights_raw = self.path_vars["inference_weights"].get().strip()
                weights = Path(weights_raw) if weights_raw else None
                report = Path(self.path_vars["report"].get())
                benchmark_mode = action == "Inferência Rápida / Benchmark"
                ssd_score_threshold = (
                    float(self.ssd_infer_threshold_var.get()) if algorithm_key == "SSD" else None
                )
                prompt_logger(
                    self._build_prompt_command(
                        "benchmark" if benchmark_mode else "inferir",
                        algorithm_key,
                        {
                            "imagens": images,
                            "pesos": weights or "padrão",
                            "relatorio": report,
                            "modo": "benchmark" if benchmark_mode else "normal",
                            **({"ssd_score_threshold": ssd_score_threshold} if algorithm_key == "SSD" else {}),
                        },
                    )
                )

                def run_action() -> OperationResult:
                    return self.controller.execute_infer(
                        algorithm_key,
                        images,
                        weights,
                        report,
                        logger=prompt_logger,
                        ssd_score_threshold=ssd_score_threshold,
                        benchmark_mode=benchmark_mode,
                    )

            elif action == "Avaliar SSD":
                if algorithm_key != "SSD":
                    raise ValueError("Selecione o algoritmo SSD para executar a avaliação dedicada.")
                dataset_dir = Path(self.path_vars["eval_dataset"].get())
                weights = Path(self.path_vars["eval_weights"].get())
                out_dir_raw = self.path_vars["eval_out"].get().strip()
                out_dir = Path(out_dir_raw) if out_dir_raw else None
                split = self.eval_split_var.get().strip() or "val"
                conf_threshold = float(self.conf_threshold_var.get())
                iou_threshold = float(self.iou_threshold_var.get())
                prompt_logger(
                    self._build_prompt_command(
                        "avaliar", algorithm_key, {"dataset": dataset_dir, "pesos": weights, "split": split, "saida": out_dir or "padrão"}
                    )
                )

                def run_action() -> OperationResult:
                    return self.controller.execute_eval_ssd(
                        algorithm_key,
                        dataset_dir=dataset_dir,
                        weights_path=weights,
                        split=split,
                        out_dir=out_dir,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        logger=prompt_logger,
                    )

            elif action == "Avaliar YOLO":
                if algorithm_key != "YOLO":
                    raise ValueError("Selecione o algoritmo YOLO para executar a avaliação dedicada.")

                data_yaml = Path(self.path_vars["yolo_eval_dataset"].get())
                weights = Path(self.path_vars["yolo_eval_weights"].get())
                out_dir = Path(self.path_vars["yolo_eval_out"].get())
                split = self.yolo_split_var.get().strip() or "val"
                imgsz = int(self.yolo_imgsz_var.get())
                batch = int(self.yolo_batch_var.get())
                device = self.yolo_device_var.get().strip() or "cpu"
                conf = float(self.yolo_conf_var.get())
                iou = float(self.yolo_iou_var.get())

                prompt_logger(
                    self._build_prompt_command(
                        "avaliar",
                        algorithm_key,
                        {
                            "dataset": data_yaml,
                            "pesos": weights,
                            "saida": out_dir,
                            "split": split,
                            "imgsz": imgsz,
                            "batch": batch,
                            "device": device,
                            "conf": conf,
                            "iou": iou,
                        },
                    )
                )

                def run_action() -> OperationResult:
                    return self.controller.execute_eval_yolo(
                        algorithm_key,
                        data_yaml=data_yaml,
                        weights_path=weights,
                        out_dir=out_dir,
                        split=split,
                        imgsz=imgsz,
                        batch=batch,
                        device=device,
                        conf=conf,
                        iou=iou,
                        logger=prompt_logger,
                        log_cb=lambda msg: self._emit_gui(msg, stdout=False),
                    )

            elif action == "Validar":
                if algorithm_key not in {"Faster R-CNN", "RetinaNet"}:
                    raise ValueError("Selecione o algoritmo suportado para executar a validação pós-treinamento.")

                train_annotations = Path(self.path_vars["annotations"].get())
                images_dir = Path(self.path_vars["images"].get())
                val_annotations_raw = self.path_vars["val_annotations"].get().strip()
                if not val_annotations_raw:
                    raise ValueError("Informe as anotações COCO de validação (.json).")
                val_annotations = Path(val_annotations_raw)
                weights = Path(self.path_vars["validation_weights"].get())
                output_dir_raw = self.path_vars["validation_out"].get().strip()
                output_dir = Path(output_dir_raw) if output_dir_raw else None
                device = self.validation_device_var.get().strip() or "cpu"
                conf_threshold = float(self.conf_threshold_var.get())
                iou_threshold = float(self.iou_threshold_var.get())

                prompt_logger(
                    self._build_prompt_command(
                        "validar",
                        algorithm_key,
                        {
                            "treino": train_annotations,
                            "val": val_annotations,
                            "imagens": images_dir,
                            "pesos": weights,
                            "saida": output_dir or "logs",
                            "device": device,
                            "modo": self.val_mode_var.get(),
                            "conf_threshold": conf_threshold,
                            "iou_threshold": iou_threshold,
                        },
                    )
                )

                def run_action() -> OperationResult:
                    if algorithm_key == "Faster R-CNN":
                        return self.controller.execute_validate_faster_rcnn(
                            algorithm_key,
                            train_annotations=train_annotations,
                            images_dir=images_dir,
                            weights_path=weights,
                            val_annotations=val_annotations,
                            val_mode=self.val_mode_var.get(),
                            output_dir=output_dir,
                            device=device,
                            conf_threshold=conf_threshold,
                            iou_threshold=iou_threshold,
                            logger=prompt_logger,
                            log_cb=lambda msg: self._emit_gui(msg, stdout=False),
                        )
                    return self.controller.execute_validate_retinanet(
                        algorithm_key,
                        train_annotations=train_annotations,
                        images_dir=images_dir,
                        weights_path=weights,
                        val_annotations=val_annotations,
                        val_mode=self.val_mode_var.get(),
                        output_dir=output_dir,
                        device=device,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        logger=prompt_logger,
                        log_cb=lambda msg: self._emit_gui(msg, stdout=False),
                        )

            elif action == "Ler metadados dos pesos":
                weights = Path(self.path_vars["metadata_weights"].get())
                if not weights.is_file():
                    raise FileNotFoundError(f"Arquivo de pesos não encontrado: {weights}")

                prompt_logger(
                    self._build_prompt_command(
                        "ler_metadados_pesos",
                        algorithm_key,
                        {"pesos": weights},
                    )
                )

                def run_action() -> OperationResult:
                    return self.controller.execute_read_weights_metadata(
                        algorithm_key,
                        weights,
                        logger=prompt_logger,
                    )

            elif action == "Normalizar dataset":
                dataset = Path(self.path_vars["dataset"].get())
                normalized = Path(self.path_vars["normalized"].get())
                dataset_type = self.dataset_type_var.get().lower()
                prompt_logger(
                    self._build_prompt_command(
                        "normalizar",
                        algorithm_key,
                        {"dataset": dataset, "tipo": dataset_type, "saida": normalized},
                    )
                )

                def run_action() -> OperationResult:
                    return self.controller.execute_normalize(
                        algorithm_key, dataset_type, dataset, normalized, prompt_logger
                    )

            else:
                raise ValueError("Ação desconhecida")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Erro", str(exc))
            self.append_log(f"Erro: {exc}")
            return

        self._set_running(True)

        def worker() -> None:
            try:
                with capture_prompt_output(prompt_logger):
                    result = run_action()
                prompt_logger(f"$ comando finalizado → {result.message}")
                self.after(0, self._on_action_complete, result)
            except Exception as exc:  # noqa: BLE001
                self.after(0, self._on_action_error, exc)

        self._worker_thread = Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _execute_eval_ssd(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showinfo("Em execução", "Aguarde o término do comando atual antes de iniciar outro.")
            return

        algorithm_key = self.algorithm_var.get()
        if algorithm_key != "SSD":
            messagebox.showerror("Erro", "Selecione o algoritmo SSD para executar a avaliação dedicada.")
            return

        dataset_dir = Path(self.path_vars["eval_dataset"].get())
        weights = Path(self.path_vars["eval_weights"].get())
        out_dir_raw = self.path_vars["eval_out"].get().strip()
        out_dir = Path(out_dir_raw) if out_dir_raw else None
        split = self.eval_split_var.get().strip() or "val"
        conf_threshold = float(self.conf_threshold_var.get())
        iou_threshold = float(self.iou_threshold_var.get())

        self._set_running(True)
        self._emit_gui("[EVAL] Preparando dataloader...")

        def worker() -> None:
            try:
                result = self.controller.execute_eval_ssd(
                    algorithm_key,
                    dataset_dir=dataset_dir,
                    weights_path=weights,
                    split=split,
                    out_dir=out_dir,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    logger=None,
                    log_cb=self._emit_gui,
                )
                self.after(0, self._on_action_complete, result)
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                self._emit_gui(tb)
                self.after(0, self._on_action_error, exc)

        self._worker_thread = Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _on_action_complete(self, result: OperationResult) -> None:
        if result.metadata:
            formatted_metadata = format_weights_metadata(result.metadata)
            self.append_log(formatted_metadata)
            self._show_metadata_window(formatted_metadata)
        if result.results_path and result.results_path.exists():
            try:
                payload = json.loads(result.results_path.read_text(encoding="utf-8"))
                params = payload.get("parameters", {})
                metrics = payload.get("metrics", {})
                diagnostic_metrics = metrics.get("diagnostic", {}) if isinstance(metrics.get("diagnostic"), dict) else metrics
                self.append_log(
                    "[RESULTADO][UNIFICADO] "
                    f"arquivo={result.results_path} imagens={payload.get('num_images')} "
                    f"detecções={payload.get('num_detections')} "
                    f"conf={params.get('conf_threshold')} iou={params.get('iou_association_threshold')} "
                    f"max_det={params.get('max_detections_per_image')} device={params.get('device')} "
                    f"map50_diag={diagnostic_metrics.get('map50')} map50_95_diag={diagnostic_metrics.get('map50_95')}"
                )
            except Exception as exc:
                self.append_log(f"[RESULTADO][UNIFICADO] Não foi possível ler {result.results_path}: {exc}")
        if result.inference_performance:
            perf = result.inference_performance
            self.append_log(
                f"Latência → {perf.images_per_second:.2f} img/s ({perf.milliseconds_per_image:.2f} ms/imagem)"
            )
        elif result.metrics:
            m = result.metrics
            self.append_log(
                f"Métricas → Precisão: {m.precision:.3f}, Recall: {m.recall:.3f}, mAP@0.50: {m.map50:.3f}, mAP@0.50:0.95: {m.map50_95:.3f}"
            )
        self.append_log(result.message)
        self._set_running(False)
        messagebox.showinfo("Concluído", result.message)

    def _show_metadata_window(self, content: str) -> None:
        window = tk.Toplevel(self)
        window.title("Metadados dos pesos")
        window.geometry("900x620")
        window.configure(bg="#0b172a")

        frame = tk.Frame(window, bg="#0b172a")
        frame.pack(fill="both", expand=True, padx=12, pady=12)

        text = tk.Text(frame, wrap="none", bg="#0f1b2d", fg="#c7d5ed")
        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        text.insert("1.0", content)
        text.configure(state="disabled")

    def _on_action_error(self, exc: Exception) -> None:
        self._set_running(False)
        self.append_log(f"Erro: {exc}")
        messagebox.showerror("Erro", str(exc))

    def _set_running(self, running: bool) -> None:
        if running:
            self.btn_execute.config(state=tk.DISABLED)
        else:
            self.btn_execute.config(state=tk.NORMAL)
        self.update_idletasks()


def run_app() -> None:
    app = DetectorApp()
    app.mainloop()


if __name__ == "__main__":
    run_app()
