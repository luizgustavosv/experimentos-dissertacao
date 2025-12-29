from __future__ import annotations

import tkinter as tk
import traceback
from pathlib import Path
from queue import Queue
from threading import Thread
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from app.controller import ExperimentController, OperationResult
from app.logging_utils import PromptLogForwarder, capture_prompt_output


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
        self.algorithm_var = tk.StringVar(value="YOLO")
        self.action_var = tk.StringVar(value="Treinar")
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
            "yolo_eval_out": tk.StringVar(),
        }
        self.epochs_var = tk.IntVar(value=10)
        self.pedestrian_only_var = tk.BooleanVar(value=False)
        self.eval_split_var = tk.StringVar(value="val")
        self.conf_threshold_var = tk.DoubleVar(value=0.05)
        self.iou_threshold_var = tk.DoubleVar(value=0.5)
        self.yolo_split_var = tk.StringVar(value="val")
        self.yolo_conf_var = tk.DoubleVar(value=0.001)
        self.yolo_iou_var = tk.DoubleVar(value=0.6)
        self.yolo_imgsz_var = tk.IntVar(value=640)
        self.yolo_batch_var = tk.IntVar(value=16)
        self.yolo_device_var = tk.StringVar(value="0")
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
            values=list(self.controller.detectors.keys()),
            state="readonly",
            width=25,
        )
        algo_combo.pack(padx=10, pady=10)
        algo_combo.bind("<<ComboboxSelected>>", lambda _: self._render_fields())

        action_frame = tk.LabelFrame(container, text="Ação", bg="#12233d", fg="white")
        action_frame.pack(fill="x", pady=5)
        action_frame.columnconfigure(0, weight=1)

        action_row = ttk.Frame(action_frame, padding=(8, 6))
        action_row.grid(row=0, column=0, sticky="ew")
        action_row.columnconfigure(0, weight=1)
        action_row.columnconfigure(1, weight=0)

        action_combo = ttk.Combobox(
            action_row,
            textvariable=self.action_var,
            values=["Treinar", "Inferir", "Avaliar SSD", "Avaliar YOLO", "Normalizar dataset"],
            state="readonly",
            width=25,
        )
        action_combo.grid(row=0, column=0, sticky="ew")
        action_combo.bind("<<ComboboxSelected>>", lambda _: self._render_fields())

        if hasattr(self, "btn_execute"):
            self.btn_execute.destroy()
        self.btn_execute = tk.Button(
            action_row,
            text="Executar",
            textvariable=self.run_button_text,
            command=self.on_execute_clicked,
            relief="raised",
            bd=2,
            padx=14,
            pady=6,
            font=("Segoe UI", 10, "bold"),
        )
        self.btn_execute.grid(row=0, column=1, sticky="e", padx=(12, 0))

        self.dynamic_frame = tk.LabelFrame(container, text="Parâmetros", bg="#12233d", fg="white")
        self.dynamic_frame.pack(fill="x", pady=5)

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
        elif action == "Inferir":
            self._add_path_selector("Pesos para inferência", "inference_weights")
            self._add_path_selector("Imagens para inferência", "images", is_dir=True)
            self._add_path_selector("Relatório PDF da inferência", "report", is_file=True, defaultextension=".pdf")
            self._add_pedestrian_filter_selector()
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
                self._add_text_selector("Dispositivo", self.yolo_device_var, placeholder="0 / cpu / cuda:0")
                self._register_yolo_eval_traces()
        elif action == "Normalizar dataset":
            self._add_dataset_type_selector()
            self._add_path_selector("Dataset bruto", "dataset", is_dir=True)
            self._add_path_selector("Destino do dataset normalizado", "normalized", is_dir=True)

        self._update_run_button_text()

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
        if placeholder and not variable.get():
            entry.insert(0, placeholder)

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

    def _add_pedestrian_filter_selector(self) -> None:
        frame = tk.Frame(self.dynamic_frame, bg="#12233d")
        frame.pack(fill="x", padx=10, pady=6)
        check = tk.Checkbutton(
            frame,
            text="Manter apenas detecções da classe pedestrian (classe 0) em inferência/validação",
            variable=self.pedestrian_only_var,
            onvalue=True,
            offvalue=False,
            bg="#12233d",
            fg="white",
            selectcolor="#0b172a",
            activebackground="#12233d",
            activeforeground="white",
            anchor="w",
            justify="left",
            wraplength=760,
        )
        check.pack(anchor="w")

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
        action = self.action_var.get()
        algorithm = self.algorithm_var.get()

        if action == "Avaliar SSD" and algorithm == "SSD":
            dataset_raw = self.path_vars["eval_dataset"].get().strip()
            weights_raw = self.path_vars["eval_weights"].get().strip()
            out_dir_raw = self.path_vars["eval_out"].get().strip()

            args: dict[str, object] = {
                "dataset": Path(dataset_raw) if dataset_raw else None,
                "pesos": Path(weights_raw) if weights_raw else None,
                "split": self.eval_split_var.get(),
                "saida": Path(out_dir_raw) if out_dir_raw else None,
                "conf": float(self.conf_threshold_var.get()),
                "iou": float(self.iou_threshold_var.get()),
            }
            self.run_button_text.set(self._build_prompt_command("avaliar", algorithm, args))
        elif action == "Avaliar YOLO" and algorithm == "YOLO":
            dataset_raw = self.path_vars["yolo_eval_dataset"].get().strip()
            weights_raw = self.path_vars["yolo_eval_weights"].get().strip()
            out_dir_raw = self.path_vars["yolo_eval_out"].get().strip()

            args = {
                "dataset": Path(dataset_raw) if dataset_raw else None,
                "pesos": Path(weights_raw) if weights_raw else None,
                "saida": Path(out_dir_raw) if out_dir_raw else None,
                "split": self.yolo_split_var.get(),
                "imgsz": int(self.yolo_imgsz_var.get()),
                "batch": int(self.yolo_batch_var.get()),
                "device": self.yolo_device_var.get(),
                "conf": float(self.yolo_conf_var.get()),
                "iou": float(self.yolo_iou_var.get()),
            }
            self.run_button_text.set(self._build_prompt_command("avaliar", algorithm, args))
        else:
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
                    images = Path(self.path_vars["images"].get())
                    prompt_dataset = annotations
                    exec_kwargs = {"dataset_path": annotations, "images_dir": images, "annotations_path": annotations}
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
                            **({"imagens": exec_kwargs.get("images_dir")} if exec_kwargs.get("images_dir") else {}),
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
                    )

            elif action == "Inferir":
                images = Path(self.path_vars["images"].get())
                weights_raw = self.path_vars["inference_weights"].get().strip()
                weights = Path(weights_raw) if weights_raw else None
                report = Path(self.path_vars["report"].get())
                pedestrian_only = bool(self.pedestrian_only_var.get())
                prompt_logger(
                    self._build_prompt_command(
                        "inferir",
                        algorithm_key,
                        {
                            "imagens": images,
                            "pesos": weights or "padrão",
                            "relatorio": report,
                            "apenas_pedestrian": pedestrian_only,
                        },
                    )
                )

                def run_action() -> OperationResult:
                    return self.controller.execute_infer(
                        algorithm_key, images, weights, report, pedestrian_only, prompt_logger
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
                device = self.yolo_device_var.get().strip() or "0"
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
