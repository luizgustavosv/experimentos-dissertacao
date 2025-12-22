from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

from app.controller import ExperimentController


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
        self.path_vars = {
            "dataset": tk.StringVar(),
            "weights": tk.StringVar(),
            "images": tk.StringVar(),
            "report": tk.StringVar(),
            "plots": tk.StringVar(),
            "normalized": tk.StringVar(),
        }

        self._build_header()
        self._build_forms()
        self._build_log_area()
        self._render_fields()

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
        action_combo = ttk.Combobox(
            action_frame,
            textvariable=self.action_var,
            values=["Treinar", "Inferir", "Validar", "Normalizar dataset"],
            state="readonly",
            width=25,
        )
        action_combo.pack(padx=10, pady=10)
        action_combo.bind("<<ComboboxSelected>>", lambda _: self._render_fields())

        self.dynamic_frame = tk.LabelFrame(container, text="Parâmetros", bg="#12233d", fg="white")
        self.dynamic_frame.pack(fill="x", pady=5)

        self.run_button = ttk.Button(container, text="Executar", command=self._execute_action)
        self.run_button.pack(pady=10)

    def _build_log_area(self) -> None:
        log_frame = tk.LabelFrame(self, text="Log de execução", bg="#12233d", fg="white")
        log_frame.pack(fill="both", expand=True, padx=20, pady=10)
        self.log_widget = tk.Text(log_frame, height=12, wrap="word", bg="#0f1b2d", fg="#c7d5ed")
        self.log_widget.pack(fill="both", expand=True)

    def _clear_dynamic(self) -> None:
        for widget in self.dynamic_frame.winfo_children():
            widget.destroy()

    def _render_fields(self) -> None:
        self._clear_dynamic()
        action = self.action_var.get()

        if action == "Treinar":
            self._add_path_selector("Dataset de treino", "dataset", is_dir=True)
            self._add_path_selector("Salvar pesos treinados", "weights", is_file=True, defaultextension=".pt")
        elif action == "Inferir":
            self._add_path_selector("Imagens para inferência", "images", is_dir=True)
            self._add_path_selector("Relatório PDF da inferência", "report", is_file=True, defaultextension=".pdf")
        elif action == "Validar":
            self._add_path_selector("Imagens de validação", "images", is_dir=True)
            self._add_path_selector("Pasta para gráficos", "plots", is_dir=True)
            self._add_path_selector("Relatório PDF da validação", "report", is_file=True, defaultextension=".pdf")
        elif action == "Normalizar dataset":
            self._add_path_selector("Dataset bruto", "dataset", is_dir=True)
            self._add_path_selector("Destino do dataset normalizado", "normalized", is_dir=True)

    def _add_path_selector(self, label: str, key: str, is_dir: bool = False, is_file: bool = False, defaultextension: str = "") -> None:
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
                path = filedialog.askopenfilename()
            if path:
                self.path_vars[key].set(path)

        ttk.Button(frame, text="Selecionar", command=browse).pack(side="left")

    def _append_log(self, message: str) -> None:
        self.log_widget.insert("end", message + "\n")
        self.log_widget.see("end")

    def _execute_action(self) -> None:
        action = self.action_var.get()
        algorithm_key = self.algorithm_var.get()
        try:
            if action == "Treinar":
                dataset = Path(self.path_vars["dataset"].get())
                weights = Path(self.path_vars["weights"].get())
                result = self.controller.execute_train(algorithm_key, dataset, weights, self._append_log)
            elif action == "Inferir":
                images = Path(self.path_vars["images"].get())
                report = Path(self.path_vars["report"].get())
                result = self.controller.execute_infer(algorithm_key, images, report, self._append_log)
            elif action == "Validar":
                images = Path(self.path_vars["images"].get())
                plots = Path(self.path_vars["plots"].get())
                report = Path(self.path_vars["report"].get())
                result = self.controller.execute_validate(algorithm_key, images, report, plots, self._append_log)
            elif action == "Normalizar dataset":
                dataset = Path(self.path_vars["dataset"].get())
                normalized = Path(self.path_vars["normalized"].get())
                result = self.controller.execute_normalize(algorithm_key, dataset, normalized, self._append_log)
            else:
                raise ValueError("Ação desconhecida")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Erro", str(exc))
            self._append_log(f"Erro: {exc}")
            return

        if result.metrics:
            m = result.metrics
            self._append_log(
                f"Métricas → Precisão: {m.precision:.3f}, Recall: {m.recall:.3f}, mAP@0.50: {m.map50:.3f}, mAP@0.50:0.95: {m.map50_95:.3f}"
            )
        self._append_log(result.message)
        messagebox.showinfo("Concluído", result.message)


def run_app() -> None:
    app = DetectorApp()
    app.mainloop()


if __name__ == "__main__":
    run_app()
