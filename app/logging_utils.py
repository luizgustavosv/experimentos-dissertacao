from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional


class PromptLogForwarder:
    """Encaminha mensagens para a interface gráfica e para um arquivo de log.

    A saída é formatada de modo parecido com um prompt interativo, com timestamp,
    permitindo que o usuário veja em tempo real tudo o que seria exibido no terminal.
    """

    def __init__(self, ui_logger: Callable[[str], None], log_file: Optional[Path] = None) -> None:
        self.ui_logger = ui_logger
        self.log_file = log_file.expanduser().resolve() if log_file else None

    def __call__(self, message: str) -> None:
        text = message.rstrip()
        if not text:
            return
        timestamped = f"[{datetime.now():%H:%M:%S}] {text}"
        for line in timestamped.splitlines():
            self.ui_logger(line)
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.log_file.open("a", encoding="utf-8") as fh:
                fh.write(timestamped + "\n")

    def write(self, message: str) -> None:
        # Implementa a interface de arquivo para permitir redirecionar stdout/stderr.
        if message:
            self.__call__(message)

    def flush(self) -> None:  # pragma: no cover - compatibilidade com protocolos de IO
        return


@contextmanager
def capture_prompt_output(forwarder: PromptLogForwarder):
    """Redireciona stdout/stderr para o logger, simulando a verbosidade do prompt."""

    original_out, original_err = sys.stdout, sys.stderr
    sys.stdout = forwarder  # type: ignore[assignment]
    sys.stderr = forwarder  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stdout = original_out
        sys.stderr = original_err
