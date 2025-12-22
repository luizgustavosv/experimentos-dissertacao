from __future__ import annotations

import math
from time import perf_counter
from typing import Optional

from app.detectors.base import Logger


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None or math.isinf(seconds):
        return "calculando..."
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class NormalizationProgressBar:
    """Lightweight textual progress bar for dataset normalization."""

    def __init__(self, total: int, logger: Optional[Logger] = None) -> None:
        self.total = max(1, total)
        self.logger = logger
        self.completed = 0
        self._start = perf_counter()
        self._next_percent = 0.0
        self._log_progress(force=True)

    def advance(self, step: int = 1) -> None:
        self.completed = min(self.total, self.completed + step)
        self._log_progress()

    def finish(self) -> None:
        self.completed = self.total
        self._log_progress(force=True)

    def _eta(self) -> Optional[float]:
        if self.completed == 0:
            return None
        elapsed = perf_counter() - self._start
        if elapsed <= 0:
            return None
        rate = self.completed / elapsed
        if rate <= 0:
            return None
        remaining = (self.total - self.completed) / rate
        return remaining

    def _log_progress(self, force: bool = False) -> None:
        if not self.logger:
            return
        percent = (self.completed / self.total) * 100
        if not force and percent < self._next_percent:
            return

        filled = int(percent / 5)  # 20-character bar
        bar = f"[{'█' * filled}{'░' * (20 - filled)}]"
        eta = _format_duration(self._eta())
        self.logger(f"[NORM] {bar} {percent:5.1f}% ({self.completed}/{self.total}) | ETA: {eta}")
        self._next_percent = min(100.0, math.floor(percent) + 1)
