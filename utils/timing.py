"""Tiny stage timer so every launcher reports comparable per-image cost.

Usage::

    timer = StageTimer("flux2")
    with timer.stage("base"):
        ... base generation ...
    with timer.stage("refine"):
        ... refinement ...
    print(timer.summary())   # -> TIMING flux2 base=.. refine=.. total=..
"""

from __future__ import annotations

import time
from contextlib import contextmanager


class StageTimer:
    def __init__(self, name: str):
        self.name = name
        self.stages: dict[str, float] = {}
        self._t0 = time.time()

    @contextmanager
    def stage(self, key: str):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.stages[key] = self.stages.get(key, 0.0) + (time.time() - t0)

    @property
    def total(self) -> float:
        return time.time() - self._t0

    def summary(self) -> str:
        parts = " ".join(f"{k}={v:.1f}s" for k, v in self.stages.items())
        return f"TIMING {self.name} {parts} total={self.total:.1f}s"
