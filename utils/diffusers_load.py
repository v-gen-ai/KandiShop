"""Robust diffusers pipeline loading.

Cached but *gated* repos (e.g. ``black-forest-labs/FLUX.2-dev``) make
``from_pretrained`` issue a HEAD request that returns 403 even when every
weight is already on disk. We retry with ``local_files_only=True`` so the
cached copy loads offline instead of failing on the gate.
"""

from __future__ import annotations

from typing import Any


def load_pipeline(pipeline_cls, model_id: str, *, torch_dtype=None, **kwargs) -> Any:
    """``pipeline_cls.from_pretrained`` with an automatic offline fallback."""
    try:
        return pipeline_cls.from_pretrained(model_id, torch_dtype=torch_dtype, **kwargs)
    except Exception as e:  # gated / network / offline HEAD failures
        msg = f"{type(e).__name__}: {e}"
        if any(tok in msg for tok in ("403", "GatedRepo", "Forbidden", "ConnectionError",
                                      "Offline", "Could not", "HTTPError", "Repository Not")):
            print(f"[load] {model_id}: {type(e).__name__} -> retrying with local_files_only=True",
                  flush=True)
            return pipeline_cls.from_pretrained(
                model_id, torch_dtype=torch_dtype, local_files_only=True, **kwargs
            )
        raise
