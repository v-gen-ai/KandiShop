"""Tensor-parallel helpers. All degrade to no-ops without ``torch.distributed``.

The single-GPU backbones (FLUX.2, Qwen) never initialise a process group, so
:func:`rank` returns 0 and :func:`broadcast_object` is the identity. The
Hunyuan backbone runs under ``torchrun`` with a real group, where face
detection / alignment / crop planning happen on rank 0 and are broadcast so
every rank feeds an identical conditioning (a divergent prefill would corrupt
the tensor-parallel all-reduce inside attention).
"""

from __future__ import annotations

from typing import Any

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None


def dist_active() -> bool:
    return dist is not None and dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if dist_active() else 0


def is_main() -> bool:
    return rank() == 0


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast a picklable object from ``src`` to all ranks (identity if the
    process group is not initialised)."""
    if not dist_active():
        return obj
    box = [obj if rank() == src else None]
    dist.broadcast_object_list(box, src=src)
    return box[0]
