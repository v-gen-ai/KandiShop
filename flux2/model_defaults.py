"""Per-checkpoint sampling defaults for FLUX.2.

The distilled ``klein-4B`` / ``klein-9B`` are **step-distilled to 4 inference
steps** and expect ``guidance_scale=1.0`` (no CFG). The un-distilled base flow
models (``klein-base-*``) and the guidance-distilled ``dev`` / ``-pro`` use the
usual ~40 steps with ``guidance_scale~=4``. These helpers let the launchers pick
sane defaults from the model id when the user does not override them.
"""

from __future__ import annotations


def is_step_distilled(model_id: str) -> bool:
    mid = model_id.lower()
    # "klein-4B"/"klein-9B" are distilled; "klein-base-*" is the raw flow model.
    return "klein" in mid and "base" not in mid


def flux2_step_defaults(model_id: str) -> tuple[int, float]:
    """Return ``(num_inference_steps, guidance_scale)`` recommended for a FLUX.2 id."""
    if is_step_distilled(model_id):
        return 4, 1.0
    return 40, 4.0
