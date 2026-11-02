"""Unified configuration for the face-preserving method (backbone-agnostic).

Two dataclasses hold every knob the method exposes, with identical semantics
across all backbones (FLUX.2, Qwen-Image-Edit, HunyuanImage-3). Backbone
launchers parse CLI flags into these and pass them down; the shared method code
in :mod:`utils.dual_stream` and :mod:`utils.refine` reads only these.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DualStreamConfig:
    """Controls dual-stream blending during the BASE generation.

    Dual-stream can be disabled entirely (the common case: the base is a plain
    single-stream generation and all identity work happens in the refinement
    pass). When enabled, blending runs over the step window
    ``[start_frac, end_frac)`` and the layer window ``[layer_start, layer_end)``.
    """

    # Fraction of steps at which dual-stream begins during the base pass.
    # 1.0 (default) => never (single-stream base). 0.25 => start at 25% of steps.
    start_frac: float = 1.0
    # Fraction of steps at which dual-stream reverts to single-stream.
    # 1.0 (default) => until the last step.
    end_frac: float = 1.0
    # Layer window for the per-block blend, as fractions of the layer count.
    # (0.0, 1.0) blends every block; (0.5, 1.0) only the second half (avoids
    # background artifacts on some backbones).
    layer_start: float = 0.0
    layer_end: float = 1.0

    @property
    def enabled(self) -> bool:
        """Dual-stream is active in the base pass iff it starts before the end."""
        return self.start_frac < 1.0

    def step_window(self, num_steps: int) -> tuple[int, int]:
        """Absolute ``[t_start, t_end)`` step indices for ``num_steps`` steps."""
        t_start = int(round(self.start_frac * num_steps))
        t_end = int(round(self.end_frac * num_steps))
        t_start = max(0, min(t_start, num_steps))
        t_end = max(t_start + 1, min(t_end, num_steps))
        return t_start, t_end

    def layer_window(self, num_layers: int) -> tuple[int, int]:
        """Absolute ``[l_start, l_end)`` layer indices for ``num_layers`` layers."""
        lo = int(round(self.layer_start * num_layers))
        hi = int(round(self.layer_end * num_layers))
        return lo, hi


@dataclass
class RefineConfig:
    """Controls the high-resolution face-refinement pass.

    The refinement crops each detected face, upscales it to ``size``, and runs an
    img2img dual-stream pass that starts ``denoise`` of the way back through the
    schedule (``denoise=0.5`` => start at half the steps; ``0.75`` => 75% back),
    then composites the result through a feathered elliptical mask.
    """

    enabled: bool = True
    # img2img strength: fraction of steps to run "backwards". 0.5 = half.
    denoise: float = 0.5
    # Resolution the face crop is upscaled to for the refine pass.
    size: int = 1024
    # Crop side = max(face_w, face_h) * crop_scale.
    crop_scale: float = 2.4
    # Alpha multiplier when compositing the refined face back.
    composite_strength: float = 1.0
    # Feather radius for the composite alpha, as a fraction of ``size``.
    composite_feather_frac: float = 0.05
    # Skip refinement if the face is already larger than this fraction of the
    # image side (None => never skip). The base pass handles big faces well.
    skip_if_face_frac_gt: float | None = None
    # Multi-person: minimum ArcFace cosine similarity to accept a face<->ref
    # match (only enforced when there are more faces than references).
    min_sim: float = 0.05


@dataclass
class MaskConfig:
    """Soft elliptical face-mask geometry (shared by base blend + refine)."""

    expand_factor: float = 1.6      # ellipse semi-axis in units of inter-ocular distance
    aspect: float = 1.3             # long/short axis ratio (taller than wide)
    blur_sigma_frac: float = 0.5    # edge blur in units of inter-ocular distance
    align_to_eyes: bool = True      # rotate the ellipse to the eye line


@dataclass
class AlignConfig:
    """Face-alignment (reference warp) parameters."""

    mode: str = "similarity_5pt"    # see utils.face_alignment.AlignMode
    scale_clamp: tuple[float, float] = (0.05, 20.0)
    background_mode: str = "gray"    # "gray" | "x0" | "reflect" (gray avoids copy-bleed artifacts)
