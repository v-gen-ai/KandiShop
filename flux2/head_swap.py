"""Head swap on FLUX.2 klein: put the reference's head (face and hair) on the photo.

The same crop -> dual-stream img2img -> paste loop as the face refine
(:func:`utils.refine_face`), with two things changed. The blend mask is a parsed
head silhouette from :mod:`utils.head_mask` instead of the eye-sized ellipse,
so it reaches the hair. And the paste-back covers the whole crop: the head that
leaves and the head that arrives have different outlines, so a paste cut to
either one leaves a rim of the old head behind.

Both streams get the copy prompt; the gen stream sees the scene crop and the
copy stream the reference aligned into it. Dual stream is on from the first
executed step (``t_switch=t_end=steps``).
"""

from __future__ import annotations

from contextlib import contextmanager

import cv2
import numpy as np
import torch
from PIL import Image

from flux2.generate import generate_dual_stream_flux2
from flux2.pipeline import PROMPT_COPY
from utils import AlignmentPlan, align_face_to_target, detect_face_robust, pixel_mask_to_token_mask
from utils.head_mask import MASK_MODES, HeadMaskBuilder

COMPOSITE_AREAS = ("crop", "mask")


def align_reference(target: Image.Image, reference: Image.Image, *,
                    background_mode: str = "gray", align_mode: str = "similarity_5pt",
                    scale_clamp: tuple[float, float] = (0.05, 20.0)) -> Image.Image:
    """Warp ``reference`` so its face sits on ``target``'s, in ``target``'s frame.

    ``background_mode``: ``gray`` canvas, ``x0``/``target`` (the target photo
    behind the warp) or ``reflect`` (no canvas, border reflection).
    """
    face_target = detect_face_robust(target)
    if face_target is None:
        raise ValueError("no face detected in the target image")
    face_source = detect_face_robust(reference)
    if face_source is None:
        raise ValueError("no face detected in the reference image")
    w, h = target.size
    if background_mode in ("x0", "target"):
        background = target
    elif background_mode == "gray":
        background = Image.new("RGB", (w, h), (128, 128, 128))
    elif background_mode == "reflect":
        background = None
    else:
        raise ValueError(f"unknown background_mode: {background_mode!r}")
    aligned = align_face_to_target(
        source=reference, target_face=face_target, source_face=face_source,
        output_size=(w, h), background=background, scale_clamp=scale_clamp, mode=align_mode,
    )
    if aligned is None:
        raise ValueError("alignment failed for the reference image")
    return aligned


@contextmanager
def _alignment(plan_fn):
    """Rebind the generator's ``plan_alignment`` (the ellipse mask) for the call."""
    from flux2 import generate as backend
    original = backend.plan_alignment
    backend.plan_alignment = plan_fn
    try:
        yield
    finally:
        backend.plan_alignment = original


def _crop_alpha(height: int, width: int, margin_frac: float) -> np.ndarray:
    """The whole crop inset by a margin, so the paste feather runs out inside the
    crop instead of sitting at half strength on its border."""
    alpha = np.zeros((height, width), dtype=np.float32)
    dy, dx = int(round(margin_frac * height)), int(round(margin_frac * width))
    alpha[dy:height - dy, dx:width - dx] = 1.0
    return alpha


def make_head_swap_generate(
    pipeline,
    mask_builder: HeadMaskBuilder,
    *,
    prompt_target: str = PROMPT_COPY,
    prompt_reference: str = PROMPT_COPY,
    steps: int = 4,
    guidance_scale: float = 1.0,
    seed: int = 0,
    background_mode: str = "gray",
    align_mode: str = "similarity_5pt",
    scale_clamp: tuple[float, float] = (0.05, 20.0),
    include_hair: bool = True,
    mask_mode: str = "union",
    composite_area: str = "crop",
    composite_margin_frac: float = 0.04,
    crop_scale: float = 4.4,
    blur_sigma_frac: float = 0.15,
    dilate_frac: float = 0.06,
    adain_alpha: float = 1.0,
    adain_last_layers: int = -1,
    adain_stats: str = "mask",
    adain_skip_first_steps: int = 1,
    device: str = "cuda",
    verbose: bool = True,
):
    """The ``generate`` closure :func:`utils.refine_face` calls on the crop.

    ``mask_mode`` decides what the two streams blend: ``union`` takes both heads
    whole and is the one that works when the photo has more hair than the
    reference (under ``parsing`` the photo's own face is partly outside the
    mask, and what the mask does not own survives the pass). ``composite_area``
    decides what is written back: ``crop`` (the whole crop, inset by
    ``composite_margin_frac``) or ``mask`` (the head mask, as in a refine).
    ``crop_scale`` here sizes the parsing crop inside the refine crop.
    """
    if mask_mode not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of {MASK_MODES}, got {mask_mode!r}")
    if composite_area not in COMPOSITE_AREAS:
        raise ValueError(f"composite_area must be one of {COMPOSITE_AREAS}, got {composite_area!r}")

    def plan_fn(source, target, *, width, height, token_h, token_w, **_):
        try:
            aligned = align_reference(target, source, background_mode=background_mode,
                                      align_mode=align_mode, scale_clamp=scale_clamp)
        except ValueError as e:
            if verbose:
                print(f"  [head-swap] {e}")
            return AlignmentPlan(ok=False)
        mask = mask_builder.build(mask_mode, target, aligned, include_hair=include_hair,
                                  crop_scale=crop_scale, blur_sigma_frac=blur_sigma_frac,
                                  dilate_frac=dilate_frac)
        if aligned.size != (width, height):
            aligned = aligned.resize((width, height), Image.LANCZOS)
        if mask.shape[:2] != (height, width):
            mask = np.clip(cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
        return AlignmentPlan(
            ok=True,
            i1_aligned=aligned,
            mask_pixel=mask,
            mask_tokens=pixel_mask_to_token_mask(mask, token_h, token_w),
            composite_pixel=(mask if composite_area == "mask"
                             else _crop_alpha(height, width, composite_margin_frac)),
        )

    def _generate(*, source_image, image_size, init_image, target_face_image, denoise_strength):
        height, width = image_size
        reference = (Image.open(source_image) if isinstance(source_image, str)
                     else source_image).convert("RGB")
        init = init_image.convert("RGB")
        with _alignment(plan_fn):
            return generate_dual_stream_flux2(
                pipeline,
                image=reference,
                prompt_gen=prompt_target,
                prompt_copy=prompt_reference,
                t_switch=steps,
                t_end=steps,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=torch.Generator(device=device).manual_seed(int(seed)),
                background_mode=background_mode,
                align_mode=align_mode,
                scale_clamp=scale_clamp,
                init_image=init,
                denoise_strength=denoise_strength,
                target_face_image=target_face_image.convert("RGB"),
                gen_cond_image=init,
                adain_alpha=adain_alpha,
                adain_last_layers=adain_last_layers,
                adain_stats=adain_stats,
                adain_skip_first_steps=adain_skip_first_steps,
                verbose=verbose,
            )

    return _generate
