"""One alignment pass, shared by every backbone.

The geometry has always been shared (:mod:`utils.face_detection`,
:mod:`utils.face_alignment`, :mod:`utils.face_mask`); what each backbone used
to repeat for itself was the short wrapper around it -- detect both faces, warp
the reference onto a canvas, build the mask -- and that is where the models
drift apart, because the canvas is not obviously the same thing in each of
them. FLUX.2 carries a single condition frame on the output token grid; Qwen
carries a list of condition frames, each at its own VAE resolution, and feeds
the same images to a vision-language text encoder as well.

The invariant that must hold everywhere is the one this function enforces: the
aligned reference is rendered on the **output** grid, because the target
landmarks it is fitted to were read there. Encoded at the reference's own
aspect instead, the face lands shifted and rescaled with respect to the mask.
What stays per backbone is only the last step -- turning this PIL image into
whatever that model calls a condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .face_alignment import AlignMode, align_face_to_target
from .face_detection import FaceInfo, detect_face_robust
from .face_mask import build_face_token_mask


@dataclass
class AlignmentPlan:
    """Everything the dual stream needs to start, or ``ok=False``.

    ``mask_tokens`` drives the per-block blend and ``composite_pixel`` decides
    what is written back into the frame afterwards. They are the same mask
    unless asked otherwise, but they answer different questions: the blend wants
    room around the face, because a token at the edge of the mask still attends
    to its neighbours and a mask cut tight to the jaw makes the two streams
    fight over the boundary; the composite wants the opposite, since every pixel
    it takes beyond the face is the reference's hair and lighting landing in a
    scene that did not ask for them.
    """
    ok: bool = False
    i1_aligned: Image.Image | None = None
    mask_pixel: np.ndarray | None = None
    mask_tokens: torch.Tensor | None = None
    composite_pixel: np.ndarray | None = None
    face_source: FaceInfo | None = None
    face_target: FaceInfo | None = None


def plan_alignment(
    source_image: Image.Image,
    target_image: Image.Image,
    *,
    width: int,
    height: int,
    token_h: int,
    token_w: int,
    background_mode: str = "gray",
    background_image: Image.Image | None = None,
    scale_clamp: tuple[float, float] = (0.05, 20.0),
    align_mode: AlignMode = "similarity_5pt",
    mask_expand_factor: float = 1.6,
    mask_aspect: float = 1.3,
    mask_blur_sigma_frac: float = 0.5,
    mask_align_to_eyes: bool = True,
    mask_shape: str = "ellipse",
    mask_forehead: float = 0.55,
    mask_grow: float = 0.0,
    composite_mask_shape: str | None = None,
    composite_mask_expand_factor: float = 1.6,
    composite_mask_aspect: float = 1.3,
    composite_mask_blur_sigma_frac: float = 0.08,
    composite_mask_forehead: float = 0.55,
    composite_mask_grow: float = 0.0,
) -> AlignmentPlan:
    """Warp ``source_image``'s face onto the face found in ``target_image``.

        ``target_image`` is the frame the result has to match: the scene crop in a
        refinement pass, or the x0 probe when the base pass switches over.
        ``scale_clamp`` only guards against a detection failure blowing the warp
        up: a full-length scene legitimately needs the reference shrunk ten-fold,
        so a floor anywhere near 1 silently pastes an oversized face.
    ``background_image`` is only read when ``background_mode="x0"``; the default
    grey fill keeps the reference's own surroundings from bleeding through the
    soft edge of the mask.
    """
    f_tgt = detect_face_robust(target_image)
    f_src = detect_face_robust(source_image)
    if f_tgt is None or f_src is None:
        return AlignmentPlan(face_source=f_src, face_target=f_tgt)

    if background_mode == "x0":
        background = background_image
    elif background_mode == "gray":
        background = Image.new("RGB", (width, height), (128, 128, 128))
    else:
        background = None

    i1_aligned = align_face_to_target(
        source=source_image, target_face=f_tgt, source_face=f_src,
        output_size=(width, height), background=background,
        scale_clamp=scale_clamp, mode=align_mode,
    )
    if i1_aligned is None:
        return AlignmentPlan(face_source=f_src, face_target=f_tgt)

    mask_pixel, mask_tokens = build_face_token_mask(
        f_tgt, height=height, width=width, token_h=token_h, token_w=token_w,
        expand_factor=mask_expand_factor, aspect=mask_aspect,
        blur_sigma_frac=mask_blur_sigma_frac, align_to_eyes=mask_align_to_eyes,
        shape=mask_shape, image=target_image, forehead=mask_forehead, grow=mask_grow,
    )

    composite_pixel = mask_pixel
    if composite_mask_shape is not None:
        composite_pixel, _ = build_face_token_mask(
            f_tgt, height=height, width=width, token_h=token_h, token_w=token_w,
            expand_factor=composite_mask_expand_factor, aspect=composite_mask_aspect,
            blur_sigma_frac=composite_mask_blur_sigma_frac,
            align_to_eyes=mask_align_to_eyes, shape=composite_mask_shape,
            image=target_image, forehead=composite_mask_forehead,
            grow=composite_mask_grow,
        )
    return AlignmentPlan(
        ok=True, i1_aligned=i1_aligned, mask_pixel=mask_pixel,
        mask_tokens=mask_tokens, composite_pixel=composite_pixel,
        face_source=f_src, face_target=f_tgt,
    )
