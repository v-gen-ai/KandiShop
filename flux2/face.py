"""Face refine on FLUX.2 klein: the bench60 recipe on any base image.

Every face is cropped, regenerated img2img with the block-blended dual stream
against its own reference and pasted back. The gen stream copies the crop (the
copy prompt with the crop as its condition), so no scene prompt is needed; the
identity arrives through the copy stream's aligned reference. AdaIN runs on
every blended step. One, two or three people go through the same pass; with
several, ArcFace deals the crops to the references.
"""

from __future__ import annotations

import torch
from PIL import Image

from flux2.generate import generate_dual_stream_flux2
from flux2.pipeline import PROMPT_COPY
from utils import RefineResult, refine_face, refine_faces_multi

# AdaIN over every layer, on every step of the pass.
ADAIN = dict(adain_alpha=1.0, adain_last_layers=-1, adain_stats="mask")


def _snap16(bw: int, bh: int) -> tuple[int, int]:
    """FLUX.2 packs latents in 16px blocks, so crops land on that grid."""
    return max(256, (bw // 16) * 16), max(256, (bh // 16) * 16)


def refine_faces(
    pipeline,
    base_image: Image.Image,
    references: list,
    *,
    steps: int = 4,
    guidance: float = 1.0,
    seed: int = 42,
    denoise: float = 0.75,
    adain: bool = True,
    prompt_copy: str = PROMPT_COPY,
    refine_size: int = 1024,
    crop_scale: float = 2.4,
    mask_shape: str = "ellipse",
    mask_expand: float = 1.6,
    mask_aspect: float = 1.3,
    mask_blur: float = 0.5,
    debug_dir: str | None = None,
    verbose: bool = True,
) -> RefineResult:
    """Refine every face of ``base_image`` against ``references`` (paths or PIL)."""
    refs = [Image.open(r).convert("RGB") if isinstance(r, str) else r.convert("RGB")
            for r in references]
    overrides = ADAIN if adain else {}

    def refine_gen(*, source_image, image_size, init_image, target_face_image,
                   denoise_strength):
        gh, gw = image_size
        g = torch.Generator(device="cuda").manual_seed(int(seed))
        return generate_dual_stream_flux2(
            pipeline, image=source_image,
            prompt_gen=prompt_copy, gen_cond_image=init_image,
            prompt_copy=prompt_copy,
            t_switch=steps, t_end=steps,   # dual driven by the target path
            num_inference_steps=steps, guidance_scale=guidance,
            height=gh, width=gw, generator=g,
            init_image=init_image, denoise_strength=denoise_strength,
            target_face_image=target_face_image,
            mask_shape=mask_shape, mask_expand_factor=mask_expand,
            mask_aspect=mask_aspect, mask_blur_sigma_frac=mask_blur,
            verbose=False, **overrides,
        )

    shared = dict(base_image=base_image, generate=refine_gen, refine_size=refine_size,
                  crop_scale=crop_scale, denoise=denoise, debug_dir=debug_dir,
                  verbose=verbose)
    if len(refs) == 1:
        return refine_face(source_image=refs[0], **shared)
    return refine_faces_multi(source_images=refs, snap_size=_snap16, **shared)
