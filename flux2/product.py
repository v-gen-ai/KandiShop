"""Product refine on FLUX.2 klein: the objects_bench recipe on any base image.

The product is found in the base by its class name (Grounding DINO + SAM), the
reference is placed onto it (SIFT + MAGSAC), and the crop is regenerated with
the plain block blend: both streams carry the copy prompt, the gen stream
reconstructs the scene crop, the copy stream reproduces the reference.
"""

from __future__ import annotations

import torch
from PIL import Image

from flux2.generate import generate_dual_stream_flux2
from flux2.pipeline import PROMPT_COPY
from objects import ObjectGeometry, object_alignment, object_plan_fn, product_spec, refine_object
from utils import RefineResult


def refine_product(
    pipeline,
    base_image: Image.Image,
    reference,
    object_class: str,
    *,
    steps: int = 4,
    guidance: float = 1.0,
    seed: int = 42,
    denoise: float = 0.7,
    prompt_copy: str = PROMPT_COPY,
    refine_size: int = 1024,
    crop_scale: float = 1.2,
    crop_min: float = 1.0,
    object_dilate_frac: float = 0.04,
    object_blur_frac: float = 0.03,
    object_conf_weight: float = 1.0,
    composite_dilate_frac: float = 0.10,
    composite_edge_frac: float = 0.08,
    debug_dir: str | None = None,
    spec=None,
    verbose: bool = True,
) -> tuple[RefineResult, ObjectGeometry]:
    """Refine the ``object_class`` in ``base_image`` against ``reference`` (path or PIL).

    ``spec`` replaces the refine setting, e.g. :func:`objects.swap_spec` to put
    a different object in its place.
    """
    ref = (Image.open(reference).convert("RGB") if isinstance(reference, str)
           else reference.convert("RGB"))
    if spec is None:
        spec = product_spec(object_class, dilate_frac=object_dilate_frac,
                            blur_sigma_frac=object_blur_frac, conf_weight=object_conf_weight,
                            crop_min=crop_min)
    geometry = ObjectGeometry()
    plan_fn = object_plan_fn(spec, geometry)

    def refine_gen(*, source_image, image_size, init_image, target_face_image,
                   denoise_strength):
        gh, gw = image_size
        return generate_dual_stream_flux2(
            pipeline, image=source_image,
            prompt_gen=prompt_copy, gen_cond_image=init_image,
            prompt_copy=prompt_copy,
            t_switch=steps, t_end=steps, num_inference_steps=steps,
            guidance_scale=guidance, height=gh, width=gw,
            generator=torch.Generator(device="cuda").manual_seed(int(seed)),
            init_image=init_image, denoise_strength=denoise_strength,
            target_face_image=target_face_image, verbose=False,
        )

    with object_alignment(plan_fn, "flux2"):
        rr = refine_object(
            base_image=base_image, source_image=ref, spec=spec, generate=refine_gen,
            refine_size=refine_size, crop_scale=crop_scale, denoise=denoise,
            composite_dilate_frac=composite_dilate_frac,
            composite_edge_frac=composite_edge_frac,
            debug_dir=debug_dir, verbose=verbose,
        )
    return rr, geometry
