"""Crop -> regenerate -> paste, for a product instead of a face.

The same three steps as :func:`utils.refine.refine_face` and deliberately a
separate function, because two of the three differ in ways that would turn the
face version into a pile of branches:

* the crop is found by the segmenter, not by a face detector, and its size is
  not a fixed multiple of the object: what matters is how magnified the
  reference lands in a pass whose canvas is pinned at ``refine_size``, so the
  multiplier is derived from the object's size here versus in the reference
  photo (``ObjectSwapSpec.auto_crop_scale``);
* the paste is cut much tighter. A face is composited with a wide fade because
  everything around it -- hair, neck, collar -- belongs to the same person and
  a visible oval is worse than a slightly repainted ear. Around a bottle the
  fade lands on countertop and tile, which the pass has no reason to touch, so
  the opaque core is dilated by a tenth of itself rather than a quarter.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from utils.compositing import composite_refined_face, square_crop_box
from utils.distributed import broadcast_object, is_main
from utils.refine import RefineResult


def refine_object(
    *,
    base_image: Image.Image | None,    # PIL on rank 0; None on the other TP ranks
    source_image,                      # the product photo (path or PIL)
    spec,                              # objects.plan.product_spec(...)
    generate: Callable[..., Any],
    refine_size: int = 1024,
    crop_scale: float = 1.2,
    denoise: float = 0.7,
    composite_strength: float = 1.0,
    composite_mode: str = "falloff",
    composite_dilate_frac: float = 0.10,
    composite_edge_frac: float = 0.08,
    debug_dir: Optional[str] = None,
    verbose: bool = True,
) -> RefineResult:
    """Locate the product in ``base_image``, regenerate it, paste it back.

    Detection and the crop live on rank 0 and are broadcast, same as
    :func:`utils.refine.refine_face`: DINO/SAM is not collective-safe, and a
    divergent ``init_image`` would desynchronise a tensor-parallel generate.
    """
    res = RefineResult()
    source_pil = (Image.open(source_image).convert("RGB")
                  if isinstance(source_image, str) else source_image.convert("RGB"))

    payload = None
    if is_main():
        if base_image is None:
            payload = {"ok": False, "reason": "no base image"}
        else:
            base_pil = base_image.convert("RGB")
            W, H = base_pil.size
            obj = spec.find(base_pil, spec.target_class)
            if obj is None:
                payload = {"ok": False,
                           "reason": f"{spec.target_class!r} not found in the base"}
            else:
                scale, note = spec.auto_crop_scale(
                    source_pil, obj, refine_size, crop_scale)
                side = max(obj.width, obj.height) * float(scale)
                if getattr(spec, "crop_center", "centroid") == "bbox":
                    cx, cy = (obj.bbox[0] + obj.bbox[2]) / 2.0, (obj.bbox[1] + obj.bbox[3]) / 2.0
                else:
                    cx, cy = obj.center[0], obj.center[1]
                box = square_crop_box(float(cx), float(cy), side, W, H)
                crop_up = base_pil.crop(box).resize(
                    (refine_size, refine_size), Image.LANCZOS)
                payload = {
                    "ok": True, "box": box, "crop_up": crop_up, "W": W, "H": H,
                    "scale": float(scale), "note": note or f"scale {scale:.2f}",
                    "label": obj.label, "score": float(obj.score),
                    "size": float(obj.size),
                }
                if verbose:
                    print(f"[object-refine] {obj.label!r} score={obj.score:.2f} "
                          f"size={obj.size:.0f}px crop={box} ({payload['note']}) "
                          f"-> {refine_size}px, denoise={denoise}", flush=True)
    payload = broadcast_object(payload)

    if not payload["ok"]:
        if verbose and is_main():
            print(f"[object-refine] {payload.get('reason')} -> keeping it as is",
                  flush=True)
        res.output_image = base_image
        return res

    crop_up = payload["crop_up"]
    box = payload["box"]
    inner = generate(
        source_image=source_pil,
        image_size=(refine_size, refine_size),
        init_image=crop_up,
        target_face_image=crop_up,
        denoise_strength=denoise,
    )
    res.inner = inner
    res.crop_box = box
    res.refined_crop = inner.output_image

    if not is_main():
        return res

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        crop_up.save(os.path.join(debug_dir, "base_crop.png"))
        if inner.i1_aligned is not None:
            inner.i1_aligned.save(os.path.join(debug_dir, "I0_aligned.png"))
        if inner.output_image is not None:
            inner.output_image.save(os.path.join(debug_dir, "crop_refined.png"))
        if inner.face_mask_pixel is not None:
            Image.fromarray(
                (np.clip(inner.face_mask_pixel, 0, 1) * 255).round().astype("uint8")
            ).save(os.path.join(debug_dir, "mask.png"))
        with open(os.path.join(debug_dir, "paste.json"), "w") as fh:
            json.dump({"crop_box": list(box),
                       "base_size": [payload["W"], payload["H"]],
                       "refine_size": refine_size, "object": spec.target_class,
                       "crop_scale": payload["scale"]}, fh)

    if (inner.output_image is None or not inner.alignment_applied
            or inner.face_mask_pixel is None):
        if verbose:
            print("[object-refine] no alignment -> keeping the base image",
                  flush=True)
        res.output_image = base_image
        return res

    res.output_image = composite_refined_face(
        base_image.convert("RGB"), inner.output_image, inner.face_mask_pixel, box,
        refine_size=refine_size, composite_strength=composite_strength,
        composite_mode=composite_mode,
        composite_dilate_frac=composite_dilate_frac,
        composite_edge_frac=composite_edge_frac,
    )
    res.applied = True
    return res
