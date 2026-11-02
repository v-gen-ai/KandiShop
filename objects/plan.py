"""An :class:`~utils.align_pass.AlignmentPlan` for an object instead of a face.

The dual stream asks the geometry for exactly three things, and never asks how
they were obtained: the reference rendered on the output grid where the target
sits (``i1_aligned``), the token mask that says which tokens the copy stream
owns (``mask_tokens``), and the pixel mask the paste is cut from
(``composite_pixel``). The face path answers with a similarity fit to five
landmarks and an ellipse. Here the same three come from the object matcher that
was written for Hunyuan and is reused verbatim (``object_segmentation``):

* **where** -- Grounding DINO turns the class string into a box, SAM turns the
  box into a silhouette, on both the scene crop and the reference photo;
* **how** -- SIFT keypoints inside the two silhouettes, matched by descriptor
  and filtered by MAGSAC, give a similarity or a homography; without enough
  agreeing points the placement falls back to equal areas, equal centroids;
* **which tokens** -- the two silhouettes intersected, dilated, feathered, and
  multiplied by the inlier density, so the copy stream only owns the parts of
  the object the fit is actually supported at. A cap with no printing on it has
  no keypoints, lands wrong under a fit made on the label, and is exactly what
  the confidence weighting drops back to the scene.

The backend is not modified. :func:`object_alignment` rebinds the name
``plan_alignment`` inside the backend's generate module for the duration of a
block and restores it after, which is why nothing in ``utils`` or ``flux2``
needs an object branch.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from PIL import Image

from utils.align_pass import AlignmentPlan
from utils.face_mask import pixel_mask_to_token_mask


def _k6_object_module():
    from . import object_segmentation  # noqa: PLC0415  (Grounding DINO and SAM load lazily)

    return object_segmentation


def product_spec(
    object_class: str,
    *,
    dilate_frac: float = 0.04,
    blur_sigma_frac: float = 0.03,
    conf_weight: float = 1.0,
    allow_homography: bool = True,
    box_threshold: float = 0.30,
    crop_target_scale: float = 1.35,
    crop_min: float = 1.1,
    crop_max: float = 2.4,
):
    """The refine-the-same-object setting of ``ObjectSwapSpec``.

    ``size_ratio=1`` and ``align="matcher"`` together mean "this is the same
    object twice, find the transform": the spec then intersects the silhouettes
    rather than unioning them, which is the difference between refining a
    product and swapping one for another.
    """
    return _k6_object_module().ObjectSwapSpec(
        target_class=object_class,
        source_class=object_class,
        size_ratio=1.0,
        align="matcher",
        allow_homography=allow_homography,
        isolate=True,
        box_threshold=box_threshold,
        dilate_frac=dilate_frac,
        blur_sigma_frac=blur_sigma_frac,
        conf_weight=conf_weight,
        crop_auto=True,
        crop_target_scale=crop_target_scale,
        crop_min=crop_min,
        crop_max=crop_max,
    )


def swap_spec(
    object_class: str,
    source_class: str | None = None,
    *,
    size_metric: str = "bbox_width",
    size_ratio: float = 1.0,
    place: str = "bbox",
    dilate_frac: float = 0.04,
    blur_sigma_frac: float = 0.03,
    box_threshold: float = 0.30,
    crop_min: float = 1.2,
    crop_max: float = 2.4,
):
    """The replace-with-another-object setting of ``ObjectSwapSpec``.

    A different object has no correspondences to fit, so it is placed by
    centroid and scaled so that ``size_metric`` comes out ``size_ratio`` times
    the old one's (``bbox_width``: equal widths). The silhouettes are unioned:
    the old object has to vanish where it sticks out past the new one.
    ``place`` is what is made to coincide: the bbox centres, or (``"bottom"``)
    the bottom edges, for a smaller object standing on the same surface. The
    crop sits on the object's box with at least ``crop_min`` of its long side,
    so both objects fit in it whole.
    """
    return _k6_object_module().ObjectSwapSpec(
        target_class=object_class,
        source_class=source_class or object_class,
        size_ratio=size_ratio,
        size_metric=size_metric,
        align="centroid",
        isolate=True,
        box_threshold=box_threshold,
        dilate_frac=dilate_frac,
        blur_sigma_frac=blur_sigma_frac,
        conf_weight=0.0,
        crop_auto=True,
        crop_min=crop_min,
        crop_max=crop_max,
        crop_center="bbox",
        place_center=place,
    )


@dataclass
class ObjectGeometry:
    """The last plan built, kept for the debug dump and for the log line."""
    ok: bool = False
    rung: str | None = None
    inliers: int = 0
    scale: float = 0.0
    mask_mean: float = 0.0
    aligned: Image.Image | None = None
    mask: np.ndarray | None = None


def object_plan_fn(spec, geometry: ObjectGeometry | None = None):
    """A drop-in replacement for :func:`utils.plan_alignment`, for one product.

    Signature-compatible by construction: the backends call it positionally with
    ``(source, target)`` and hand it every face mask knob as a keyword, all of
    which are meaningless here and swallowed. The result is cached on the pair
    of images it was built from, because the six blend variants of one scene ask
    for the identical plan and neither DINO nor SAM is cheap.
    """
    cache: dict = {}

    def plan_objects(
        source_image: Image.Image,
        target_image: Image.Image,
        *,
        width: int,
        height: int,
        token_h: int,
        token_w: int,
        background_mode: str = "gray",
        background_image: Image.Image | None = None,
        **_face_only,
    ) -> AlignmentPlan:
        key = (hash(source_image.tobytes()), hash(target_image.tobytes()),
               width, height, token_h, token_w, background_mode)
        hit = cache.get(key)
        if hit is None:
            background = (background_image if background_mode == "x0"
                          else Image.new("RGB", (width, height), (128, 128, 128)))
            swap = spec.plan(target_image.convert("RGB"), source_image.convert("RGB"),
                             (width, height), background=background)
            if swap is None:
                hit = AlignmentPlan()
                if geometry is not None:
                    geometry.ok = False
            else:
                mask = np.clip(np.asarray(swap.mask, dtype=np.float32), 0.0, 1.0)
                hit = AlignmentPlan(
                    ok=True,
                    i1_aligned=swap.aligned,
                    mask_pixel=mask,
                    mask_tokens=pixel_mask_to_token_mask(mask, token_h, token_w),
                    composite_pixel=mask,
                )
                if geometry is not None:
                    fit = swap.fit
                    geometry.ok = True
                    geometry.rung = getattr(fit, "rung", "centroid")
                    geometry.inliers = int(len(getattr(fit, "inlier_xy", ())))
                    geometry.scale = float(swap.scale)
                    geometry.mask_mean = float(mask.mean())
                    geometry.aligned, geometry.mask = swap.aligned, mask
            cache.clear()          # one scene at a time; do not hold on to crops
            cache[key] = hit
        return hit

    return plan_objects


@contextmanager
def object_alignment(plan_fn, backend: str = "flux2"):
    """Run the block with the backend's alignment pass replaced by ``plan_fn``.

    A rebinding rather than an argument: the face path is mid-benchmark and the
    point of doing it this way is that not one line of it changes. Restored on
    the way out, including after an exception.
    """
    if backend == "flux2":
        from flux2 import generate as backend_module
    else:
        raise ValueError(f"unknown backend {backend!r}")
    original = backend_module.plan_alignment
    backend_module.plan_alignment = plan_fn
    try:
        yield
    finally:
        backend_module.plan_alignment = original
