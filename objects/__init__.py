"""The object track of the method, kept apart from the face one on purpose.

Everything the face path needs to start a dual stream -- where the thing is,
how the reference maps onto it, which tokens the copy stream owns -- comes from
five landmarks with known meaning. An arbitrary product has none of that, so
this package supplies the same three answers by other means: Grounding DINO +
SAM for the silhouette, SIFT + RANSAC for the placement, inlier density for the
mask. What comes after -- the per-block blend, the divergence gate, the paste --
is the code the faces already use, untouched.

Nothing here is imported by the face path. The backbone hook is a temporary
rebinding of :func:`utils.plan_alignment` inside the backend module, held only
for the duration of :func:`objects.plan.object_alignment`, so a face run in the
same process (or, as it happens, on the other seven GPUs) sees the original.
"""

from .plan import ObjectGeometry, object_alignment, object_plan_fn, product_spec, swap_spec
from .refine import refine_object

__all__ = [
    "ObjectGeometry", "object_alignment", "object_plan_fn", "product_spec",
    "refine_object", "swap_spec",
]
