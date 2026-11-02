"""Backbone-agnostic core of the TrueFace face-preserving method.

Everything here is independent of any specific image backbone. A backbone
package (``flux2``, ``qwen_image_edit``, ``hunyuan_image_3``) supplies a
:class:`DualStreamAdapter` and an img2img ``generate`` closure, and reuses this
package for face detection/alignment, mask construction, the block-level blend
driver, and the refinement/compositing orchestration.
"""

from __future__ import annotations

from .config import AlignConfig, DualStreamConfig, MaskConfig, RefineConfig
from .distributed import broadcast_object, dist_active, is_main, rank
from .dual_stream import (
    DivergenceGate,
    DualStreamAdapter,
    MaskPerturb,
    dit_dual_stream_velocity,
)
from .face_alignment import AlignMode, align_face_to_target, compute_alignment_transform
from .face_detection import (
    FaceInfo,
    detect_face_robust,
    detect_faces,
    detect_primary_face,
)
from .face_mask import (
    build_face_contour_mask_pixel,
    build_face_mask_pixel,
    build_face_token_mask,
    pixel_mask_to_token_mask,
)
from .align_pass import AlignmentPlan, plan_alignment
from .compositing import (
    clip_box_to_neighbors,
    composite_refined_face,
    face_crop_boxes,
    square_crop_box,
)
from .arcface import (
    assign_faces_to_refs,
    detect_faces_with_embeddings,
    primary_embedding,
)
from .refine import RefinedFace, RefineResult, refine_face, refine_faces_multi
from .diffusers_load import load_pipeline
from .timing import StageTimer

__all__ = [
    # config
    "DualStreamConfig", "RefineConfig", "MaskConfig", "AlignConfig",
    # distributed
    "dist_active", "rank", "is_main", "broadcast_object",
    # dual-stream core
    "DualStreamAdapter", "dit_dual_stream_velocity", "DivergenceGate", "MaskPerturb",
    # face utils
    "FaceInfo", "detect_faces", "detect_primary_face", "detect_face_robust",
    "AlignMode", "align_face_to_target", "compute_alignment_transform",
    "build_face_mask_pixel", "build_face_contour_mask_pixel",
    "pixel_mask_to_token_mask", "build_face_token_mask",
    "plan_alignment", "AlignmentPlan",
    "square_crop_box", "clip_box_to_neighbors", "face_crop_boxes",
    "composite_refined_face",
    # arcface
    "detect_faces_with_embeddings", "primary_embedding", "assign_faces_to_refs",
    # refine
    "refine_face", "refine_faces_multi", "RefineResult", "RefinedFace",
    # loading
    "load_pipeline",
    # timing
    "StageTimer",
]
