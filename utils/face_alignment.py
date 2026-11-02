"""Warp the identity reference so its face aligns with a target face.

Main entry point: :func:`align_face_to_target`. Three modes (see
:func:`compute_alignment_transform`):

- ``"similarity_5pt"`` (default) - 4-DOF similarity (uniform scale + rotation +
  translation) fit by least squares to the 5 landmarks. Handles in-plane tilt.
- ``"eye_line"`` - closed-form: scale from inter-ocular distance, rotation from
  the eye axis, translation from the face center. Robust fallback.
- ``"scale_translate"`` - legacy (scale + translation only, no rotation).
"""

from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np
from PIL import Image

from .face_detection import FaceInfo, detect_primary_face

AlignMode = Literal["similarity_5pt", "eye_line", "scale_translate"]


def _eye_line_transform(
    source_face: FaceInfo, target_face: FaceInfo,
) -> tuple[np.ndarray, float]:
    """Closed-form similarity from eye axis + IOD + face center."""
    s_lm = np.asarray(source_face.landmarks, dtype=np.float64)
    t_lm = np.asarray(target_face.landmarks, dtype=np.float64)
    src_angle = math.atan2(s_lm[1, 1] - s_lm[0, 1], s_lm[1, 0] - s_lm[0, 0])
    tgt_angle = math.atan2(t_lm[1, 1] - t_lm[0, 1], t_lm[1, 0] - t_lm[0, 0])
    theta = tgt_angle - src_angle
    scale = float(target_face.size / source_face.size)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    a, b = scale * cos_t, -scale * sin_t
    c, d = scale * sin_t, scale * cos_t
    cx, cy = float(source_face.center[0]), float(source_face.center[1])
    tx = float(target_face.center[0]) - (a * cx + b * cy)
    ty = float(target_face.center[1]) - (c * cx + d * cy)
    return np.array([[a, b, tx], [c, d, ty]], dtype=np.float64), scale


def compute_alignment_transform(
    source_face: FaceInfo,
    target_face: FaceInfo,
    output_size: tuple[int, int],
    mode: AlignMode = "similarity_5pt",
) -> tuple[np.ndarray, float]:
    """Compute a 2x3 affine matrix mapping *source* image coords onto
    *target_face*. Returns ``(M, effective_scale)``."""
    if mode == "similarity_5pt":
        src_lm = np.asarray(source_face.landmarks, dtype=np.float32)
        dst_lm = np.asarray(target_face.landmarks, dtype=np.float32)
        M, _ = cv2.estimateAffinePartial2D(src_lm, dst_lm, method=cv2.LMEDS)
        if M is not None:
            scale = math.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
            return M.astype(np.float64), float(scale)
        # Fall through to eye_line on degenerate input.

    if mode in ("eye_line", "similarity_5pt"):
        return _eye_line_transform(source_face, target_face)

    # mode == "scale_translate" (legacy: no rotation)
    scale = float(target_face.size / source_face.size)
    tx = target_face.center[0] - source_face.center[0] * scale
    ty = target_face.center[1] - source_face.center[1] * scale
    M = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float64)
    return M, scale


def align_face_to_target(
    source: Image.Image,
    target_face: FaceInfo,
    source_face: FaceInfo | None = None,
    output_size: tuple[int, int] = (1024, 1024),
    background: Image.Image | None = None,
    scale_clamp: tuple[float, float] = (0.05, 20.0),
    mode: AlignMode = "similarity_5pt",
) -> Image.Image | None:
    """Warp *source* (I1) so its face matches *target_face*.

    ``output_size`` is ``(width, height)``. ``background`` (if given) is the
    canvas behind the warped source (e.g. the x0 estimate). Returns the aligned
    PIL image, or None if source-face detection fails.
    """
    if source_face is None:
        source_face = detect_primary_face(source)
    if source_face is None:
        return None

    M, scale = compute_alignment_transform(
        source_face, target_face, output_size, mode=mode,
    )

    # Clamp extreme scales: scale the 2x2 block uniformly, then recompute the
    # translation so the source face center still maps to the target center.
    if not (scale_clamp[0] <= scale <= scale_clamp[1]):
        clamped = max(scale_clamp[0], min(scale, scale_clamp[1]))
        ratio = clamped / scale
        M[:, :2] *= ratio
        cx, cy = float(source_face.center[0]), float(source_face.center[1])
        M[0, 2] = target_face.center[0] - (M[0, 0] * cx + M[0, 1] * cy)
        M[1, 2] = target_face.center[1] - (M[1, 0] * cx + M[1, 1] * cy)
        scale = clamped

    w, h = output_size
    source_bgr = np.array(source.convert("RGB"))[:, :, ::-1].copy()

    if background is not None:
        canvas = np.array(background.convert("RGB").resize((w, h)))[:, :, ::-1].copy()
        cv2.warpAffine(
            source_bgr, M, (w, h), dst=canvas,
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_TRANSPARENT,
        )
        result_bgr = canvas
    else:
        result_bgr = cv2.warpAffine(
            source_bgr, M, (w, h),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101,
        )

    return Image.fromarray(result_bgr[:, :, ::-1])
