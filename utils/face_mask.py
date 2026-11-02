"""Soft face masks: pixel space and token-grid space.

The mask decides which tokens the copy stream owns, both for the per-block
dual-stream blend (resampled to the transformer's image-token grid) and for the
final composite of a refined crop. Two shapes are available. The ellipse is
sized in inter-ocular distances and aligned to the eye line: simple, but it is
a stand-in for a face rather than a face, so it always spends part of its area
on hair and background. The contour traces the actual outline from the 106
dense landmarks, which is what to use when the reference's hair must not come
along with its face.

Geometry is backbone-independent; only the token grid shape differs per model.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from PIL import Image

from .face_detection import FaceInfo


def _eye_line_angle_deg(face: FaceInfo) -> float:
    lm = np.asarray(face.landmarks, dtype=np.float64)
    dx = lm[1, 0] - lm[0, 0]
    dy = lm[1, 1] - lm[0, 1]
    return float(math.degrees(math.atan2(dy, dx)))


def build_face_mask_pixel(
    face: FaceInfo,
    height: int,
    width: int,
    *,
    expand_factor: float = 1.6,
    aspect: float = 1.3,
    blur_sigma_frac: float = 0.5,
    align_to_eyes: bool = True,
) -> np.ndarray:
    """Soft elliptic face mask in pixel space, values in ``[0, 1]``.

    Semi-axes are ``face.size * expand_factor`` (short) and that times
    ``aspect`` (long, i.e. taller than wide), rotated to the eye line, then
    Gaussian-blurred by ``face.size * blur_sigma_frac``.
    """
    cx, cy = float(face.center[0]), float(face.center[1])
    half_short = max(1.0, face.size * expand_factor)
    half_long = max(1.0, face.size * expand_factor * aspect)
    angle = _eye_line_angle_deg(face) if align_to_eyes else 0.0

    mask = np.zeros((height, width), dtype=np.float32)
    cv2.ellipse(
        mask, (int(round(cx)), int(round(cy))),
        (int(round(half_short)), int(round(half_long))),
        angle=angle, startAngle=0, endAngle=360, color=1.0, thickness=-1,
    )
    sigma = max(0.5, face.size * blur_sigma_frac)
    mask = cv2.GaussianBlur(mask, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(mask, 0.0, 1.0)


_JAW_106 = list(range(0, 33))                              # temple to temple
_BROW_106 = list(range(43, 52)) + list(range(97, 106))     # both brows
_lm_app = None


def _landmark_app():
    """InsightFace with the dense-landmark head, kept alive between calls."""
    global _lm_app
    if _lm_app is None:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(allowed_modules=["detection", "landmark_2d_106"],
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _lm_app = app
    return _lm_app


def build_face_contour_mask_pixel(
    image: Image.Image,
    face: FaceInfo,
    height: int,
    width: int,
    *,
    forehead: float = 0.55,
    grow: float = 0.0,
    blur_sigma_frac: float = 0.08,
) -> np.ndarray | None:
    """The face's own outline: the 106-point jaw line closed over the brows.

    The dense contour stops at the temples, so the forehead is extrapolated
    upward from the brows by ``forehead`` of the brow-to-chin distance, which is
    roughly where the hairline sits. ``grow`` dilates the hull by that fraction
    of the inter-ocular distance before feathering.

    Returns ``None`` when the dense-landmark model finds nothing, which leaves
    the caller on the ellipse rather than on no mask at all.
    """
    bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
    faces = _landmark_app().get(bgr, max_num=0)
    faces = [f for f in faces if getattr(f, "landmark_2d_106", None) is not None]
    if not faces:
        return None
    # The primary face was already chosen upstream; this second detector has to
    # agree with that choice, so take the one whose eyes sit closest to it.
    center = np.asarray(face.center, dtype=np.float64)
    f = min(faces, key=lambda x: float(np.linalg.norm(x.kps[:2].mean(0) - center)))

    sx, sy = width / image.width, height / image.height
    lm = np.asarray(f.landmark_2d_106, dtype=np.float64) * np.array([sx, sy])
    jaw, brow = lm[_JAW_106], lm[_BROW_106]
    top = brow.copy()
    top[:, 1] -= forehead * (jaw[:, 1].max() - brow[:, 1].mean())

    hull = cv2.convexHull(np.vstack([jaw, top]).astype(np.float32))
    mask = np.zeros((height, width), dtype=np.float32)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 1.0)

    iod = float(np.linalg.norm(f.kps[1] - f.kps[0]) * (sx + sy) / 2.0)
    if grow > 0:
        k = max(1, int(round(grow * iod)) * 2 + 1)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(0.5, blur_sigma_frac * iod))
    return np.clip(mask, 0.0, 1.0)


def pixel_mask_to_token_mask(
    mask_pix: np.ndarray, token_h: int, token_w: int,
) -> torch.Tensor:
    """Downsample a pixel mask to the ``token_h x token_w`` image-token grid and
    flatten row-major to ``(token_h * token_w, 1)``.

    Row-major matches the standard ``b c h w -> b (h w) c`` patch flattening
    used by the backbones we target.
    """
    mask_grid = cv2.resize(mask_pix, (token_w, token_h), interpolation=cv2.INTER_AREA)
    mask_grid = np.clip(mask_grid, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(mask_grid).reshape(token_h * token_w, 1)


def build_face_token_mask(
    face: FaceInfo,
    *,
    height: int,
    width: int,
    token_h: int,
    token_w: int,
    expand_factor: float = 1.6,
    aspect: float = 1.3,
    blur_sigma_frac: float = 0.5,
    align_to_eyes: bool = True,
    shape: str = "ellipse",
    image: Image.Image | None = None,
    forehead: float = 0.55,
    grow: float = 0.0,
) -> tuple[np.ndarray, torch.Tensor]:
    """Convenience: pixel mask + token mask in one call.

    ``shape="contour"`` needs the frame the face was found on, since it re-reads
    the dense landmarks off it; it falls back to the ellipse if they are missing.

    Returns ``(mask_pixel[H, W], mask_tokens[token_h*token_w, 1])``.
    """
    mask_pix = None
    if shape == "contour" and image is not None:
        mask_pix = build_face_contour_mask_pixel(
            image, face, height=height, width=width,
            forehead=forehead, grow=grow, blur_sigma_frac=blur_sigma_frac,
        )
    if mask_pix is None:
        mask_pix = build_face_mask_pixel(
            face, height=height, width=width,
            expand_factor=expand_factor, aspect=aspect,
            blur_sigma_frac=blur_sigma_frac, align_to_eyes=align_to_eyes,
        )
    mask_tok = pixel_mask_to_token_mask(mask_pix, token_h, token_w)
    return mask_pix, mask_tok
