"""Fit the transform that puts the reference object where the generated one is.

The face path gets this for free: five landmarks with known meaning, one
least-squares fit. An arbitrary object has no canonical points, so the
correspondences have to be found by appearance -- SIFT keypoints inside the two
silhouettes, matched by descriptor, filtered by MAGSAC. What survives the filter
("inliers") is both the fit and the confidence in it: a few hundred agreeing
points means the two images really do show the same view of the same thing, a
dozen means they do not and no warp will help.

Two rungs, deliberately:

``similarity``   4 DOF -- scale, in-plane rotation, translation. The rigid
                 object seen from the same viewpoint, just elsewhere in frame.
``homography``   8 DOF -- adds the perspective tilt of a plane. A label, a box
                 face or a front panel turned away from the camera. Straight
                 lines stay straight, which is why text survives it.

Anything past those (per-point warping, cylinder unwrapping) is left out on
purpose: in-context models mostly re-render rigid objects under exactly these
two transforms, and the extra freedom buys distortion more often than fit.

The rung is chosen by inlier support, with a margin: a homography can always
match a similarity's residual because it contains it, so it has to beat it by
enough to justify the extra freedom. When neither rung is supported, the caller
falls back to the centroid/area placement, which needs no correspondences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
from PIL import Image

AlignRung = Literal["homography", "similarity", "centroid"]

# Working resolution for the matcher: both objects are resampled so their
# longest side is this, which puts a small generated object and a large
# reference photo on the same footing (SIFT is scale invariant, but not
# resolution invariant -- a 120 px object simply has fewer keypoints).
WORK_SIDE = 640


# eq=False: the generated __eq__ would compare the matrices element-wise, and
# the selection below tests candidates by identity.
@dataclass(eq=False)
class AlignFit:
    """The chosen transform plus everything needed to defend the choice."""

    matrix: np.ndarray                  # 3x3, source image px -> target frame px
    rung: AlignRung
    n_matches: int = 0
    n_inliers: int = 0
    scale: float = 1.0
    mask_iou: float = 0.0               # warped source vs target silhouette
    spread: float = 0.0                 # inlier hull area / object area
    inlier_xy: np.ndarray = field(       # inliers in the OUTPUT frame
        default_factory=lambda: np.zeros((0, 2), np.float32))
    rejected: dict = field(default_factory=dict)   # rung -> why not

    @property
    def inlier_ratio(self) -> float:
        return self.n_inliers / max(self.n_matches, 1)

    def describe(self) -> str:
        bits = [
            f"{self.rung}",
            f"inliers={self.n_inliers}/{self.n_matches}",
            f"scale={self.scale:.3f}",
            f"iou={self.mask_iou:.2f}",
            f"spread={self.spread:.2f}",
        ]
        if self.rejected:
            bits.append("rejected: " + "; ".join(
                f"{k}: {v}" for k, v in self.rejected.items()))
        return " | ".join(bits)


def _work_crop(pil: Image.Image, mask: np.ndarray, bbox: np.ndarray,
               pad_frac: float = 0.08) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop to the object, resample to WORK_SIDE, return (gray, mask, T).

    ``T`` is the 3x3 taking original image pixels to this working frame, so the
    fit can be expressed back in image coordinates once it is done.
    """
    img = np.asarray(pil.convert("RGB"))
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    pad = pad_frac * max(x2 - x1, y2 - y1)
    x1 = int(max(0, np.floor(x1 - pad)))
    y1 = int(max(0, np.floor(y1 - pad)))
    x2 = int(min(w, np.ceil(x2 + pad)))
    y2 = int(min(h, np.ceil(y2 + pad)))
    crop = img[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    s = WORK_SIDE / max(crop.shape[0], crop.shape[1], 1)
    size = (max(1, int(round(crop.shape[1] * s))), max(1, int(round(crop.shape[0] * s))))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    gray = cv2.cvtColor(cv2.resize(crop, size, interpolation=interp), cv2.COLOR_RGB2GRAY)
    m = cv2.resize(crop_mask.astype(np.float32), size, interpolation=cv2.INTER_NEAREST)

    T = np.array([[s, 0.0, -s * x1], [0.0, s, -s * y1], [0.0, 0.0, 1.0]])
    return gray, (m > 0.5).astype(np.uint8), T


def _match(src_gray, src_mask, dst_gray, dst_mask, *, ratio: float = 0.75,
           n_features: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """SIFT + Lowe ratio test, keypoints restricted to the silhouettes.

    Restricting to the masks is what keeps the fit honest: the reference's own
    background and the generated scene share nothing, and matches between them
    are pure noise that MAGSAC then has to spend its budget rejecting.
    """
    sift = cv2.SIFT_create(nfeatures=n_features)
    kp_s, des_s = sift.detectAndCompute(src_gray, src_mask * 255)
    kp_d, des_d = sift.detectAndCompute(dst_gray, dst_mask * 255)
    if des_s is None or des_d is None or len(kp_s) < 4 or len(kp_d) < 4:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs = bf.knnMatch(des_s, des_d, k=2)
    src_pts, dst_pts = [], []
    for pair in pairs:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            src_pts.append(kp_s[m.queryIdx].pt)
            dst_pts.append(kp_d[m.trainIdx].pt)
    return (np.asarray(src_pts, np.float32).reshape(-1, 2),
            np.asarray(dst_pts, np.float32).reshape(-1, 2))


def _quad_is_sane(matrix: np.ndarray, gray_shape: tuple[int, int]) -> str | None:
    """Reject homographies that fold, mirror or stretch the object absurdly.

    A homography fit to points crowded in one corner stays numerically valid
    while mapping the object's frame to a bow tie. Checking what it does to the
    corners costs nothing and catches every such case.
    """
    h, w = gray_shape
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    if not np.isfinite(warped).all():
        return "non-finite corners"
    if not cv2.isContourConvex(warped.astype(np.float32)):
        return "quad not convex"
    area = abs(cv2.contourArea(warped.astype(np.float32)))
    if area < 1e-6:
        return "degenerate quad"
    ratio = area / float(w * h)
    if not (0.02 <= ratio <= 50.0):
        return f"area ratio {ratio:.3f}"
    sides = np.linalg.norm(np.roll(warped, -1, axis=0) - warped, axis=1)
    if sides.min() < 1e-3 or sides.max() / max(sides.min(), 1e-6) > 12.0:
        return "extreme aspect"
    return None


def _mask_iou(src_mask: np.ndarray, matrix: np.ndarray, dst_mask: np.ndarray,
              size: tuple[int, int]) -> float:
    """IoU of the warped source silhouette with the target's, in the output frame."""
    w, h = size
    warped = cv2.warpPerspective(src_mask.astype(np.float32), matrix, (w, h),
                                 flags=cv2.INTER_NEAREST)
    a = warped > 0.5
    b = cv2.resize(dst_mask.astype(np.float32), (w, h),
                   interpolation=cv2.INTER_NEAREST) > 0.5
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _hull_spread(points: np.ndarray, gray_shape: tuple[int, int]) -> float:
    """Fraction of the object's box covered by the convex hull of the inliers.

    Inliers packed into the logo alone can pin a similarity down but leave a
    homography's perspective terms guessing, so spread decides which rungs are
    allowed, independently of how many inliers there are.
    """
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32).reshape(-1, 1, 2))
    h, w = gray_shape
    return float(abs(cv2.contourArea(hull)) / max(w * h, 1))


def confidence_map(inlier_xy: np.ndarray, output_size: tuple[int, int], *,
                   sigma: float, radius: float = 0.0) -> np.ndarray:
    """How far each pixel is from a correspondence that actually agreed, as a
    weight in [0, 1]: full weight within ``radius`` of an inlier, then
    ``exp(-d^2 / 2 sigma^2)`` on the distance past it.

    The transform is fitted to whatever surface carries the printing, because
    that is where the keypoints are, and it is only true on that surface. A
    smooth cap contributes no keypoints, gets warped by the label's transform
    anyway, and lands a few pixels off -- which is visible as a doubled rim.
    Weighting the blend by this map hands those places back to the generator,
    which had no trouble with them, and keeps the copy stream where it is
    demonstrably right.

    The plateau is what makes that separable. Keypoints come in clusters, so a
    bare Gaussian has to choose: wide enough to cover the gaps between clusters
    inside one label, or narrow enough to stop before the cap. ``radius``
    covers the gaps and ``sigma`` stops the reach.
    """
    w, h = output_size
    if len(inlier_xy) == 0:
        return np.ones((h, w), np.float32)
    seeds = np.ones((h, w), np.uint8)
    xs = np.clip(np.round(inlier_xy[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(inlier_xy[:, 1]).astype(int), 0, h - 1)
    seeds[ys, xs] = 0
    dist = cv2.distanceTransform(seeds, cv2.DIST_L2, 3)
    past = np.maximum(dist - max(float(radius), 0.0), 0.0)
    return np.exp(-0.5 * (past / max(sigma, 1e-3)) ** 2).astype(np.float32)


def _principal_axis(mask: np.ndarray) -> tuple[float, float]:
    """Angle of the silhouette's long axis (radians, image frame) and its elongation."""
    ys, xs = np.nonzero(mask > 0.5)
    if len(xs) < 10:
        return 0.0, 1.0
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    mu20, mu02, mu11 = float((xs * xs).mean()), float((ys * ys).mean()), float((xs * ys).mean())
    angle = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
    spread = math.hypot(mu20 - mu02, 2.0 * mu11)
    l1, l2 = 0.5 * (mu20 + mu02 + spread), 0.5 * (mu20 + mu02 - spread)
    return angle, math.sqrt(l1 / max(l2, 1e-6))


def centroid_matrix(source_obj, target_obj, size_ratio: float = 1.0,
                    min_elongation: float = 1.5) -> np.ndarray:
    """The correspondence-free placement: match areas, centroids and long axes.

    The axis says nothing about which end is which, so of the two rotations
    that line the axes up the smaller one is taken: an upright product laid at
    a diagonal keeps its cap on the upper end. Round silhouettes have no axis
    to speak of and are not rotated.
    """
    scale = float(size_ratio) * float(target_obj.size) / float(source_obj.size)
    a_src, e_src = _principal_axis(source_obj.mask)
    a_dst, e_dst = _principal_axis(target_obj.mask)
    rot = 0.0
    if min(e_src, e_dst) >= min_elongation:
        rot = (a_dst - a_src + math.pi / 2) % math.pi - math.pi / 2
    c, s = scale * math.cos(rot), scale * math.sin(rot)
    cx, cy = float(source_obj.center[0]), float(source_obj.center[1])
    tx = float(target_obj.center[0]) - (c * cx - s * cy)
    ty = float(target_obj.center[1]) - (s * cx + c * cy)
    return np.array([[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]])


def fit_object_transform(
    source_pil: Image.Image,
    target_pil: Image.Image,
    source_obj,
    target_obj,
    output_size: tuple[int, int],
    *,
    size_ratio: float = 1.0,
    allow_homography: bool = True,
    min_inliers: int = 15,
    homography_margin: float = 1.15,
    min_homography_spread: float = 0.06,
    min_mask_iou: float = 0.25,
    scale_clamp: tuple[float, float] = (0.05, 20.0),
    verbose: bool = False,
) -> AlignFit:
    """Choose between the two rungs, or report that neither is supported.

    ``source_obj`` / ``target_obj`` are ``ObjectInfo``; ``target_pil`` must
    already be in the output frame, which is what the pipeline hands us.
    """
    src_gray, src_m, T_src = _work_crop(source_pil, source_obj.mask, source_obj.bbox)
    dst_gray, dst_m, T_dst = _work_crop(target_pil, target_obj.mask, target_obj.bbox)
    src_pts, dst_pts = _match(src_gray, src_m, dst_gray, dst_m)

    rejected: dict[str, str] = {}
    fallback = AlignFit(
        matrix=centroid_matrix(source_obj, target_obj, size_ratio),
        rung="centroid", n_matches=len(src_pts), rejected=rejected,
    )
    fallback.scale = float(np.hypot(fallback.matrix[0, 0], fallback.matrix[1, 0]))
    if len(src_pts) < 4:
        rejected["matcher"] = f"only {len(src_pts)} matches"
        fallback.mask_iou = _mask_iou(source_obj.mask, fallback.matrix,
                                      target_obj.mask, output_size)
        return fallback

    T_dst_inv = np.linalg.inv(T_dst)
    candidates: list[AlignFit] = []

    def to_output_frame(points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.zeros((0, 2), np.float32)
        return cv2.perspectiveTransform(
            points.astype(np.float32).reshape(-1, 1, 2), T_dst_inv).reshape(-1, 2)

    # Plain RANSAC here: estimateAffinePartial2D takes only RANSAC / LMEDS,
    # MAGSAC is a findHomography-only method in OpenCV.
    sim, sim_in = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=20000, confidence=0.999,
    )
    if sim is None:
        rejected["similarity"] = "no fit"
    else:
        work = np.vstack([sim, [0.0, 0.0, 1.0]])
        scale = float(np.hypot(work[0, 0], work[0, 1]))
        matrix = T_dst_inv @ work @ T_src
        full_scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
        if not (scale_clamp[0] <= full_scale <= scale_clamp[1]):
            rejected["similarity"] = f"scale {full_scale:.3f} out of clamp"
        else:
            inl = sim_in.ravel().astype(bool) if sim_in is not None else np.zeros(len(src_pts), bool)
            candidates.append(AlignFit(
                matrix=matrix, rung="similarity", n_matches=len(src_pts),
                n_inliers=int(inl.sum()), scale=full_scale,
                spread=_hull_spread(dst_pts[inl], dst_gray.shape),
                inlier_xy=to_output_frame(dst_pts[inl]),
            ))

    if allow_homography:
        hom, hom_in = cv2.findHomography(
            src_pts, dst_pts, method=cv2.USAC_MAGSAC, ransacReprojThreshold=3.0,
            maxIters=20000, confidence=0.999,
        )
        if hom is None:
            rejected["homography"] = "no fit"
        else:
            bad = _quad_is_sane(hom, src_gray.shape)
            inl = hom_in.ravel().astype(bool) if hom_in is not None else np.zeros(len(src_pts), bool)
            spread = _hull_spread(dst_pts[inl], dst_gray.shape)
            matrix = T_dst_inv @ hom @ T_src
            if bad:
                rejected["homography"] = bad
            elif spread < min_homography_spread:
                rejected["homography"] = f"inliers cover only {spread:.3f} of the object"
            else:
                candidates.append(AlignFit(
                    matrix=matrix, rung="homography", n_matches=len(src_pts),
                    n_inliers=int(inl.sum()),
                    scale=float(np.hypot(matrix[0, 0], matrix[0, 1])),
                    spread=spread,
                    inlier_xy=to_output_frame(dst_pts[inl]),
                ))

    for cand in candidates:
        cand.mask_iou = _mask_iou(source_obj.mask, cand.matrix, target_obj.mask,
                                  output_size)

    ok = [c for c in candidates if c.n_inliers >= min_inliers and c.mask_iou >= min_mask_iou]
    for c in candidates:
        if c not in ok:
            rejected.setdefault(c.rung, f"inliers={c.n_inliers} iou={c.mask_iou:.2f}")

    if not ok:
        fallback.mask_iou = _mask_iou(source_obj.mask, fallback.matrix,
                                      target_obj.mask, output_size)
        if verbose:
            print(f"    [align] {fallback.describe()}", flush=True)
        return fallback

    sim_fit = next((c for c in ok if c.rung == "similarity"), None)
    hom_fit = next((c for c in ok if c.rung == "homography"), None)
    if hom_fit is not None and (
        sim_fit is None or hom_fit.n_inliers >= homography_margin * sim_fit.n_inliers
    ):
        chosen = hom_fit
        if sim_fit is not None:
            rejected["similarity"] = (
                f"homography wins {hom_fit.n_inliers} vs {sim_fit.n_inliers} inliers")
    else:
        chosen = sim_fit
        if hom_fit is not None:
            rejected["homography"] = (
                f"extra freedom unearned: {hom_fit.n_inliers} vs "
                f"{sim_fit.n_inliers} inliers")
    chosen.rejected = rejected
    if verbose:
        print(f"    [align] {chosen.describe()}", flush=True)
    return chosen
