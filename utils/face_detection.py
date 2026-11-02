"""InsightFace-based face detection -> structured :class:`FaceInfo`.

Model-agnostic. The single-``det_size`` API is the fast path; the method also
needs :func:`detect_face_robust`, a multi-scale wrapper (RetinaFace anchors
cover only a bounded face-size range, so one ``det_size`` cannot see both a huge
close-up face and a tiny distant one). ArcFace embeddings for multi-person
matching live in :mod:`utils.arcface`.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis


@contextmanager
def _suppress_stdout_stderr():
    """Temporarily silence stdout/stderr (e.g. ONNX provider spam)."""
    devnull = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = devnull, devnull
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        devnull.close()


@dataclass
class FaceInfo:
    """Detected face metadata."""
    bbox: np.ndarray        # [x1, y1, x2, y2]
    landmarks: np.ndarray   # (5, 2) - left_eye, right_eye, nose, left_mouth, right_mouth
    score: float
    center: np.ndarray      # (2,) - face center (x, y)
    size: float             # characteristic face size (inter-ocular distance based)

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


_app: FaceAnalysis | None = None
_app_det_size: tuple[int, int] | None = None


def get_detector(det_size: tuple[int, int] = (640, 640)) -> FaceAnalysis:
    """Lazily initialise the InsightFace detector (singleton).

    Re-creates the detector if called with a different *det_size* than before.
    """
    global _app, _app_det_size
    if _app is None or _app_det_size != det_size:
        with _suppress_stdout_stderr():
            _app = FaceAnalysis(
                allowed_modules=["detection"],
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            _app.prepare(ctx_id=0, det_size=det_size)
        _app_det_size = det_size
    return _app


def _to_bgr_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    if image.ndim == 3 and image.shape[2] == 3:
        return image[:, :, ::-1].copy()  # RGB -> BGR
    return image


def _compute_face_info(bbox: np.ndarray, kps: np.ndarray, score: float) -> FaceInfo:
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]

    # Face center: midpoint between eyes, shifted slightly toward nose
    eye_center = (left_eye + right_eye) / 2.0
    center = 0.7 * eye_center + 0.3 * nose

    # Characteristic size: inter-ocular distance (robust to pose variation)
    iod = max(float(np.linalg.norm(right_eye - left_eye)), 1e-3)

    return FaceInfo(bbox=bbox, landmarks=kps, score=score, center=center, size=iod)


def detect_faces(
    image: Image.Image | np.ndarray,
    det_size: tuple[int, int] = (640, 640),
    min_score: float = 0.4,
) -> list[FaceInfo]:
    """Detect all faces in *image* and return structured metadata."""
    app = get_detector(det_size)
    bgr = _to_bgr_array(image)
    raw_faces = app.get(bgr, max_num=0)

    results = []
    for face in raw_faces:
        if face.det_score < min_score:
            continue
        results.append(_compute_face_info(face.bbox, face.kps, face.det_score))

    results.sort(key=lambda f: f.score, reverse=True)
    return results


def detect_primary_face(
    image: Image.Image | np.ndarray,
    det_size: tuple[int, int] = (640, 640),
    min_score: float = 0.4,
) -> FaceInfo | None:
    """Return the subject's face, or None if no face is found.

    The subject is the largest face, not the most confidently detected one: a
    sharp bystander behind the subject often scores higher than the person the
    photograph is of, and refining that instead leaves the subject untouched --
    silently, since a face was found and a crop was refined. Size is measured by
    inter-ocular distance rather than box area, because a blurred figure at the
    edge of the frame gets a tall box that beats the subject on area while its
    eyes stay a few pixels apart.
    """
    faces = detect_faces(image, det_size, min_score)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.size, f.score))


# Multi-scale detection. A SINGLE det_size cannot see both regimes we hit:
#   * a LARGE face (clean reference, upscaled refine crop filling the frame) is
#     MISSED at a big det_size (640+), detected at ~384;
#   * a SMALL / distant face (full-length scene, early-step x0) is MISSED at a
#     small det_size, detected only at a big one (1024-1600).
# We try several det_sizes small->large and take the first hit.
ROBUST_DET_SIZES = ((384, 384), (640, 640), (1024, 1024), (1600, 1600))
ROBUST_MIN_SCORE = 0.30


def detect_face_robust(
    image: Image.Image | np.ndarray,
    det_sizes: tuple[tuple[int, int], ...] = ROBUST_DET_SIZES,
    min_score: float = ROBUST_MIN_SCORE,
) -> FaceInfo | None:
    """Primary face via multi-scale detection (first det_size that hits)."""
    for ds in det_sizes:
        face = detect_primary_face(image, det_size=ds, min_score=min_score)
        if face is not None:
            return face
    return None
