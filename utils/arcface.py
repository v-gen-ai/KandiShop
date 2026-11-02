"""ArcFace embeddings + face<->reference assignment for multi-person refinement.

With several references the base image must have each generated face refined
with the CORRECT identity. We embed every detected face and every reference with
InsightFace ``buffalo_l`` (ArcFace) and match by cosine similarity (Hungarian if
SciPy is available, else greedy). Backbone-independent.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .face_detection import FaceInfo, _compute_face_info, _suppress_stdout_stderr

# Multi-scale for the same reason as detect_face_robust (RetinaFace anchors
# cover a bounded face-size range).
ARC_DET_SIZES = ((384, 384), (640, 640), (1024, 1024), (1600, 1600))
ARC_MIN_SCORE = 0.30

_arc_app = None


def get_arcface_app():
    """Lazily build the recognition-capable InsightFace app (singleton)."""
    global _arc_app
    if _arc_app is None:
        from insightface.app import FaceAnalysis
        with _suppress_stdout_stderr():
            _arc_app = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection", "recognition"],
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            _arc_app.prepare(ctx_id=0, det_size=ARC_DET_SIZES[0])
    return _arc_app


def _bbox_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return float(inter / (area_a + area_b - inter + 1e-9))


def detect_faces_with_embeddings(
    pil: Image.Image, det_sizes=ARC_DET_SIZES, min_score: float = ARC_MIN_SCORE,
) -> list[tuple[FaceInfo, np.ndarray]]:
    """Detect ALL faces across multiple det_sizes; return ``(FaceInfo, arcface
    embedding)`` per unique face (deduped by bbox IoU, highest score kept)."""
    app = get_arcface_app()
    bgr = np.asarray(pil.convert("RGB"))[:, :, ::-1]
    collected = []
    for ds in det_sizes:
        app.prepare(ctx_id=0, det_size=ds)
        for f in app.get(bgr):
            if float(f.det_score) >= min_score:
                collected.append(f)
    collected.sort(key=lambda f: float(f.det_score), reverse=True)
    kept: list = []
    for f in collected:
        if all(_bbox_iou(f.bbox, k.bbox) < 0.5 for k in kept):
            kept.append(f)
    items = []
    for f in kept:
        info = _compute_face_info(
            np.asarray(f.bbox, dtype=np.float64),
            np.asarray(f.kps, dtype=np.float64),
            float(f.det_score),
        )
        emb = np.asarray(f.normed_embedding, dtype=np.float32)
        items.append((info, emb))
    return items


def primary_embedding(pil: Image.Image):
    """L2-normalised ArcFace embedding of the most confident face, or None."""
    items = detect_faces_with_embeddings(pil)
    if not items:
        return None
    items.sort(key=lambda t: t[0].score, reverse=True)
    return items[0][1]


def assign_faces_to_refs(face_embs: np.ndarray, ref_embs: np.ndarray, min_sim: float):
    """Assign generated faces to references by max cosine similarity.

    ``face_embs`` (M, D) and ``ref_embs`` (K, D) are L2-normalised. Returns
    ``(pairs, sim_matrix)`` where ``pairs`` is a list of
    ``(face_idx, ref_local_idx, similarity)``.

    ``min_sim`` guards against a *bystander* face stealing a reference, which can
    only happen with MORE faces than references (M > K). When M <= K, every
    detected face is expected to be a target person, so a low similarity just
    means poor base likeness -- exactly what we most want to refine -- and we
    keep the 1:1 assignment regardless of ``min_sim``.
    """
    sim = face_embs @ ref_embs.T  # (M, K)
    M, K = sim.shape
    effective_min = min_sim if M > K else -1.0
    pairs = []
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-sim)
        for r, c in zip(rows, cols):
            if sim[r, c] >= effective_min:
                pairs.append((int(r), int(c), float(sim[r, c])))
    except Exception:
        used_f, used_r = set(), set()
        flat = sorted(
            ((sim[r, c], r, c) for r in range(M) for c in range(K)),
            reverse=True,
        )
        for s, r, c in flat:
            if r in used_f or c in used_r or s < effective_min:
                continue
            used_f.add(r)
            used_r.add(c)
            pairs.append((int(r), int(c), float(s)))
    return pairs, sim
