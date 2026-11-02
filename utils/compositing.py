"""Crop-box geometry and feathered alpha compositing for face refinement.

All backbone-independent. The refinement pass crops a face region, upscales it,
regenerates it, then pastes it back through a feathered elliptical alpha derived
from the refinement's own face mask.
"""

from __future__ import annotations

from math import ceil, floor

import cv2
import numpy as np
from PIL import Image


def square_crop_box(
    cx: float, cy: float, side: float, W: int, H: int,
) -> tuple[int, int, int, int]:
    """A square box of ``side`` px centred at ``(cx, cy)``, shrunk/shifted to fit
    inside a ``W x H`` image (kept square so the upscaled crop maps back without
    warping). Returns ``(x0, y0, x1, y1)``."""
    side = int(round(min(side, W, H)))
    side = max(8, side)
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x0 = max(0, min(x0, W - side))
    y0 = max(0, min(y0, H - side))
    return x0, y0, x0 + side, y0 + side


def clip_box_to_neighbors(
    box: tuple[float, float, float, float],
    cx: float, cy: float,
    neighbor_centers: list[np.ndarray],
) -> tuple[float, float, float, float]:
    """Clip ``box`` so it never crosses the midline toward any neighbouring face.

    For each neighbour, the box is clipped on the dominant separation axis
    (whichever of |dx|, |dy| is larger) at the midpoint between the two centers.
    This keeps every pair of crops strictly disjoint, so their masks (a subset of
    each crop) are disjoint too -> one person's refinement can't drift another's
    identity. Returns the (possibly rectangular) clipped box.
    """
    bx0, by0, bx1, by1 = box
    for c in neighbor_centers:
        jx, jy = float(c[0]), float(c[1])
        dx, dy = jx - cx, jy - cy
        if abs(dx) >= abs(dy):
            mid = 0.5 * (cx + jx)
            if dx > 0:
                bx1 = min(bx1, mid)
            else:
                bx0 = max(bx0, mid)
        else:
            mid = 0.5 * (cy + jy)
            if dy > 0:
                by1 = min(by1, mid)
            else:
                by0 = max(by0, mid)
    return bx0, by0, bx1, by1


def face_crop_boxes(
    bboxes: list,
    size: tuple[int, int],
    crop_scale: float = 2.4,
    gap_frac: float = 0.08,
    max_side: float | None = None,
) -> list:
    """One square crop box per face: as large as the scene allows, never
    reaching another face.

    Faces are refined one at a time, each pasted before the next is cropped, so
    two crops overlapping is harmless -- the later one simply sees the earlier
    one's result. What must not happen is a crop *containing* a second face,
    because then the refinement has two faces to choose from and may work on
    the wrong one.

    That is a far weaker constraint than keeping the crops disjoint, and it is
    the one worth enforcing. A crop may run most of the way to its neighbour
    instead of stopping at the midline between them, which is what leaves room
    for the paste to fade out: the alpha has to reach zero by the crop border,
    so a border pressed against the face is a visible seam there.

    Each side grows to ``crop_scale`` times the face, or until it comes within
    ``gap_frac`` of the face size of a neighbouring face box, or until the frame
    ends. The result is squared off, because a non-square canvas makes
    HunyuanImage-3 draw a smaller square picture and pad it white.

    ``max_side`` caps how large a box may get, so the refinement always works
    on a crop it has to upscale; the face still overrides it, since a box that
    cannot hold the face is useless.
    """
    W, H = size
    n = len(bboxes)
    if n == 0:
        return []
    b = [tuple(float(v) for v in bb[:4]) for bb in bboxes]

    out = []
    for i, (fx0, fy0, fx1, fy1) in enumerate(b):
        side = max(fx1 - fx0, fy1 - fy0)
        half = 0.5 * crop_scale * side
        gap = gap_frac * side
        cx, cy = 0.5 * (fx0 + fx1), 0.5 * (fy0 + fy1)

        lo_x, hi_x = cx - half, cx + half
        lo_y, hi_y = cy - half, cy + half
        for j, (ox0, oy0, ox1, oy1) in enumerate(b):
            if j == i:
                continue
            ocx, ocy = 0.5 * (ox0 + ox1), 0.5 * (oy0 + oy1)
            # A neighbour blocks along whichever axis separates the two faces;
            # on the other axis they overlap and blocking would be meaningless.
            if abs(ocx - cx) >= abs(ocy - cy):
                if ocx >= cx:
                    hi_x = min(hi_x, ox0 - gap)
                else:
                    lo_x = max(lo_x, ox1 + gap)
            else:
                if ocy >= cy:
                    hi_y = min(hi_y, oy0 - gap)
                else:
                    lo_y = max(lo_y, oy1 + gap)
        lo_x = max(lo_x, 0.0); hi_x = min(hi_x, float(W))
        lo_y = max(lo_y, 0.0); hi_y = min(hi_y, float(H))

        # Square it off, then slide the square to sit as centred on the face as
        # the room allows. Both ends matter: the square is as wide as the
        # narrower of the two spans, so along the wider one it can still be
        # pushed past the limit unless that is checked too.
        s = int(min(hi_x - lo_x, hi_y - lo_y))
        if max_side is not None:
            s = min(s, int(round(max_side)))
        s = max(s, int(ceil(max(fx1 - fx0, fy1 - fy0))))
        s = min(s, W, H)

        def slide(c, f0, f1, lo, hi, extent):
            # Cover the face first, sit inside the allowed span second, and be
            # centred on the face only with whatever freedom is left.
            a = max(int(lo), int(ceil(f1)) - s, 0)
            z = min(int(hi) - s, int(floor(f0)), extent - s)
            if z < a:
                z = a
            return max(0, min(min(max(int(c - 0.5 * s), a), z), extent - s))

        x = slide(cx, fx0, fx1, lo_x, hi_x, W)
        y = slide(cy, fy0, fy1, lo_y, hi_y, H)
        out.append((x, y, x + s, y + s))
    return out


def _interior_border_distance(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
    base_size: tuple[int, int],
) -> np.ndarray:
    """Distance in crop pixels to the nearest crop side that is *interior* to
    the frame. A side flush with the image border gets a huge distance: the crop
    ends there because the picture does, so nothing has to be blended away."""
    x0, y0, x1, y1 = box
    base_w, base_h = base_size
    h, w = shape
    inf = 1.0e6
    cols = np.arange(w, dtype=np.float32)
    rows = np.arange(h, dtype=np.float32)
    d_left = cols if x0 > 0 else np.full(w, inf, dtype=np.float32)
    d_right = (w - 1 - cols) if x1 < base_w else np.full(w, inf, dtype=np.float32)
    d_top = rows if y0 > 0 else np.full(h, inf, dtype=np.float32)
    d_bot = (h - 1 - rows) if y1 < base_h else np.full(h, inf, dtype=np.float32)
    return np.minimum(np.minimum(d_left[None, :], d_right[None, :]),
                      np.minimum(d_top[:, None], d_bot[:, None]))


def flat_border_alpha(
    before: Image.Image,
    after: Image.Image,
    *,
    diff_thresh: float = 28.0,
    flat_thresh: float = 5.0,
    max_frac: float = 0.25,
    feather_frac: float = 0.01,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Zero the alpha over padding bands the generator drew along the crop edge.

    Some canvases come back letterboxed: the model paints the picture in part of
    the frame and fills the rest with one flat colour, and pasting that band back
    lays a grey or white bar across the scene. The band is trivial to recognise
    without a model of why it happened -- it runs the full width or height of the
    crop, it is very nearly constant, and it bears no relation to what was there
    before. Both conditions are needed: a row that is flat in the refined crop
    *and* flat in the original is a wall or a sky and must be kept, and a row
    that changed a great deal but is not flat is just a refined face.

    Returns the alpha to multiply the composite by -- 0 over each band, ramping
    back to 1 over ``feather_frac`` of the crop so the cut is not a second seam
    -- and the four band widths in crop pixels, for logging.
    """
    if before.size != after.size:
        before = before.resize(after.size, Image.LANCZOS)
    a = np.asarray(after.convert("RGB"), dtype=np.float32)
    b = np.asarray(before.convert("RGB"), dtype=np.float32)
    gray = a.mean(axis=2)
    diff = np.abs(a - b).mean(axis=2)
    h, w = gray.shape

    def run_length(flat: np.ndarray, dif: np.ndarray, n: int) -> int:
        """How many lines from index 0 on are flat-and-changed, capped."""
        bad = (flat < flat_thresh) & (dif > diff_thresh)
        k = 0
        limit = int(max_frac * n)
        while k < limit and bad[k]:
            k += 1
        return k

    row_flat, row_diff = gray.std(axis=1), diff.mean(axis=1)
    top = run_length(row_flat, row_diff, h)
    bottom = run_length(row_flat[::-1], row_diff[::-1], h)
    # Columns are measured on what the row scan left, or a band along the top
    # would sit in every column and hide a band along the left.
    inner = slice(top, h - bottom if bottom else h)
    col_flat = gray[inner].std(axis=0)
    col_diff = diff[inner].mean(axis=0)
    left = run_length(col_flat, col_diff, w)
    right = run_length(col_flat[::-1], col_diff[::-1], w)

    feather = max(1.0, feather_frac * max(h, w))

    def profile(n: int, lo: int, hi: int) -> np.ndarray:
        p = np.ones(n, dtype=np.float32)
        idx = np.arange(n, dtype=np.float32)
        if lo:
            p = np.minimum(p, np.clip((idx - lo + 1) / feather, 0.0, 1.0))
        if hi:
            p = np.minimum(p, np.clip((n - hi - idx) / feather, 0.0, 1.0))
        return p

    alpha = np.minimum(profile(h, top, bottom)[:, None],
                       profile(w, left, right)[None, :])
    return alpha, (top, bottom, left, right)


def composite_alpha(
    face_mask_pixel: np.ndarray,
    box: tuple[int, int, int, int],
    base_size: tuple[int, int],
    *,
    refine_size: int,
    composite_strength: float = 1.0,
    composite_feather_frac: float = 0.05,
    composite_mode: str = "canon",
    composite_dilate_frac: float = 0.40,
    composite_soft_frac: float = 0.25,
    composite_falloff_bias: float = 1.0,
    composite_edge_frac: float = 0.05,
    composite_fit: bool = True,
) -> np.ndarray:
    """The alpha the refined crop is pasted through, in crop pixels.

    ``composite_mode="canon"`` blurs the soft blend mask by
    ``composite_feather_frac * refine_size`` and uses that directly. Blurring a
    mask that is already soft costs peak alpha, so the middle of the face comes
    back only partly: this is the historical behaviour, kept for comparison.

    ``composite_mode="silhouette"`` is the head-swap paste. The mask's core
    (``> 0.5``) is opaque and stays opaque; it is dilated by
    ``composite_dilate_frac`` of its own bounding box -- anisotropically, so a
    tall mask grows more vertically -- and outside that the alpha ramps to zero
    over ``composite_soft_frac`` of the core's size, measured as a distance, so
    the width of the fade is set directly rather than inferred from a blur
    radius.

    ``composite_mode="falloff"`` keeps the same opaque core but spends *all* the
    room that is left on the fade: at every pixel the alpha is read off how far
    it lies between the core and the crop border, so the ramp is as wide as the
    crop allows and reaches zero exactly where the crop ends. That is the widest
    possible transition for a given core, and it is what a thin band cannot do:
    a short ramp reads as an edge no matter how smooth it is in itself.
    ``composite_falloff_bias`` above 1 holds the alpha high for longer before it
    drops, below 1 starts the drop right at the core.

    ``composite_fit`` shrinks the ring, keeping the ratio between the opaque
    part and the fade, until it dies out inside the crop on every side interior
    to the frame. Without it a wide ring simply runs off the crop and the
    transition happens along the crop rectangle -- a straight seam through the
    background, which is far more visible than the oval one it was meant to
    hide. Sides flush against the image border do not constrain it: there the
    crop ends where the frame ends and there is nothing to blend into.

    ``composite_edge_frac`` is the remaining safety net, ramping the alpha down
    over the last fraction of the crop on interior sides.
    """
    if composite_mode not in ("canon", "silhouette", "falloff"):
        raise ValueError(f"unknown composite_mode: {composite_mode!r}")
    mask = np.clip(face_mask_pixel.astype(np.float32), 0.0, 1.0)

    if composite_mode == "canon":
        feather = max(1.0, composite_feather_frac * refine_size)
        alpha = cv2.GaussianBlur(mask, ksize=(0, 0), sigmaX=feather, sigmaY=feather)
        return np.clip(alpha * float(composite_strength), 0.0, 1.0)

    core = (mask > 0.5).astype(np.float32)
    alpha = core
    ys, xs = np.where(core > 0.5)
    if xs.size > 0:
        core_w = float(xs.max() - xs.min() + 1)
        core_h = float(ys.max() - ys.min() + 1)
        rx = composite_dilate_frac * core_w
        ry = composite_dilate_frac * core_h
        soft = composite_soft_frac * max(core_w, core_h)
        if composite_mode == "falloff":
            soft = 0.0
        if composite_fit:
            h, w = mask.shape
            x0, y0, x1, y1 = box
            base_w, base_h = base_size
            inf = 1.0e6
            room_x = min(float(xs.min()) if x0 > 0 else inf,
                         float(w - 1 - xs.max()) if x1 < base_w else inf)
            room_y = min(float(ys.min()) if y0 > 0 else inf,
                         float(h - 1 - ys.max()) if y1 < base_h else inf)
            margin = 0.02 * refine_size
            k = min(1.0,
                    max(0.0, room_x - margin) / max(1.0, rx + soft),
                    max(0.0, room_y - margin) / max(1.0, ry + soft))
            rx, ry, soft = rx * k, ry * k, soft * k
        solid = core
        if rx >= 1 and ry >= 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * int(round(rx)) + 1, 2 * int(round(ry)) + 1))
            solid = cv2.dilate(core, kernel)
        alpha = solid
        outside = (solid < 0.5).astype(np.uint8)
        if composite_mode == "falloff":
            d_core = cv2.distanceTransform(outside, cv2.DIST_L2, 3)
            d_edge = _interior_border_distance(mask.shape, box, base_size)
            u = d_core / np.maximum(d_core + d_edge, 1e-6)
            u = np.clip(u, 0.0, 1.0) ** (1.0 / max(1e-3, composite_falloff_bias))
            alpha = np.maximum(solid, 1.0 - u * u * (3.0 - 2.0 * u))
        elif soft >= 1:
            dist = cv2.distanceTransform(outside, cv2.DIST_L2, 3)
            t = np.clip(1.0 - dist / soft, 0.0, 1.0)
            alpha = np.maximum(solid, t * t * (3.0 - 2.0 * t))

    edge_r = int(round(refine_size * composite_edge_frac))
    if edge_r > 0:
        dist = _interior_border_distance(alpha.shape, box, base_size)
        w_box = np.clip(dist / float(edge_r), 0.0, 1.0)
        alpha = alpha * (w_box * w_box * (3.0 - 2.0 * w_box))
    return np.clip(alpha * float(composite_strength), 0.0, 1.0)


def composite_refined_face(
    base_pil: Image.Image,
    refined: Image.Image,
    face_mask_pixel: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    refine_size: int,
    composite_strength: float = 1.0,
    composite_feather_frac: float = 0.05,
    composite_mode: str = "canon",
    composite_dilate_frac: float = 0.40,
    composite_soft_frac: float = 0.25,
    composite_falloff_bias: float = 1.0,
    composite_edge_frac: float = 0.05,
    composite_fit: bool = True,
) -> Image.Image:
    """Paste ``refined`` back into ``base_pil`` at ``box`` through
    :func:`composite_alpha`. Returns a new image."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    alpha = composite_alpha(
        face_mask_pixel, box, base_pil.size, refine_size=refine_size,
        composite_strength=composite_strength,
        composite_feather_frac=composite_feather_frac,
        composite_mode=composite_mode,
        composite_dilate_frac=composite_dilate_frac,
        composite_soft_frac=composite_soft_frac,
        composite_falloff_bias=composite_falloff_bias,
        composite_edge_frac=composite_edge_frac,
        composite_fit=composite_fit,
    )
    alpha_u8 = (alpha * 255.0).round().astype(np.uint8)
    refined_box = refined.resize((bw, bh), Image.LANCZOS)
    alpha_box = Image.fromarray(alpha_u8).resize((bw, bh), Image.BILINEAR).convert("L")
    out = base_pil.copy()
    out.paste(refined_box, (x0, y0), alpha_box)
    return out
