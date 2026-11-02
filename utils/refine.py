"""Backbone-agnostic face-refinement orchestration (single + multi-person).

The refinement itself (crop -> upscale -> img2img dual-stream -> composite) is
identical across backbones. The only backbone-specific piece is *how* to run an
img2img dual-stream generation, which is injected as a ``generate`` callable:

    def generate(*, source_image, image_size, init_image,
                 target_face_image, denoise_strength) -> GenResult

where ``GenResult`` is any object exposing:
    - ``output_image``     : PIL.Image | None (the regenerated crop, rank 0)
    - ``alignment_applied``: bool
    - ``face_mask_pixel``  : np.ndarray | None (soft mask over the crop)
    - ``i1_aligned``       : PIL.Image | None (aligned reference, for debug)

Per-model launchers build ``generate`` as a closure that captures all the
model-specific knobs (prompts, steps, seed, guidance, mask/align config, blend
layers), so this module stays free of any backbone import.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from .arcface import (
    assign_faces_to_refs,
    detect_faces_with_embeddings,
    primary_embedding,
)
from .compositing import (
    clip_box_to_neighbors,
    composite_refined_face,
    face_crop_boxes,
    square_crop_box,
)
from .distributed import broadcast_object, is_main
from .face_detection import detect_face_robust

GenerateFn = Callable[..., Any]
SnapSizeFn = Callable[[int, int], tuple[int, int]]


@dataclass
class RefinedFace:
    """One regenerated crop and everything needed to paste it again later.

    The paste is a pure post-process, so a different alpha can always be tried
    on a finished run -- but only if these three outlive it. The crop cost a
    diffusion pass and cannot be recovered from the composited frame, since
    wherever the alpha was zero the refined pixels never reached it. The mask
    and the box are cheap, and are kept beside the crop anyway so that a
    re-paste does not have to re-detect anything and hope its detector agrees
    with the one the run used.
    """
    crop: Image.Image | None = None            # regenerated crop (rank 0)
    mask_pixel: Any = None                     # np.ndarray blend mask over the crop
    box: tuple[int, int, int, int] | None = None   # where it came from, in base px
    ref_index: int = 0                         # which reference it was refined against
    applied: bool = False                      # did it actually get composited


@dataclass
class RefineResult:
    output_image: Image.Image | None = None   # composited full image (rank 0)
    refined_crop: Image.Image | None = None   # most prominent regenerated crop (rank 0)
    crop_box: tuple[int, int, int, int] | None = None
    applied: bool = False
    inner: Any = None
    # One entry per refined face, in the order they were composited. Empty off
    # rank 0, where the crops are never decoded.
    faces: list[RefinedFace] = field(default_factory=list)


def cap_crop_side(
    side: float, face_side: float, refine_size: int, min_upscale: float,
) -> float:
    """Shrink ``side`` so the crop is genuinely upscaled to ``refine_size``.

    The pass renders at ``refine_size`` and adds detail at that scale whatever
    it was given. On a crop that had to be upscaled to get there the added
    detail reads as resolution the original never had, and the composite's
    resize back to the box averages away what is left. On a close-up face,
    where ``face_side * crop_scale`` exceeds the frame and the box degenerates
    to the whole image, the crop arrives at its native size, the same detail
    lands on top of real detail, and it is pasted back one to one -- which
    comes out as a crunchy over-sharpened texture across the whole frame. That
    is a property of the generator, not of the blend: it is just as bad with
    the dual stream switched off entirely.

    The face sets the floor, since a cap tight enough to crop it off would be
    worse than the crunch; when the two conflict the face wins and the pass
    runs at whatever upscale is left.
    """
    if min_upscale <= 1.0:
        return side
    cap = float(refine_size) / float(min_upscale)
    return max(min(side, cap), face_side * 1.1)


def _default_snap(bw: int, bh: int) -> tuple[int, int]:
    """Round a crop's dimensions down to a multiple of 32 (safe for the packed
    latents of FLUX.2 / Qwen: 8x VAE downsample x 2x2 patch)."""
    gw = max(64, (bw // 32) * 32)
    gh = max(64, (bh // 32) * 32)
    return gw, gh


# ---------------------------------------------------------------------------
# Single-face refinement
# ---------------------------------------------------------------------------

def refine_face(
    *,
    base_image: Image.Image | None,   # PIL (generated) on rank 0; None elsewhere
    source_image,                     # identity reference (path or PIL), all ranks
    generate: GenerateFn,
    refine_size: int = 1024,
    crop_scale: float = 2.4,
    denoise: float = 0.5,
    composite_strength: float = 1.0,
    composite_feather_frac: float = 0.05,
    composite_mode: str = "falloff",
    composite_dilate_frac: float = 0.25,
    skip_if_face_frac_gt: Optional[float] = None,
    min_upscale: float = 2.0,
    debug_dir: Optional[str] = None,
    verbose: bool = True,
) -> RefineResult:
    """Detect the face in ``base_image``, crop+upscale it, regenerate it with
    ``generate`` (img2img dual-stream), and composite the refined face back."""
    do_log = verbose and is_main()
    res = RefineResult()

    payload = None
    if is_main():
        if base_image is None:
            payload = {"ok": False, "reason": "no base image"}
        else:
            base_pil = base_image.convert("RGB")
            W, H = base_pil.size
            face = detect_face_robust(base_pil)
            if face is None:
                payload = {"ok": False, "reason": "no face in base image"}
            else:
                fw, fh = face.width, face.height
                face_frac = max(fw, fh) / float(max(W, H))
                if skip_if_face_frac_gt is not None and face_frac > skip_if_face_frac_gt:
                    payload = {"ok": False, "reason": f"face large enough ({face_frac:.2f})"}
                else:
                    want = max(fw, fh) * crop_scale
                    side = cap_crop_side(want, max(fw, fh), refine_size, min_upscale)
                    box = square_crop_box(float(face.center[0]), float(face.center[1]), side, W, H)
                    crop_up = base_pil.crop(box).resize((refine_size, refine_size), Image.LANCZOS)
                    payload = {"ok": True, "box": box, "crop_up": crop_up, "face_frac": face_frac}
                    if do_log:
                        capped = (f" (side {want:.0f} capped to {side:.0f})"
                                  if side < want else "")
                        print(f"[refine] face_frac={face_frac:.3f} crop_box={box}{capped} "
                              f"-> upscaled to {refine_size}px, denoise={denoise}", flush=True)
    payload = broadcast_object(payload, src=0)

    if not payload["ok"]:
        if do_log:
            print(f"[refine] skipped: {payload.get('reason')}", flush=True)
        res.output_image = base_image
        return res

    crop_up = payload["crop_up"]  # identical PIL on every rank
    box = payload["box"]

    inner = generate(
        source_image=source_image,
        image_size=(refine_size, refine_size),
        init_image=crop_up,
        target_face_image=crop_up,
        denoise_strength=denoise,
    )
    res.inner = inner
    res.crop_box = box

    if is_main():
        refined = inner.output_image
        res.refined_crop = refined
        face_rec = RefinedFace(crop=refined, mask_pixel=inner.face_mask_pixel,
                               box=box, ref_index=0)
        res.faces = [face_rec]
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            try:
                # Clear, self-explanatory names. (base.png + refined.png are
                # written by the launcher into the same folder.)
                crop_up.save(os.path.join(debug_dir, "base_crop.png"))
                if inner.i1_aligned is not None:
                    inner.i1_aligned.save(os.path.join(debug_dir, "I0_aligned.png"))
                if refined is not None:
                    refined.save(os.path.join(debug_dir, "crop_refined.png"))
                if inner.face_mask_pixel is not None:
                    m = (np.clip(inner.face_mask_pixel, 0, 1) * 255).astype("uint8")
                    Image.fromarray(m).save(os.path.join(debug_dir, "mask.png"))
                blend = getattr(inner, "blend_mask_pixel", None)
                if blend is not None and blend is not inner.face_mask_pixel:
                    m = (np.clip(blend, 0, 1) * 255).astype("uint8")
                    Image.fromarray(m).save(os.path.join(debug_dir, "blend_mask.png"))
            except Exception as e:  # noqa: BLE001
                if do_log:
                    print(f"[refine] debug save failed: {e}", flush=True)
        if refined is None or not inner.alignment_applied or inner.face_mask_pixel is None:
            if do_log:
                print("[refine] no alignment in refine pass -> keeping base image", flush=True)
            res.output_image = base_image
            return res
        res.output_image = composite_refined_face(
            base_image.convert("RGB"), refined, inner.face_mask_pixel, box,
            refine_size=refine_size, composite_strength=composite_strength,
            composite_feather_frac=composite_feather_frac,
            composite_mode=composite_mode,
            composite_dilate_frac=composite_dilate_frac,
        )
        res.applied = True
        face_rec.applied = True
        if do_log:
            print("[refine] composited refined face back into base image.", flush=True)
    return res


# ---------------------------------------------------------------------------
# Multi-person refinement (ArcFace identity assignment + non-overlap crops)
# ---------------------------------------------------------------------------

def refine_faces_multi(
    *,
    base_image: Image.Image | None,
    source_images: list,             # ordered person references (paths or PIL)
    generate: GenerateFn,
    snap_size: Optional[SnapSizeFn] = None,
    refine_size: int = 1024,
    crop_scale: float = 2.4,
    denoise: float = 0.5,
    composite_strength: float = 1.0,
    composite_feather_frac: float = 0.05,
    composite_mode: str = "falloff",
    composite_dilate_frac: float = 0.25,
    min_sim: float = 0.05,
    square_crops: bool = True,
    min_upscale: float = 2.0,
    debug_dir: Optional[str] = None,
    verbose: bool = True,
) -> RefineResult:
    """Refine every face in ``base_image`` with its ArcFace-matched reference.

    Every refinement parameter means the same thing as in :func:`refine_face`;
    the only thing the several faces change is which crop goes to which
    reference. Faces are taken one at a time, largest first, and each is pasted
    before the next is cut, so a crop always sees the finished work of the ones
    before it. Crops are taken to ``refine_size`` on their long side and
    snapped by ``snap_size`` (the backbone's discrete resolution grid); pass
    None to snap to a multiple of 32.

    ``square_crops`` sizes each crop with :func:`utils.compositing.face_crop_boxes`
    -- square, as large as the scene allows, overlapping a neighbour's crop but
    never reaching its face. It defaults on; off restores the older
    :func:`utils.compositing.clip_box_to_neighbors`, which keeps the crops
    disjoint at the cost of cutting them rectangular.
    """
    do_log = verbose and is_main()
    snap = snap_size or _default_snap
    res = RefineResult()

    payload = None
    if is_main():
        if base_image is None:
            payload = {"ok": False, "reason": "no base image"}
        else:
            base_pil = base_image.convert("RGB")
            W, H = base_pil.size
            gen_items = detect_faces_with_embeddings(base_pil)
            ref_embs = []
            for s in source_images:
                spil = Image.open(s).convert("RGB") if isinstance(s, str) else s.convert("RGB")
                ref_embs.append(primary_embedding(spil))
            valid_ref = [i for i, e in enumerate(ref_embs) if e is not None]
            if not gen_items or not valid_ref:
                payload = {"ok": False,
                           "reason": f"faces={len(gen_items)} valid_refs={len(valid_ref)}"}
            else:
                fe = np.stack([it[1] for it in gen_items])        # (M, D)
                re_ = np.stack([ref_embs[i] for i in valid_ref])  # (K, D)
                pairs, _sim = assign_faces_to_refs(fe, re_, min_sim)
                centers = [np.asarray(it[0].center, dtype=float) for it in gen_items]
                max_side = (float(refine_size) / min_upscale
                            if min_upscale > 1.0 else None)
                boxes = (face_crop_boxes(
                    [gen_items[gi][0].bbox for gi, _ci, _s in pairs],
                    (W, H), crop_scale=crop_scale, max_side=max_side)
                    if square_crops else None)
                plan = []
                for p, (gi, ci, s) in enumerate(pairs):
                    face = gen_items[gi][0]
                    face_side = float(max(face.width, face.height))
                    cx, cy = float(face.center[0]), float(face.center[1])
                    if boxes is not None:
                        box = boxes[p]
                        bx0i, by0i, bx1i, by1i = box
                    else:
                        side = cap_crop_side(face_side * crop_scale, face_side,
                                             refine_size, min_upscale)
                        box0 = square_crop_box(cx, cy, side, W, H)
                        neighbors = [centers[j] for j in range(len(centers))
                                     if j != gi]
                        bx0, by0, bx1, by1 = clip_box_to_neighbors(
                            box0, cx, cy, neighbors)
                        bx0i, by0i = int(round(bx0)), int(round(by0))
                        bx1i = max(int(round(bx1)), bx0i + 8)
                        by1i = max(int(round(by1)), by0i + 8)
                        box = (bx0i, by0i, bx1i, by1i)
                    bw, bh = bx1i - bx0i, by1i - by0i
                    # Same working resolution as the single-face path:
                    # ``refine_size`` is the side a crop is taken to, not a cap,
                    # and a face refined at its native 250px is not the same
                    # operation as one refined at 1024. The clipped box may be
                    # rectangular, so the long side lands on ``refine_size``,
                    # the aspect is kept, and the backbone's grid snaps it.
                    scale = float(refine_size) / float(max(bw, bh))
                    gw, gh = snap(max(1, int(round(bw * scale))),
                                  max(1, int(round(bh * scale))))
                    plan.append({
                        "box": box, "gen_size": (gh, gw),
                        "ref_index": valid_ref[ci], "sim": s, "face_side": face_side,
                    })
                # Refine larger faces first (more context/tokens).
                plan.sort(key=lambda p: p["face_side"], reverse=True)
                payload = {"ok": True, "plan": plan, "n_faces": len(gen_items)}
                if do_log:
                    summ = ", ".join(
                        f"face#{k}->ref{p['ref_index']}(sim={p['sim']:.2f})"
                        for k, p in enumerate(plan)
                    )
                    print(f"[multi-refine] {len(gen_items)} face(s), "
                          f"{len(plan)} matched: {summ}", flush=True)
    payload = broadcast_object(payload, src=0)

    if not payload["ok"]:
        if do_log:
            print(f"[multi-refine] skipped: {payload.get('reason')}", flush=True)
        res.output_image = base_image
        return res

    plan = payload["plan"]
    running = base_image.convert("RGB") if (is_main() and base_image is not None) else None
    applied_any = False
    last_inner = None

    for k, item in enumerate(plan):
        box = item["box"]
        gh, gw = item["gen_size"]
        # Cut from what the earlier passes have already produced, not from the
        # untouched base. Crops are allowed to overlap here, so a later face
        # may look into a region an earlier one has refined; taking it from the
        # base would show it the stale pixels and paste them back over the good
        # ones. Rank 0 owns the running image, so the crop is shared from there.
        crop_up = broadcast_object(
            running.crop(box).resize((gw, gh), Image.LANCZOS)
            if (is_main() and running is not None) else None, src=0)
        source_image = source_images[item["ref_index"]]
        if do_log:
            print(f"[multi-refine] pass {k + 1}/{len(plan)} "
                  f"ref={item['ref_index']} box={box} "
                  f"({box[2] - box[0]}x{box[3] - box[1]} -> {gw}x{gh})", flush=True)
        inner = generate(
            source_image=source_image,
            image_size=item["gen_size"],
            init_image=crop_up,
            target_face_image=crop_up,
            denoise_strength=denoise,
        )
        last_inner = inner

        if is_main() and debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            tag = f"face{k}_ref{item['ref_index']}"
            try:
                crop_up.save(os.path.join(debug_dir, f"{tag}_base_crop.png"))
                if inner.i1_aligned is not None:
                    inner.i1_aligned.save(os.path.join(debug_dir, f"{tag}_I0_aligned.png"))
                if inner.output_image is not None:
                    inner.output_image.save(os.path.join(debug_dir, f"{tag}_crop_refined.png"))
                if inner.face_mask_pixel is not None:
                    m = (np.clip(inner.face_mask_pixel, 0, 1) * 255).astype("uint8")
                    Image.fromarray(m).save(os.path.join(debug_dir, f"{tag}_mask.png"))
            except Exception as e:  # noqa: BLE001
                if do_log:
                    print(f"[multi-refine] {tag}: debug save failed: {e}", flush=True)

        if is_main():
            refined = inner.output_image
            face_rec = RefinedFace(crop=refined, mask_pixel=inner.face_mask_pixel,
                                   box=box, ref_index=item["ref_index"])
            res.faces.append(face_rec)
            if refined is not None and inner.alignment_applied and inner.face_mask_pixel is not None:
                running = composite_refined_face(
                    running, refined, inner.face_mask_pixel, box,
                    refine_size=refine_size, composite_strength=composite_strength,
                    composite_feather_frac=composite_feather_frac,
                    composite_mode=composite_mode,
                    composite_dilate_frac=composite_dilate_frac,
                )
                applied_any = True
                face_rec.applied = True
            elif do_log:
                print(f"[multi-refine] pass {k + 1}: no alignment -> face kept as-is", flush=True)

    res.inner = last_inner
    if is_main():
        res.output_image = running if running is not None else base_image
        res.applied = applied_any
        if res.faces:
            # ``plan`` is ordered largest face first, so these name the most
            # prominent person. They used to be left unset on this path, which
            # silently cost every multi-person run its crops.
            res.refined_crop = res.faces[0].crop
            res.crop_box = res.faces[0].box
        if do_log:
            print(f"[multi-refine] done. refined {len(plan)} face(s), applied={applied_any}", flush=True)
    return res
