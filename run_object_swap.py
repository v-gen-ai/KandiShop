"""Object swap on FLUX.2 klein-9B: the reference object in place of one in a photo.

Grounding DINO finds ``--object-class`` in ``--target`` and ``--source-class``
(default: the same) in ``--reference``. The new object is placed on the old
one (``--place``), scaled so that ``--size-metric`` matches (times
``--size-ratio``), and the two silhouettes are unioned so the old object
disappears. The crop around it is regenerated and pasted back.

    python run_object_swap.py --target scene.png --reference jar.png \
        --object-class bottle --save out/swap.png
"""

from __future__ import annotations

import argparse
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flux2.pipeline import DEFAULT_MODEL, load_flux, sampling  # noqa: E402
from flux2.product import refine_product  # noqa: E402
from objects import swap_spec  # noqa: E402
from utils import StageTimer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True, help="the photo to keep")
    p.add_argument("--reference", required=True, help="the new object's photo")
    p.add_argument("--object-class", required=True,
                   help="short English class name of the object to replace, e.g. 'bottle'")
    p.add_argument("--source-class", default=None,
                   help="class to find in the reference (default: --object-class)")
    p.add_argument("--save", required=True)
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--steps", type=int, default=None, help="default: the checkpoint's (klein 4)")
    p.add_argument("--guidance", type=float, default=None, help="default: the checkpoint's (klein 1.0)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--denoise", type=float, default=1.0)
    p.add_argument("--size-metric", default="bbox_width",
                   choices=["bbox_width", "bbox_height", "sqrt_area", "bbox_max", "bbox_diag"],
                   help="what is made equal between the two objects")
    p.add_argument("--size-ratio", type=float, default=1.0, help="new object's size over the old one's")
    p.add_argument("--place", default="bbox", choices=["bbox", "bottom", "centroid"],
                   help="align the bbox centres, the bottom edges or the mask centroids")
    p.add_argument("--refine-size", type=int, default=1024)
    p.add_argument("--refine-crop-scale", type=float, default=1.2,
                   help="fallback when the reference object cannot be measured")
    p.add_argument("--refine-crop-min", type=float, default=1.2,
                   help="floor of the auto crop, in object long sides")
    p.add_argument("--object-dilate-frac", type=float, default=0.04)
    p.add_argument("--object-blur-frac", type=float, default=0.03)
    p.add_argument("--object-conf-weight", type=float, default=1.0)
    p.add_argument("--composite-dilate-frac", type=float, default=0.10)
    p.add_argument("--composite-edge-frac", type=float, default=0.08)
    p.add_argument("--save-debug", type=int, default=0,
                   help="1: crop, aligned reference, mask and paste box into <save stem>_debug/")
    return p.parse_args()


def main():
    args = parse_args()
    for path in (args.target, args.reference):
        if not os.path.exists(path):
            raise SystemExit(f"{path} does not exist")
    steps, guidance = sampling(args.model_id, args.steps, args.guidance)
    stem, _ = os.path.splitext(args.save)
    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
    target = Image.open(args.target).convert("RGB")
    ref = Image.open(args.reference).convert("RGB")

    pipeline = load_flux(args.model_id)
    spec = swap_spec(args.object_class, args.source_class, size_metric=args.size_metric,
                     size_ratio=args.size_ratio, place=args.place,
                     dilate_frac=args.object_dilate_frac,
                     blur_sigma_frac=args.object_blur_frac, crop_min=args.refine_crop_min)
    print(f"[object-swap] steps={steps} guidance={guidance} denoise={args.denoise} "
          f"{args.source_class or args.object_class!r} for {args.object_class!r}, "
          f"equal {args.size_metric} x{args.size_ratio}", flush=True)
    timer = StageTimer(os.path.basename(args.model_id))

    with timer.stage("swap"):
        rr, geometry = refine_product(
            pipeline, target, ref, args.object_class, steps=steps, guidance=guidance,
            seed=args.seed, denoise=args.denoise, refine_size=args.refine_size,
            crop_scale=args.refine_crop_scale, crop_min=args.refine_crop_min,
            object_dilate_frac=args.object_dilate_frac,
            object_blur_frac=args.object_blur_frac,
            object_conf_weight=args.object_conf_weight,
            composite_dilate_frac=args.composite_dilate_frac,
            composite_edge_frac=args.composite_edge_frac,
            debug_dir=f"{stem}_debug" if args.save_debug else None, spec=spec,
        )
    (rr.output_image or target).save(args.save)
    print(f"[object-swap] done (applied={rr.applied}, fit={geometry.rung}, "
          f"mask={geometry.mask_mean:.3f}) -> {args.save}", flush=True)
    print(timer.summary(), flush=True)


if __name__ == "__main__":
    main()
