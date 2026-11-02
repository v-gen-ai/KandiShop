"""Product refine on FLUX.2 klein-9B, one image.

Either ``--prompt`` or ``--base``. With ``--prompt`` the base is generated with
the product photo as the image condition, then refined; both are saved (the
base as ``<save stem>_base.png``). With ``--base`` only the refine runs, on that
image. ``--object-class`` is what Grounding DINO looks for, in both pictures.

    python run_object.py --prompt "This lotion bottle on a kitchen counter" \
        --reference product.png --object-class bottle --save out/obj.png
    python run_object.py --base photo.png --reference product.png \
        --object-class bottle --save out/obj.png

To put a different object in place of the one found, see ``run_object_swap.py``.
"""

from __future__ import annotations

import argparse
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flux2.pipeline import DEFAULT_MODEL, generate_base, load_flux, parse_size, sampling  # noqa: E402
from flux2.product import refine_product  # noqa: E402
from utils import StageTimer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="generate the base from this prompt, then refine")
    src.add_argument("--base", help="refine this image; no generation")
    p.add_argument("--reference", required=True, help="the product photo")
    p.add_argument("--object-class", required=True,
                   help="short English class name, e.g. 'bottle', 'jar'")
    p.add_argument("--save", required=True)
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--image-size", default="1024x1024", help="'HxW' of a generated base")
    p.add_argument("--reference-in-base", type=int, default=1,
                   help="0: generate the base from the prompt alone, the reference goes to the refine only")
    p.add_argument("--steps", type=int, default=None, help="default: the checkpoint's (klein 4)")
    p.add_argument("--guidance", type=float, default=None, help="default: the checkpoint's (klein 1.0)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refine-denoise", type=float, default=0.7)
    p.add_argument("--refine-size", type=int, default=1024)
    p.add_argument("--refine-crop-scale", type=float, default=1.2,
                   help="fallback when the reference object cannot be measured")
    p.add_argument("--refine-crop-min", type=float, default=1.0,
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
    for path in (args.base, args.reference):
        if path and not os.path.exists(path):
            raise SystemExit(f"{path} does not exist")
    steps, guidance = sampling(args.model_id, args.steps, args.guidance)
    stem, ext = os.path.splitext(args.save)
    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
    debug_dir = f"{stem}_debug" if args.save_debug else None
    ref = Image.open(args.reference).convert("RGB")

    pipeline = load_flux(args.model_id)
    print(f"[object] steps={steps} guidance={guidance} denoise={args.refine_denoise} "
          f"class={args.object_class!r}", flush=True)
    timer = StageTimer(os.path.basename(args.model_id))

    if args.prompt:
        h, w = parse_size(args.image_size)
        with timer.stage("base"):
            base = generate_base(pipeline, args.prompt, [ref] if args.reference_in_base else None,
                                 height=h, width=w,
                                 steps=steps, guidance=guidance, seed=args.seed)
        base.save(f"{stem}_base{ext}")
        print(f"[object] base -> {stem}_base{ext}", flush=True)
    else:
        base = Image.open(args.base).convert("RGB")

    with timer.stage("refine"):
        rr, geometry = refine_product(
            pipeline, base, ref, args.object_class, steps=steps, guidance=guidance,
            seed=args.seed, denoise=args.refine_denoise, refine_size=args.refine_size,
            crop_scale=args.refine_crop_scale, crop_min=args.refine_crop_min,
            object_dilate_frac=args.object_dilate_frac,
            object_blur_frac=args.object_blur_frac,
            object_conf_weight=args.object_conf_weight,
            composite_dilate_frac=args.composite_dilate_frac,
            composite_edge_frac=args.composite_edge_frac, debug_dir=debug_dir,
        )
    (rr.output_image or base).save(args.save)
    print(f"[object] refined (applied={rr.applied}, fit={geometry.rung}, "
          f"inliers={geometry.inliers}, mask={geometry.mask_mean:.3f}) -> {args.save}",
          flush=True)
    print(timer.summary(), flush=True)


if __name__ == "__main__":
    main()
