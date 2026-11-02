"""Face refine on FLUX.2 klein-9B, one image.

Either ``--prompt`` or ``--base``. With ``--prompt`` the base is generated with
the references as image conditions, then refined; both are saved (the base as
``<save stem>_base.png``). With ``--base`` only the refine runs, on that image.

    python run_face.py --prompt "A photo of this man in a cafe" \
        --reference person.png --save out/face.png
    python run_face.py --base photo.png --reference person.png --save out/face.png

Several ``--refine-denoise`` values refine the same base in turn on one loaded
model; ``--save`` then needs a ``{dn}`` placeholder (0.5 -> 05, 0.75 -> 075,
1.0 -> 10):

    python run_face.py --base photo.png --reference person.png \
        --refine-denoise 0.5 0.75 1.0 --save out/face_dn{dn}.png
"""

from __future__ import annotations

import argparse
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flux2.face import refine_faces  # noqa: E402
from flux2.pipeline import DEFAULT_MODEL, generate_base, load_flux, parse_size, sampling  # noqa: E402
from utils import StageTimer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="generate the base from this prompt, then refine")
    src.add_argument("--base", help="refine this image; no generation")
    p.add_argument("--reference", nargs="+", required=True,
                   help="one identity reference per person")
    p.add_argument("--save", required=True)
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--image-size", default="1024x1024", help="'HxW' of a generated base")
    p.add_argument("--reference-in-base", type=int, default=1,
                   help="0: generate the base from the prompt alone, references go to the refine only")
    p.add_argument("--steps", type=int, default=None, help="default: the checkpoint's (klein 4)")
    p.add_argument("--guidance", type=float, default=None, help="default: the checkpoint's (klein 1.0)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refine-denoise", type=float, nargs="+", default=[0.75],
                   help="several values need {dn} in --save")
    p.add_argument("--adain", type=int, default=1, help="AdaIN on every blended step")
    p.add_argument("--refine-size", type=int, default=1024)
    p.add_argument("--refine-crop-scale", type=float, default=2.4)
    p.add_argument("--mask-expand", type=float, default=1.6)
    p.add_argument("--mask-aspect", type=float, default=1.3)
    p.add_argument("--mask-blur", type=float, default=0.5)
    p.add_argument("--save-debug", type=int, default=0,
                   help="1: crops, aligned reference and mask into <save stem>_debug/")
    return p.parse_args()


def dn_tag(denoise: float) -> str:
    s = f"{denoise:g}"
    return (s if "." in s else s + ".0").replace(".", "")


def main():
    args = parse_args()
    for path in [args.base] + args.reference:
        if path and not os.path.exists(path):
            raise SystemExit(f"{path} does not exist")
    if len(args.refine_denoise) > 1 and "{dn}" not in args.save:
        raise SystemExit("several --refine-denoise values need {dn} in --save")
    saves = [args.save.replace("{dn}", dn_tag(d)) for d in args.refine_denoise]
    steps, guidance = sampling(args.model_id, args.steps, args.guidance)
    for save in saves:
        os.makedirs(os.path.dirname(os.path.abspath(save)), exist_ok=True)
    refs = [Image.open(r).convert("RGB") for r in args.reference]

    pipeline = load_flux(args.model_id)
    print(f"[face] steps={steps} guidance={guidance} denoise={args.refine_denoise} "
          f"adain={bool(args.adain)}", flush=True)
    timer = StageTimer(os.path.basename(args.model_id))

    if args.prompt:
        h, w = parse_size(args.image_size)
        with timer.stage("base"):
            base = generate_base(pipeline, args.prompt, refs if args.reference_in_base else None,
                                 height=h, width=w,
                                 steps=steps, guidance=guidance, seed=args.seed)
        for save in saves:
            stem, ext = os.path.splitext(save)
            base.save(f"{stem}_base{ext}")
            print(f"[face] base -> {stem}_base{ext}", flush=True)
    else:
        base = Image.open(args.base).convert("RGB")

    for denoise, save in zip(args.refine_denoise, saves):
        stem, _ = os.path.splitext(save)
        with timer.stage(f"refine_{denoise:g}"):
            rr = refine_faces(
                pipeline, base, refs, steps=steps, guidance=guidance, seed=args.seed,
                denoise=denoise, adain=bool(args.adain),
                refine_size=args.refine_size, crop_scale=args.refine_crop_scale,
                mask_expand=args.mask_expand, mask_aspect=args.mask_aspect,
                mask_blur=args.mask_blur, debug_dir=f"{stem}_debug" if args.save_debug else None,
            )
        (rr.output_image or base).save(save)
        print(f"[face] refined (denoise={denoise:g}, applied={rr.applied}) -> {save}", flush=True)
    print(timer.summary(), flush=True)


if __name__ == "__main__":
    main()
