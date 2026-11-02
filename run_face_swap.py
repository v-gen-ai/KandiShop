"""Head swap on FLUX.2 klein-9B: the reference's face and hair onto a photo.

Scene, pose, body and light come from ``--target``; the head from
``--reference``. The head is cropped at ``--crop-scale`` face sizes (big enough
for both heads of hair), regenerated with a parsed head mask as the blend
region, and the whole crop is pasted back. See :mod:`flux2.head_swap`.

    python run_face_swap.py --target scene.png --reference person.png \
        --seed 0 1 2 3 4 --denoise 1.0 0.75 --save out/seed{seed}_dn{dn}.png

``{seed}`` (two digits) and ``{dn}`` (0.75 -> 075, 1.0 -> 10) are needed in
``--save`` when there are several seeds or several denoise values; one model
load covers them all. ``--mask-mode parsing`` keeps the photo's face outline
(faces intersected, hair unioned); ``--include-hair 0`` keeps its hairstyle.
The face-parsing checkpoint defaults to ``weights/face_parsing_79999_iter.pth``
(``FACE_PARSING_CKPT`` or ``--parsing-ckpt`` to override).
"""

from __future__ import annotations

import argparse
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flux2.head_swap import COMPOSITE_AREAS, make_head_swap_generate  # noqa: E402
from flux2.pipeline import DEFAULT_MODEL, PROMPT_COPY, load_flux, sampling  # noqa: E402
from utils import StageTimer, refine_face  # noqa: E402
from utils.head_mask import DEFAULT_PARSING_CKPT, MASK_MODES, HeadMaskBuilder  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True, help="the photo to keep")
    p.add_argument("--reference", required=True, help="whose head to put in")
    p.add_argument("--save", required=True, help="output path; {seed} / {dn} placeholders")
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--steps", type=int, default=None, help="default: the checkpoint's (klein 4)")
    p.add_argument("--guidance", type=float, default=None, help="default: the checkpoint's (klein 1.0)")
    p.add_argument("--seed", type=int, nargs="+", default=[0])
    p.add_argument("--denoise", type=float, nargs="+", default=[1.0, 0.75])
    p.add_argument("--prompt-copy", default=PROMPT_COPY, help="prompt of both streams")
    p.add_argument("--mask-mode", default="union", choices=MASK_MODES,
                   help="union: both heads whole; parsing: faces intersected, hair unioned")
    p.add_argument("--include-hair", type=int, default=1, help="0 keeps the photo's hairstyle")
    p.add_argument("--mask-blur-frac", type=float, default=0.15, help="x inter-ocular distance")
    p.add_argument("--mask-dilate-frac", type=float, default=0.06, help="x inter-ocular distance")
    p.add_argument("--crop-scale", type=float, default=4.4, help="crop side = crop-scale x face size")
    p.add_argument("--refine-size", type=int, default=1024)
    p.add_argument("--composite-area", default="crop", choices=COMPOSITE_AREAS,
                   help="crop: paste the whole crop; mask: paste the head mask only")
    p.add_argument("--composite-margin-frac", type=float, default=0.04)
    p.add_argument("--composite-feather-frac", type=float, default=0.02)
    p.add_argument("--adain-alpha", type=float, default=1.0, help="0 = off")
    p.add_argument("--adain-last-layers", type=int, default=-1, help="-1 = every layer")
    p.add_argument("--adain-skip-first-steps", type=int, default=1)
    p.add_argument("--parsing-ckpt", default=DEFAULT_PARSING_CKPT)
    p.add_argument("--save-debug", type=int, default=0,
                   help="1: crop, aligned reference and mask into <save stem>_debug/")
    return p.parse_args()


def dn_tag(denoise: float) -> str:
    s = f"{denoise:g}"
    return (s if "." in s else s + ".0").replace(".", "")


def main():
    args = parse_args()
    for path in (args.target, args.reference):
        if not os.path.exists(path):
            raise SystemExit(f"{path} does not exist")
    if len(args.seed) > 1 and "{seed}" not in args.save:
        raise SystemExit("several --seed values need {seed} in --save")
    if len(args.denoise) > 1 and "{dn}" not in args.save:
        raise SystemExit("several --denoise values need {dn} in --save")
    steps, guidance = sampling(args.model_id, args.steps, args.guidance)
    target = Image.open(args.target).convert("RGB")
    reference = Image.open(args.reference).convert("RGB")

    masks = HeadMaskBuilder(args.parsing_ckpt)
    pipeline = load_flux(args.model_id)
    print(f"[swap] steps={steps} guidance={guidance} mask={args.mask_mode} "
          f"hair={bool(args.include_hair)} crop={args.crop_scale} paste={args.composite_area} "
          f"denoise={args.denoise}", flush=True)
    timer = StageTimer(os.path.basename(args.model_id))

    for seed in args.seed:
        generate = make_head_swap_generate(
            pipeline, masks, prompt_target=args.prompt_copy, prompt_reference=args.prompt_copy,
            steps=steps, guidance_scale=guidance, seed=seed,
            include_hair=bool(args.include_hair), mask_mode=args.mask_mode,
            composite_area=args.composite_area, composite_margin_frac=args.composite_margin_frac,
            crop_scale=args.crop_scale, blur_sigma_frac=args.mask_blur_frac,
            dilate_frac=args.mask_dilate_frac, adain_alpha=args.adain_alpha,
            adain_last_layers=args.adain_last_layers,
            adain_skip_first_steps=args.adain_skip_first_steps,
        )
        for denoise in args.denoise:
            save = args.save.replace("{seed}", f"{seed:02d}").replace("{dn}", dn_tag(denoise))
            os.makedirs(os.path.dirname(os.path.abspath(save)), exist_ok=True)
            stem, _ = os.path.splitext(save)
            with timer.stage(f"swap_{denoise:g}"):
                rr = refine_face(
                    base_image=target, source_image=reference, generate=generate,
                    refine_size=args.refine_size, crop_scale=args.crop_scale, denoise=denoise,
                    composite_feather_frac=args.composite_feather_frac,
                    # No upscale cap: on a close-up it would crop the hair out.
                    composite_mode="canon", min_upscale=1.0,
                    debug_dir=f"{stem}_debug" if args.save_debug else None,
                )
            (rr.output_image or target).save(save)
            print(f"[swap] seed={seed} denoise={denoise:g} applied={rr.applied} -> {save}", flush=True)
    print(timer.summary(), flush=True)


if __name__ == "__main__":
    main()
