"""Loading FLUX.2 klein and generating the base image.

The base is kept apart from every refine on purpose: each refine reads a
finished PIL image and nothing else, so it works the same on a base generated
here and on any picture the caller brings.
"""

from __future__ import annotations

import torch
from PIL import Image

from flux2.generate import generate_dual_stream_flux2
from flux2.model_defaults import flux2_step_defaults
from utils import load_pipeline

DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-9B"
PROMPT_COPY = "Make no edits."


def load_flux(model_id: str = DEFAULT_MODEL, device: str = "cuda"):
    """The diffusers pipeline for ``model_id`` (Flux2KleinPipeline for klein)."""
    from diffusers import DiffusionPipeline

    print(f"[flux2] loading {model_id} ...", flush=True)
    pipeline = load_pipeline(DiffusionPipeline, model_id, torch_dtype=torch.bfloat16).to(device)
    print(f"[flux2] pipeline class: {type(pipeline).__name__}", flush=True)
    return pipeline


def sampling(model_id: str, steps: int | None = None,
             guidance: float | None = None) -> tuple[int, float]:
    """``(steps, guidance)``: the checkpoint's defaults unless overridden."""
    d_steps, d_guidance = flux2_step_defaults(model_id)
    return (d_steps if steps is None else int(steps),
            d_guidance if guidance is None else float(guidance))


def parse_size(size: str) -> tuple[int, int]:
    """``'HxW'`` -> ``(height, width)``."""
    h, w = (int(x) for x in size.lower().split("x"))
    return h, w


def sequence_length(pipeline, prompt: str) -> int:
    """A ``max_sequence_length`` the whole prompt fits in: 512, or more for a
    longer prompt, which the text encoder would otherwise cut at 512 tokens."""
    tok = pipeline.tokenizer
    try:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except Exception:  # noqa: BLE001 -- a tokenizer without a chat template
        text = prompt
    n = len(tok(text)["input_ids"])
    return max(512, (n + 63) // 64 * 64)


@torch.no_grad()
def generate_base(
    pipeline,
    prompt: str,
    references: list[Image.Image] | None = None,
    *,
    height: int = 1024,
    width: int = 1024,
    steps: int = 4,
    guidance: float = 1.0,
    seed: int = 42,
) -> Image.Image:
    """One plain single-stream generation.

    With ``references`` they are the image conditions (the in-context base the
    face and object refines start from); without, plain text-to-image.
    """
    g = torch.Generator(device="cuda").manual_seed(int(seed))
    max_len = sequence_length(pipeline, prompt)
    if max_len > 512:
        print(f"[flux2] long prompt: max_sequence_length={max_len}", flush=True)
    if references:
        return generate_dual_stream_flux2(
            pipeline, image=list(references), prompt_gen=prompt, prompt_copy=PROMPT_COPY,
            t_switch=steps, t_end=steps,          # dual-stream off -> base
            num_inference_steps=steps, guidance_scale=guidance,
            height=height, width=width, generator=g, verbose=False,
            max_sequence_length=max_len,
        ).output_image
    return pipeline(prompt=prompt, height=height, width=width, num_inference_steps=steps,
                    guidance_scale=guidance, max_sequence_length=max_len,
                    generator=g).images[0]
