"""Dual-stream face-preserving generation for FLUX.2 (base + img2img).

Extends the reference ``generate_dual_stream_flux2`` with the two ingredients
the refinement pass needs and that the original lacked:

* **img2img init** (``init_image`` + ``denoise_strength``): start the loop from a
  noised version of an image instead of pure noise;
* **upfront alignment target** (``target_face_image``): compute the alignment +
  mask before the loop from a clean image, so dual-stream blending is active
  from the very first (img2img) step -- no x0 probe.

The dual-stream window is ``[t_switch, t_end)``; set ``t_switch >=
num_inference_steps`` (or ``dual_stream=False`` in the launcher) to disable
dual-stream during the base pass entirely.

The blend itself has four variants, selected by the flags below and shared with
the other backbones (see :mod:`utils.dual_stream`):

* **crop-only** (``mask_scale=0``): no blending, the crop is simply regenerated
  from the prompt with the aligned reference as the image condition. The
  baseline that has to be beaten before any of the rest is worth its cost;
* **block** (default): the mask blends both streams after every block;
* **shared+gate** / **private+gate** (``divergence_gate``, with or without
  ``copy_private``): the mask is narrowed per block to the tokens where the two
  streams still agree, so the scene keeps the expression the reference cannot
  supply.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from diffusers.pipelines.flux2.pipeline_flux2 import compute_empirical_mu, retrieve_timesteps

from utils import (
    AlignMode,
    DivergenceGate,
    FaceInfo,
    MaskPerturb,
    dit_dual_stream_velocity,
    plan_alignment,
)
from utils.face_detection import detect_face_robust

from .adapter import Flux2DualStreamAdapter
from .latents import (
    FLUX2_MAX_INPUT_AREA,
    decode_packed_x0_flux2,
    encode_pil_to_packed_latents_flux2,
)
from .token_zoom import crop_and_upsample, dit_token_zoom_velocity, face_token_box

MULTIPLE_OF = 16  # FLUX.2 packed-token stride: vae_scale_factor(8) * patch(2)


@dataclass
class Flux2DualStreamResult:
    output_image: Image.Image | None = None
    height: int = 0
    width: int = 0
    t_switch: int = 0
    alignment_applied: bool = False
    face_source: FaceInfo | None = None
    face_target: FaceInfo | None = None
    i1_aligned: Image.Image | None = None
    x0_at_switch: Image.Image | None = None
    face_mask_pixel: np.ndarray | None = None
    blend_mask_pixel: np.ndarray | None = None
    face_mask_tokens: torch.Tensor | None = None
    x0_dumps: list = field(default_factory=list)


def _prep_cond(pipeline, i1_aligned, width, height, device):
    """Preprocess the aligned reference to the OUTPUT grid and pack it as the
    conditioning latents (so cond and noise share the same token grid)."""
    cond_t = pipeline.image_processor.preprocess(
        i1_aligned, height=height, width=width, resize_mode="crop",
    )
    latents, ids = pipeline.prepare_image_latents(
        images=[cond_t], batch_size=1, generator=None,
        device=device, dtype=pipeline.vae.dtype,
    )
    return latents, ids


@torch.no_grad()
def generate_dual_stream_flux2(
    pipeline,
    image,                       # identity reference I_0 (PIL or list); cond + face source
    prompt_gen: str,
    *,
    prompt_copy: str = "Make no edits.",
    t_switch: int = 5,
    t_end: int | None = None,
    num_inference_steps: int = 40,
    guidance_scale: float = 4.0,
    height: int | None = None,
    width: int | None = None,
    generator: torch.Generator | None = None,
    max_sequence_length: int = 512,
    text_encoder_out_layers: tuple[int, ...] | None = None,
    background_mode: str = "gray",
    scale_clamp: tuple[float, float] = (0.05, 20.0),
    align_mode: AlignMode = "similarity_5pt",
    mask_expand_factor: float = 1.6,
    mask_aspect: float = 1.3,
    mask_blur_sigma_frac: float = 0.5,
    mask_align_to_eyes: bool = True,
    rewind_to: int | None = None,
    mask_shape: str = "ellipse",
    mask_forehead: float = 0.55,
    mask_grow: float = 0.0,
    composite_mask_shape: str | None = None,
    composite_mask_expand_factor: float = 1.6,
    composite_mask_aspect: float = 1.3,
    composite_mask_blur_sigma_frac: float = 0.08,
    composite_mask_forehead: float = 0.55,
    composite_mask_grow: float = 0.0,
    blend_layer_start: float = 0.0,
    blend_layer_end: float = 1.0,
    copy_private: bool = False,
    divergence_gate: bool = False,
    divergence_q_lo: float = 0.05,
    divergence_q_hi: float = 0.35,
    divergence_layer_start: float = 0.0,
    divergence_shuffle: bool = False,
    divergence_dump: str | None = None,
    mask_scale: float = 1.0,
    mask_dropout: float = 0.0,
    token_zoom: bool = False,
    zoom_prompt: str = "Generate this person.",
    zoom_crop_scale: float = 2.4,
    zoom_size: int = 64,
    zoom_layer_start: float = 0.0,
    zoom_layer_end: float = 1.0,
    zoom_bypass: bool = False,
    init_image=None,
    denoise_strength: float = 1.0,
    target_face_image=None,
    gen_cond_image=None,
    x0_dump_dir: str | None = None,
    enable_vae_tiling: bool = True,
    adain_alpha: float = 0.0,
    adain_last_layers: int | None = None,
    adain_stats: str = "mask",
    adain_skip_first_steps: int = 0,
    verbose: bool = True,
) -> Flux2DualStreamResult:
    device = pipeline._execution_device
    if not isinstance(image, list):
        image = [image]
    if len(image) == 0:
        raise ValueError("`image` must contain at least one PIL.Image.")
    refs = [im.convert("RGB") if isinstance(im, Image.Image) else Image.open(im).convert("RGB")
            for im in image]
    # The alignment machinery reads one face off one reference and then replaces
    # the condition latents wholesale, which would silently drop refs 2..n.
    # Extra references are therefore single-stream only.
    if len(refs) > 1 and (target_face_image is not None or int(t_switch) < int(num_inference_steps)):
        raise ValueError(
            f"{len(refs)} reference images given, but the dual-stream/refine path "
            "conditions on a single aligned reference. Pass one image, or run "
            "single-stream (t_switch >= num_inference_steps, no target_face_image)."
        )
    source_pil = refs[0]
    image = refs

    # ── 1. Encode both prompts ─────────────────────────────────────────────
    # The hidden-state layers the transformer expects differ per checkpoint
    # family (dev/Mistral-3 uses 10/20/30, klein/Qwen3 uses 9/18/27); leaving
    # this unset lets each pipeline class apply its own trained default.
    layer_kw = ({} if text_encoder_out_layers is None
                else {"text_encoder_out_layers": text_encoder_out_layers})
    prompt_embeds_gen, text_ids_gen = pipeline.encode_prompt(
        prompt=prompt_gen, device=device, num_images_per_prompt=1,
        max_sequence_length=max_sequence_length, **layer_kw,
    )
    prompt_embeds_copy, _ = pipeline.encode_prompt(
        prompt=prompt_copy, device=device, num_images_per_prompt=1,
        max_sequence_length=max_sequence_length, **layer_kw,
    )
    prompt_embeds_zoom = None
    if token_zoom:
        prompt_embeds_zoom, _ = pipeline.encode_prompt(
            prompt=zoom_prompt, device=device, num_images_per_prompt=1,
            max_sequence_length=max_sequence_length, **layer_kw,
        )

    # ── 2. Preprocess condition image, derive resolution ───────────────────
    # One condition per reference, each sized on its own like ``Flux2Pipeline``
    # does it; the output resolution follows the first reference.
    multiple_of = pipeline.vae_scale_factor * 2
    cond_ts, cond_sizes = [], []
    for ref in refs:
        iw, ih = ref.size
        img_r = (pipeline.image_processor._resize_to_target_area(ref, FLUX2_MAX_INPUT_AREA)
                 if iw * ih > FLUX2_MAX_INPUT_AREA else ref)
        rw, rh = img_r.size
        rw = (rw // multiple_of) * multiple_of
        rh = (rh // multiple_of) * multiple_of
        cond_ts.append(pipeline.image_processor.preprocess(
            img_r, height=rh, width=rw, resize_mode="crop"))
        cond_sizes.append((rw, rh))
    rw, rh = cond_sizes[0]

    if width is None:
        width = rw
    if height is None:
        height = rh
    width = (width // multiple_of) * multiple_of
    height = (height // multiple_of) * multiple_of

    # ── 3. Noise latents + condition latents ───────────────────────────────
    num_channels_latents = pipeline.transformer.config.in_channels // 4
    latents, latent_ids = pipeline.prepare_latents(
        batch_size=1, num_latents_channels=num_channels_latents,
        height=height, width=width, dtype=prompt_embeds_gen.dtype,
        device=device, generator=generator,
    )
    image_latents, image_latent_ids = pipeline.prepare_image_latents(
        images=cond_ts, batch_size=1, generator=generator,
        device=device, dtype=pipeline.vae.dtype,
    )

    # ── 4. Timesteps + guidance ─────────────────────────────────────────────
    sigmas_np = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    if getattr(pipeline.scheduler.config, "use_flow_sigmas", False):
        sigmas_np = None
    mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=num_inference_steps)
    timesteps, num_inference_steps = retrieve_timesteps(
        pipeline.scheduler, num_inference_steps, device, sigmas=sigmas_np, mu=mu,
    )
    guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0])

    result = Flux2DualStreamResult(height=height, width=width, t_switch=t_switch)
    adapter = Flux2DualStreamAdapter(pipeline.transformer)
    n_blocks = adapter.num_blocks()
    blend_layers = (int(round(blend_layer_start * n_blocks)), int(round(blend_layer_end * n_blocks)))
    token_h, token_w = height // MULTIPLE_OF, width // MULTIPLE_OF

    gate = DivergenceGate(
        q_lo=divergence_q_lo, q_hi=divergence_q_hi,
        layer_start=int(round(divergence_layer_start * n_blocks)),
        shuffle=divergence_shuffle,
    ) if divergence_gate else None
    perturb = MaskPerturb(scale=mask_scale, dropout=mask_dropout)
    perturb = perturb if perturb.active else None
    # A zero mask makes the copy stream unobservable: the velocity is read off
    # the gen stream, which never sees a blended token. Run one stream instead
    # of two -- same image, half the compute.
    crop_only = mask_scale == 0.0 and mask_dropout == 0.0 and gate is None

    t_end = num_inference_steps if t_end is None else int(t_end)
    use_dual = False
    mask_tokens = None
    x0_prev = None
    # Per-stream image conditions. They coincide (both see the aligned
    # reference) unless ``gen_cond_image`` is given, which turns the gen stream
    # into a reconstruction of the scene crop: the two streams then reproduce
    # two different images rather than one generating and one copying. Nothing
    # else changes -- same mask, same alignment, same composite.
    gen_lat, gen_ids = image_latents, image_latent_ids
    copy_lat, copy_ids = image_latents, image_latent_ids

    pipeline.scheduler.set_begin_index(0)

    # ── img2img init ────────────────────────────────────────────────────────
    start_i = 0
    if init_image is not None:
        start_i = int(round(num_inference_steps * (1.0 - float(denoise_strength))))
        start_i = max(0, min(start_i, num_inference_steps - 1))
        x0_init = encode_pil_to_packed_latents_flux2(
            pipeline, init_image.convert("RGB"), width=width, height=height,
            device=device, dtype=pipeline.vae.dtype, generator=generator,
        ).to(latents.dtype)
        sigma0 = pipeline.scheduler.sigmas[start_i].to(device=latents.device, dtype=latents.dtype)
        latents = (1.0 - sigma0) * x0_init + sigma0 * latents
        pipeline.scheduler.set_begin_index(start_i)
        if verbose:
            print(f"  [img2img] strength={denoise_strength} -> start step {start_i}/{num_inference_steps}")

    # ── in-block face zoom ──────────────────────────────────────────────────
    # The face box is read once, off the frame that is being denoised: the
    # geometry does not move during a pass, and re-detecting it on a noisy
    # intermediate would only add jitter.
    zoom_plan = zoom_ids = None
    if token_zoom:
        zoom_src = (init_image or target_face_image)
        zoom_face = detect_face_robust(zoom_src.convert("RGB")) if zoom_src is not None else None
        if zoom_face is None:
            if verbose:
                print("  [zoom] no face in the frame -> plain single stream")
        else:
            zoom_plan = face_token_box(
                zoom_face, width=width, height=height, token_h=token_h, token_w=token_w,
                crop_scale=zoom_crop_scale, size=zoom_size,
            )
            gh, gw = zoom_plan.grid_hw
            _, zoom_ids = pipeline.prepare_latents(
                batch_size=1, num_latents_channels=num_channels_latents,
                height=gh * MULTIPLE_OF, width=gw * MULTIPLE_OF,
                dtype=prompt_embeds_gen.dtype, device=device, generator=generator,
            )
            if verbose:
                bh, bw = zoom_plan.box_hw
                print(f"  [zoom] face box {zoom_plan.box} = {bw}x{bh} tokens "
                      f"-> {gw}x{gh} (x{zoom_plan.factor} per side)")
            if zoom_plan.factor == 1:
                # The crop already fills the frame; blowing it up would mean a
                # resolution the model was not trained at, for no extra detail.
                if verbose:
                    print("  [zoom] face already fills the frame -> nothing to zoom")
                zoom_plan = None

    def _try_align(target_face_src_image, x0_for_bg):
        """Shared alignment pass, then pack the result as condition latents.
        Returns (plan, cond_latents, cond_ids)."""
        plan = plan_alignment(
            source_pil, target_face_src_image, width=width, height=height,
            token_h=token_h, token_w=token_w,
            background_mode=background_mode, background_image=x0_for_bg,
            scale_clamp=scale_clamp, align_mode=align_mode,
            mask_expand_factor=mask_expand_factor, mask_aspect=mask_aspect,
            mask_blur_sigma_frac=mask_blur_sigma_frac,
            mask_align_to_eyes=mask_align_to_eyes,
            mask_shape=mask_shape, mask_forehead=mask_forehead, mask_grow=mask_grow,
            composite_mask_shape=composite_mask_shape,
            composite_mask_expand_factor=composite_mask_expand_factor,
            composite_mask_aspect=composite_mask_aspect,
            composite_mask_blur_sigma_frac=composite_mask_blur_sigma_frac,
            composite_mask_forehead=composite_mask_forehead,
            composite_mask_grow=composite_mask_grow,
        )
        if not plan.ok:
            return plan, None, None
        cond_latents, cond_ids = _prep_cond(pipeline, plan.i1_aligned, width, height, device)
        return plan, cond_latents, cond_ids

    # ── upfront alignment target (refine pass) ──────────────────────────────
    if target_face_image is not None:
        tgt = target_face_image.convert("RGB")
        plan, cond_lat, cond_ids = _try_align(tgt, tgt)
        result.face_source, result.face_target = plan.face_source, plan.face_target
        if plan.ok:
            copy_lat, copy_ids = cond_lat, cond_ids
            gen_lat, gen_ids = (
                _prep_cond(pipeline, gen_cond_image.convert("RGB"), width, height, device)
                if gen_cond_image is not None else (cond_lat, cond_ids)
            )
            mtok = plan.mask_tokens
            mask_tokens = mtok.to(device=device)
            result.i1_aligned, result.face_mask_pixel = plan.i1_aligned, plan.composite_pixel
            result.blend_mask_pixel = plan.mask_pixel
            result.face_mask_tokens = mtok.detach().cpu()
            use_dual = True
            result.alignment_applied = True
            if verbose:
                print(f"  [target] upfront alignment ok, mask cov={float(mtok.mean())*100:.2f}% "
                      f"-> dual from step {start_i}")
        elif verbose:
            print("  [target] face not detected -> single-stream")

    if verbose:
        variant = "crop-only" if crop_only else ("private" if copy_private else "shared")
        if gen_cond_image is not None:
            variant += "/gen-cond=scene"
        if gate is not None:
            variant += f"+gate(q={divergence_q_lo},{divergence_q_hi}"
            variant += ",shuffled)" if divergence_shuffle else ")"
        if perturb is not None:
            variant += f"+perturb(scale={mask_scale},drop={mask_dropout})"
        print(f"[flux2 dual-stream] steps={num_inference_steps} size={width}x{height} "
              f"guidance={guidance_scale} dual=[{t_switch},{t_end}) blend_layers={blend_layers} "
              f"blend={variant}")

    # ── Denoising loop ───────────────────────────────────────────────────────
    # ``rewind_to`` separates the two things ``t_switch`` conflates: where the
    # face becomes legible enough to align against, and where the blend should
    # start. A face is not there at step 0 and there is nothing to align to, but
    # by the time it is there the composition is settled and the blend arrives
    # too late to shape it. So the pass runs to ``t_switch``, aligns on that x0,
    # then goes back to the latent it had at ``rewind_to`` and re-runs those
    # steps with the alignment it could not have had the first time.
    latents_snapshot = None
    rewound = False
    pbar = tqdm(total=num_inference_steps - start_i, desc="Denoising", disable=not verbose)
    i = start_i
    while i < num_inference_steps:
        t = timesteps[i]
        sigma = pipeline.scheduler.sigmas[i]
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        if rewind_to is not None and i == rewind_to and not rewound:
            latents_snapshot = latents.detach().clone()

        # switch-over for the base pass (align on previous x0)
        if i == t_switch and not use_dual and x0_prev is not None:
            x0_image = decode_packed_x0_flux2(pipeline, x0_prev, latent_ids)
            result.x0_at_switch = x0_image
            plan, cond_lat, cond_ids = _try_align(x0_image, x0_image)
            result.face_source, result.face_target = plan.face_source, plan.face_target
            if plan.ok:
                copy_lat, copy_ids = cond_lat, cond_ids
                gen_lat, gen_ids = (
                    _prep_cond(pipeline, gen_cond_image.convert("RGB"), width, height, device)
                    if gen_cond_image is not None else (cond_lat, cond_ids)
                )
                mtok = plan.mask_tokens
                mask_tokens = mtok.to(device=device)
                result.i1_aligned, result.face_mask_pixel = plan.i1_aligned, plan.composite_pixel
                result.blend_mask_pixel = plan.mask_pixel
                result.face_mask_tokens = mtok.detach().cpu()
                use_dual = True
                result.alignment_applied = True
                if verbose:
                    print(f"  [t_switch={t_switch}] alignment ok, mask cov={float(mtok.mean())*100:.2f}%")
                if rewind_to is not None and latents_snapshot is not None and not rewound:
                    rewound = True
                    latents, x0_prev = latents_snapshot, None
                    pipeline.scheduler._step_index = None
                    pipeline.scheduler.set_begin_index(rewind_to)
                    i = rewind_to
                    if verbose:
                        print(f"  [rewind] back to step {rewind_to}, blending from there")
                    continue
            elif verbose:
                print(f"  [t_switch={t_switch}] face not detected -> single-stream")

        dual_now = use_dual and i < t_end and not crop_only

        if zoom_plan is not None:
            noise_lat_in = latents.to(pipeline.transformer.dtype)
            ts_in = timestep / 1000
            state_main = adapter.prepare_state(
                noise_lat=noise_lat_in, cond_lat=gen_lat.to(pipeline.transformer.dtype),
                text_embeds=prompt_embeds_gen, timestep=ts_in, guidance=guidance,
                noise_ids=latent_ids, cond_ids=gen_ids, text_ids=text_ids_gen,
            )
            state_zoom = adapter.prepare_state(
                noise_lat=crop_and_upsample(noise_lat_in, zoom_plan),
                cond_lat=gen_lat.to(pipeline.transformer.dtype),
                text_embeds=prompt_embeds_zoom, timestep=ts_in, guidance=guidance,
                noise_ids=zoom_ids, cond_ids=gen_ids, text_ids=text_ids_gen,
            )
            noise_pred = dit_token_zoom_velocity(
                adapter, state_main, state_zoom, zoom_plan,
                layers=(int(round(zoom_layer_start * n_blocks)),
                        int(round(zoom_layer_end * n_blocks))),
                bypass=zoom_bypass,
            )
        elif not dual_now:
            # single stream (warm-up, or tail after t_end): full forward w/ p_gen
            cat_lat = torch.cat([latents, gen_lat], dim=1).to(pipeline.transformer.dtype)
            cat_ids = torch.cat([latent_ids, gen_ids], dim=1)
            noise_pred = pipeline.transformer(
                hidden_states=cat_lat, timestep=timestep / 1000, guidance=guidance,
                encoder_hidden_states=prompt_embeds_gen, txt_ids=text_ids_gen,
                img_ids=cat_ids, joint_attention_kwargs=None, return_dict=False,
            )[0][:, :latents.shape[1]]
        else:
            noise_lat_in = latents.to(pipeline.transformer.dtype)
            ts_in = timestep / 1000
            state_gen = adapter.prepare_state(
                noise_lat=noise_lat_in, cond_lat=gen_lat.to(pipeline.transformer.dtype),
                text_embeds=prompt_embeds_gen,
                timestep=ts_in, guidance=guidance, noise_ids=latent_ids,
                cond_ids=gen_ids, text_ids=text_ids_gen,
            )
            state_copy = adapter.prepare_state(
                noise_lat=noise_lat_in, cond_lat=copy_lat.to(pipeline.transformer.dtype),
                text_embeds=prompt_embeds_copy,
                timestep=ts_in, guidance=guidance, noise_ids=latent_ids,
                cond_ids=copy_ids, text_ids=text_ids_gen,
            )
            step_k = i - start_i
            adain_now = (
                float(adain_alpha) != 0.0
                and adain_last_layers is not None
                and step_k >= int(adain_skip_first_steps)
            )
            if verbose and adain_last_layers is not None and float(adain_alpha) != 0.0:
                print(
                    f"  [adain] step_k={step_k} "
                    f"{'ON' if adain_now else 'off'} "
                    f"(skip_first={adain_skip_first_steps})",
                    flush=True,
                )
            noise_pred = dit_dual_stream_velocity(
                adapter, state_gen, state_copy, mask_tokens=mask_tokens, blend_layers=blend_layers,
                copy_private=copy_private, divergence_gate=gate, mask_perturb=perturb, step=i,
                adain_alpha=(adain_alpha if adain_now else 0.0),
                adain_last_layers=adain_last_layers,
                adain_stats=adain_stats,
            )

        x0_prev = (latents - sigma * noise_pred).detach()
        if x0_dump_dir is not None:
            import os
            os.makedirs(x0_dump_dir, exist_ok=True)
            phase = "dual" if dual_now else ("warm" if not use_dual else "tail")
            decode_packed_x0_flux2(pipeline, x0_prev.cpu(), latent_ids).save(
                os.path.join(x0_dump_dir, f"x0_step_{i:02d}_{phase}.png"))

        latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        i += 1
        pbar.update(1)
    pbar.close()

    if gate is not None and divergence_dump:
        gate.save(divergence_dump, token_h, token_w, mask=mask_tokens)

    # ── VAE decode ────────────────────────────────────────────────────────────
    if enable_vae_tiling and hasattr(pipeline.vae, "enable_tiling"):
        pipeline.vae.enable_tiling()
    result.output_image = decode_packed_x0_flux2(pipeline, latents.detach(), latent_ids)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if verbose:
        print(f"[flux2 dual-stream] done. alignment_applied={result.alignment_applied}")
    return result
