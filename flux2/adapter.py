"""FLUX.2 implementation of :class:`utils.DualStreamAdapter`.

Replays ``Flux2Transformer2DModel.forward`` one block at a time. FLUX.2 has two
block phases: ``n_double`` double-stream blocks (text and image are separate
tensors) followed by ``n_single`` single-stream blocks (everything in one
concatenated tensor). The adapter hides that transition so the driver can
read/write the noise-token slice uniformly. Ported from the reference
``flux2_dual_stream.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class _Flux2State:
    """Opaque per-stream state tracking both FLUX.2 phases."""
    text_stream: torch.Tensor | None
    image_stream: torch.Tensor | None
    concat_stream: torch.Tensor | None
    in_concat: bool

    temb: torch.Tensor
    mod_double_img: torch.Tensor
    mod_double_txt: torch.Tensor
    mod_single: torch.Tensor
    rotary_concat: tuple[torch.Tensor, torch.Tensor]

    n_txt: int
    n_lat: int
    n_cond: int


class Flux2DualStreamAdapter:
    """Block-by-block replay of ``Flux2Transformer2DModel.forward`` (inference)."""

    def __init__(self, transformer):
        self.transformer = transformer
        self.n_double = len(transformer.transformer_blocks)
        self.n_single = len(transformer.single_transformer_blocks)

    def num_blocks(self) -> int:
        return self.n_double + self.n_single

    @torch.no_grad()
    def prepare_state(
        self, *,
        noise_lat: torch.Tensor,
        cond_lat: torch.Tensor | None,
        text_embeds: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor,
        noise_ids: torch.Tensor,
        cond_ids: torch.Tensor | None,
        text_ids: torch.Tensor,
    ) -> _Flux2State:
        tr = self.transformer
        dtype = tr.dtype

        if cond_lat is not None:
            hs = torch.cat([noise_lat, cond_lat], dim=1).to(dtype)
            img_ids = torch.cat([noise_ids, cond_ids], dim=1)
            n_cond = cond_lat.shape[1]
        else:
            hs = noise_lat.to(dtype)
            img_ids = noise_ids
            n_cond = 0

        text_embeds = text_embeds.to(dtype)

        ts = timestep.to(dtype) * 1000
        gd = guidance.to(dtype) * 1000
        temb = tr.time_guidance_embed(ts, gd)
        mod_di = tr.double_stream_modulation_img(temb)
        mod_dt = tr.double_stream_modulation_txt(temb)
        mod_s = tr.single_stream_modulation(temb)[0]

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if text_ids.ndim == 3:
            text_ids = text_ids[0]
        image_rotary_emb = tr.pos_embed(img_ids)
        text_rotary_emb = tr.pos_embed(text_ids)
        rotary_concat = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        image_stream = tr.x_embedder(hs)
        text_stream = tr.context_embedder(text_embeds)

        return _Flux2State(
            text_stream=text_stream, image_stream=image_stream,
            concat_stream=None, in_concat=False,
            temb=temb, mod_double_img=mod_di, mod_double_txt=mod_dt,
            mod_single=mod_s, rotary_concat=rotary_concat,
            n_txt=text_stream.shape[1], n_lat=noise_lat.shape[1], n_cond=n_cond,
        )

    @torch.no_grad()
    def run_block(self, idx: int, state: _Flux2State) -> _Flux2State:
        tr = self.transformer

        if idx < self.n_double:
            block = tr.transformer_blocks[idx]
            text_out, image_out = block(
                hidden_states=state.image_stream,
                encoder_hidden_states=state.text_stream,
                temb_mod_params_img=state.mod_double_img,
                temb_mod_params_txt=state.mod_double_txt,
                image_rotary_emb=state.rotary_concat,
                joint_attention_kwargs=None,
            )
            state.text_stream = text_out
            state.image_stream = image_out
            return state

        # Single-stream phase: lazily build the concat tensor on first entry.
        if not state.in_concat:
            state.concat_stream = torch.cat([state.text_stream, state.image_stream], dim=1)
            state.text_stream = None
            state.image_stream = None
            state.in_concat = True

        block = tr.single_transformer_blocks[idx - self.n_double]
        out = block(
            hidden_states=state.concat_stream,
            encoder_hidden_states=None,
            temb_mod_params=state.mod_single,
            image_rotary_emb=state.rotary_concat,
            joint_attention_kwargs=None,
        )
        state.concat_stream = out if not isinstance(out, tuple) else out[0]
        return state

    def get_noise_tokens(self, state: _Flux2State) -> torch.Tensor:
        if state.in_concat:
            return state.concat_stream[:, state.n_txt:state.n_txt + state.n_lat]
        return state.image_stream[:, :state.n_lat]

    def set_noise_tokens(self, state: _Flux2State, new_noise: torch.Tensor) -> _Flux2State:
        if state.in_concat:
            cs = state.concat_stream
            cs[:, state.n_txt:state.n_txt + state.n_lat] = new_noise.to(dtype=cs.dtype)
        else:
            ims = state.image_stream
            ims[:, :state.n_lat] = new_noise.to(dtype=ims.dtype)
        return state

    @torch.no_grad()
    def finalize_noise_pred(self, state: _Flux2State) -> torch.Tensor:
        tr = self.transformer
        if state.in_concat:
            image_part = state.concat_stream[:, state.n_txt:]
        else:
            image_part = state.image_stream  # n_lat + n_cond tokens
        h = tr.norm_out(image_part, state.temb)
        out = tr.proj_out(h)
        return out[:, :state.n_lat]
