"""Model-agnostic dual-stream core: the block-level blend driver.

Two private denoising streams (``gen`` and ``copy``) are stepped through the
transformer one block at a time. After each block their image-token hidden
states are linearly blended by a soft face mask::

    h_img = h_copy * m + h_gen * (1 - m)

so identity (from the tightly-prompted, reference-conditioned copy stream) is
injected only inside the face region, while the scene (gen stream) owns the
rest. Everything model-specific is hidden behind :class:`DualStreamAdapter`;
each backbone implements that protocol over its own transformer.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from .distributed import rank


@runtime_checkable
class DualStreamAdapter(Protocol):
    """Block-level stepper for one transformer forward pass.

    An adapter wraps a backbone's transformer and exposes it as a sequence of
    blocks that can be replayed while reading/writing the image-token slice of
    the hidden state. ``state`` is an opaque per-stream object (whatever the
    adapter needs: hidden states, RoPE tables, KV cache, masks, ...).
    """

    def num_blocks(self) -> int:
        """Number of transformer blocks/layers to iterate over."""
        ...

    def run_block(self, idx: int, state: Any) -> Any:
        """Run block ``idx`` in place on ``state``; return the updated state."""
        ...

    def get_noise_tokens(self, state: Any) -> torch.Tensor:
        """Return the image (noise) token hidden states, shape ``(1, n_img, D)``."""
        ...

    def set_noise_tokens(self, state: Any, new_noise: torch.Tensor) -> Any:
        """Write blended image tokens back into ``state``; return it."""
        ...

    def finalize_noise_pred(self, state: Any) -> torch.Tensor:
        """Map the final hidden state to the flow-matching velocity prediction."""
        ...


@torch.no_grad()
def _masked_mean_std(
    x: torch.Tensor,
    m: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std of ``x`` (1, N, D) weighted by soft mask ``m`` (1, N, 1)."""
    w = m.clamp(0.0, 1.0)
    wsum = w.sum(dim=1, keepdim=True).clamp_min(eps)
    mean = (x * w).sum(dim=1, keepdim=True) / wsum
    var = ((x - mean).square() * w).sum(dim=1, keepdim=True) / wsum
    std = (var + eps).sqrt()
    return mean, std


@dataclasses.dataclass
class MaskPerturb:
    """Weaken the blend mask without looking at anything -- the gate's control.

    A gate that removes a fixed share of the mask can appear to save an
    expression for no better reason than that the mask got weaker. This
    reproduces that much and no more: ``scale`` thins the mask everywhere,
    ``dropout`` removes a random share of it and redraws at every block. If
    either matches the gate's result, the gate was measuring nothing.

    ``scale=0`` switches the blend off completely, which is also the cheapest
    way to get the single-stream baseline: the crop re-generated from the same
    prompt and the same reference condition, with no blending at all.

    The draw is seeded from the block and the step so that every rank removes
    the same tokens; the streams would otherwise diverge across ranks.
    """
    scale: float = 1.0
    dropout: float = 0.0
    seed: int = 1234

    @property
    def active(self) -> bool:
        return self.scale != 1.0 or self.dropout > 0.0

    def __call__(self, layer: int, step: int, mask: torch.Tensor) -> torch.Tensor:
        out = mask * self.scale
        if self.dropout > 0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + 1000 * step + layer)
            keep = torch.rand(mask.numel(), generator=generator) >= self.dropout
            out = out * keep.to(device=mask.device, dtype=mask.dtype).view(mask.shape)
        return out


@dataclasses.dataclass
class DivergenceGate:
    """Drop from the blend mask the tokens where the two streams disagree.

    Both streams start each forward from the same latents, so at the same token
    position their features normally agree to a cosine around 0.95. Where they
    do not -- a shouting mouth against a closed one, a shut eye against an open
    one -- the cosine falls away sharply and locally. Removing those tokens from
    the mask lets the scene keep what the reference cannot supply, and it needs
    no face model, no landmarks and no hand-set thresholds per region: the
    transformer states say it themselves, per block.

    The cut is a quantile of the agreement *inside the mask*, not an absolute
    cosine, because the absolute value drifts with depth -- in the early blocks
    the two streams are still identical, by the last ones nothing agrees to
    better than 0.9. A quantile asks the only question that survives that drift:
    which parts of this face agree least with the reference right now.

    Measured on HunyuanImage-3: with the copy stream re-synced to the blend
    (``copy_private=False``) the map is remade every block and lands on no
    stable region, yet a permutation control -- same weights, scattered over
    other tokens of the mask -- destroys the effect, so the placement is what
    matters. With a private copy stream the map is stable across blocks and
    sits on the mouth, but it arrives after the early blocks have already
    pasted the reference.
    """
    q_lo: float = 0.05          # below this quantile of agreement: dropped
    q_hi: float = 0.35          # above it: kept, linear ramp in between
    layer_start: int = 0        # the earliest blocks carry no signal
    shuffle: bool = False       # permutation control: same weights, wrong place
    log: list = dataclasses.field(default_factory=list)

    def __call__(self, layer: int, step: int, gen, copy, mask):
        if layer < self.layer_start:
            return None
        g = torch.nn.functional.normalize(gen[0].float(), dim=-1)
        c = torch.nn.functional.normalize(copy[0].float(), dim=-1)
        cosine = (g * c).sum(-1)                       # (n_img,)
        inside = mask.view(-1) > 0.05
        if int(inside.sum()) < 8:
            return None
        lo, hi = torch.quantile(
            cosine[inside], torch.tensor([self.q_lo, self.q_hi], device=cosine.device),
        )
        gate = ((cosine - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)
        if self.shuffle:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(7000 + 100 * step + layer)
            where = torch.nonzero(inside, as_tuple=True)[0]
            order = torch.randperm(len(where), generator=generator).to(where.device)
            gate = gate.clone().index_copy_(0, where, gate[where][order])
        self.log.append({
            "step": step, "layer": layer,
            # Stored as the deficit 1-cos: inside the mask the cosine sits so
            # close to one that half precision quantises the whole signal away,
            # while near zero the same sixteen bits are more than enough.
            "dissim": (1.0 - cosine).detach().float().cpu().numpy().astype(np.float16),
            "gate": gate.detach().float().cpu().numpy().astype(np.float16),
        })
        return gate.view(1, -1, 1).to(dtype=gen.dtype)

    def save(self, path: str, token_h: int, token_w: int, mask=None) -> None:
        """Write the per-block cosine and gate maps, to see what was excluded."""
        if not self.log:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = dict(
            step=np.array([r["step"] for r in self.log]),
            layer=np.array([r["layer"] for r in self.log]),
            dissim=np.stack([r["dissim"] for r in self.log]),
            gate=np.stack([r["gate"] for r in self.log]),
            token_hw=np.array([token_h, token_w]),
        )
        if mask is not None:
            payload["mask"] = mask.detach().float().cpu().numpy().reshape(-1)
        np.savez_compressed(path, **payload)
        print(f"    [gate] {len(self.log)} blocks -> {path}", flush=True)


@torch.no_grad()
def dit_dual_stream_velocity(
    adapter: DualStreamAdapter,
    state_gen: Any,
    state_copy: Any,
    *,
    mask_tokens: torch.Tensor,             # (n_img, 1) or broadcastable
    blend_layers: tuple[int, int] | None = None,  # [start, end) layer indices
    copy_private: bool = False,            # keep the copy stream off the blend
    divergence_gate: DivergenceGate | None = None,
    mask_perturb: MaskPerturb | None = None,
    step: int = 0,                         # denoising step, for the gate's log
    adain_alpha: float = 0.0,
    adain_last_layers: int | None = None,
    adain_stats: str = "mask",
    log_blocks: int = 0,                   # if >0 (rank0), progress every N layers
    debug_env: str = "TRUEFACE_DEBUG_BLEND",
) -> torch.Tensor:
    """Run both streams block-by-block, blending image tokens after each block.

    Both states must already hold the same image latents. The driver only
    touches the image-token slice; text / condition / cache stay private per
    state. ``blend_layers`` restricts the mix to a ``[start, end)`` layer range
    (outside it both streams still run to keep valid caches/residuals, but their
    image tokens are not mixed). ``None`` blends every block.

    ``copy_private`` stops the blended tokens from being written back into the
    copy stream: both streams still start the forward from the same latents,
    but from the first block on the copy stream is a plain reproduction of the
    reference rather than a re-reading of the mixture. That is what keeps the
    two states far enough apart for ``divergence_gate`` to have anything to
    measure -- and, on its own, it also changes what the copy stream is: a
    fixed suggestion rather than a controller correcting the merged state.

    ``divergence_gate`` narrows the mask per block to the tokens where the
    streams still agree; ``mask_perturb`` weakens it while looking at nothing,
    which is both the control for the gate and (at ``scale=0``) the
    single-stream baseline.

    ``adain_last_layers`` / ``adain_alpha`` optionally restyle the copy stream
    toward the gen stream's feature statistics before the mask blend, on the
    last N layers (or every layer when ``adain_last_layers < 0``). The restyled
    copy is ``lerp(copy, AdaIN(copy, gen), α)``. Off when ``adain_alpha=0`` or
    ``adain_last_layers`` is None — existing callers are unchanged.
    """
    if adain_stats not in ("mask", "all"):
        raise ValueError(
            f'unknown adain_stats: {adain_stats!r} (expected "mask" or "all")'
        )
    n = adapter.num_blocks()
    lo, hi = (0, n) if blend_layers is None else blend_layers
    adain_alpha = float(adain_alpha)
    if adain_last_layers is None or adain_alpha == 0.0:
        adain_n = 0
    else:
        raw = int(adain_last_layers)
        adain_n = n if raw < 0 else max(0, min(raw, n))
    adain_lo = n - adain_n if adain_n > 0 else n
    debug = bool(int(os.environ.get(debug_env, "0"))) and rank() == 0
    if debug:
        m_dbg = mask_tokens.float()
        print(
            f"      [blend-dbg] mask: sum={float(m_dbg.sum()):.1f} "
            f"mean={float(m_dbg.mean()):.4f} min={float(m_dbg.min()):.3f} "
            f"max={float(m_dbg.max()):.3f} n={m_dbg.numel()} | "
            f"blend layers [{lo},{hi})",
            flush=True,
        )
    for i in range(n):
        if log_blocks and (i % log_blocks == 0):
            print(f"      block {i}/{n}", flush=True)
        state_gen = adapter.run_block(i, state_gen)
        state_copy = adapter.run_block(i, state_copy)

        if not (lo <= i < hi):
            continue

        n_g = adapter.get_noise_tokens(state_gen)   # (1, n_img, D)
        n_c = adapter.get_noise_tokens(state_copy)
        m = mask_tokens.to(dtype=n_g.dtype, device=n_g.device).view(1, -1, 1)

        if divergence_gate is not None:
            gate = divergence_gate(i, step, n_g, n_c, m)
            if gate is not None:
                m = m * gate
        if mask_perturb is not None:
            m = mask_perturb(i, step, m)

        if i >= adain_lo:
            stats_m = m if adain_stats == "mask" else torch.ones_like(m)
            mu_g, std_g = _masked_mean_std(n_g, stats_m)
            mu_c, std_c = _masked_mean_std(n_c, stats_m)
            n_c_adain = (n_c - mu_c) / std_c * std_g + mu_g
            n_c = n_c * (1.0 - adain_alpha) + n_c_adain * adain_alpha

        n_new = n_c * m + n_g * (1.0 - m)

        if debug and (i % max(1, (hi - lo) // 4) == 0 or i == hi - 1):
            diff = (n_c.float() - n_g.float()).abs()
            mm = m.float()
            in_mask = float((diff * mm).sum() / mm.sum().clamp_min(1)) if float(mm.sum()) > 0 else 0.0
            out_mask = float((diff * (1 - mm)).sum() / (1 - mm).sum().clamp_min(1))
            changed = float((n_new.float() - n_g.float()).abs().mean())
            print(
                f"      [blend-dbg] layer {i:>2}: |copy-gen| in_mask={in_mask:.4f} "
                f"out_mask={out_mask:.4f} | applied|new-gen|={changed:.4f}",
                flush=True,
            )

        state_gen = adapter.set_noise_tokens(state_gen, n_new)
        if not copy_private:
            state_copy = adapter.set_noise_tokens(state_copy, n_new)

    return adapter.finalize_noise_pred(state_gen)
