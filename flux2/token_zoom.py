"""Give the face a full token budget inside every block, without a second pass.

The refinement pass exists because a face in a full-length frame owns a few
dozen tokens, and no amount of conditioning buys detail the model has nowhere
to put. It buys those tokens by leaving the frame: crop, upscale, denoise, paste
back -- which is why the seam, the lighting mismatch and the ghost hair are
things we keep fighting.

This does the same trade inside the transformer. At every block the face tokens
are lifted out of the shared latent, resampled up to a full 64x64 grid -- the
token count the model sees for a whole 1024px image, so the operation lands in
its training regime rather than outside it -- and run through *the same block*
alongside the frame, with a prompt naming the person and the reference as the
image condition. The result is resampled back down and written into the crop's
place. The frame never leaves the frame; only the resolution the face is
processed at changes, and it changes at every layer instead of once.

The zoomed stream keeps its own text and condition tokens across blocks, since
those evolve layer by layer like any others; its image tokens are re-derived
from the shared latent each block, so it carries no history of its own and
cannot drift away from the frame it is refining.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ZoomPlan:
    """Where the face sits on the token grid, and how far it is blown up.

    The scale is an integer, and that is the point: with ``factor`` copies of
    each token going up and an average pool of the same size coming back, the
    round trip is the identity. Interpolating instead makes it a low-pass
    filter, and one that runs on every block, so the crop dissolves whether or
    not the zoomed stream contributes anything.
    """
    box: tuple[int, int, int, int]   # (x0, y0, x1, y1) in token coordinates
    token_h: int
    token_w: int
    factor: int = 1

    @property
    def box_hw(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.box
        return y1 - y0, x1 - x0

    @property
    def grid_hw(self) -> tuple[int, int]:
        bh, bw = self.box_hw
        return bh * self.factor, bw * self.factor


def face_token_box(
    face, *, width: int, height: int, token_h: int, token_w: int,
    crop_scale: float = 2.4, size: int = 64,
) -> ZoomPlan:
    """The refinement crop, expressed on the token grid.

    Same square box the pixel-space refinement uses (``crop_scale`` times the
    face's longer side, clipped to the frame), rounded outwards to whole tokens
    so nothing of the face is cut by the quantisation.
    """
    px_per_token_x = width / token_w
    px_per_token_y = height / token_h
    side = max(face.width, face.height) * crop_scale
    side = min(side, width, height)
    cx, cy = float(face.center[0]), float(face.center[1])

    x0 = int((cx - side / 2) // px_per_token_x)
    y0 = int((cy - side / 2) // px_per_token_y)
    x1 = int(-(-(cx + side / 2) // px_per_token_x))
    y1 = int(-(-(cy + side / 2) // px_per_token_y))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(token_w, max(x1, x0 + 1)), min(token_h, max(y1, y0 + 1))
    factor = max(1, size // max(y1 - y0, x1 - x0))
    return ZoomPlan(box=(x0, y0, x1, y1), token_h=token_h, token_w=token_w, factor=factor)


def _to_grid(tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
    b, n, c = tokens.shape
    return tokens.transpose(1, 2).reshape(b, c, h, w)


def _to_tokens(grid: torch.Tensor) -> torch.Tensor:
    b, c, h, w = grid.shape
    return grid.reshape(b, c, h * w).transpose(1, 2)


def crop_and_upsample(tokens: torch.Tensor, plan: ZoomPlan) -> torch.Tensor:
    """Face tokens of the frame, each one repeated ``factor`` times per side."""
    x0, y0, x1, y1 = plan.box
    grid = _to_grid(tokens, plan.token_h, plan.token_w)[:, :, y0:y1, x0:x1]
    k = plan.factor
    up = grid.repeat_interleave(k, dim=2).repeat_interleave(k, dim=3)
    return _to_tokens(up)


def downsample_and_paste(
    tokens: torch.Tensor, zoom_tokens: torch.Tensor, plan: ZoomPlan,
) -> torch.Tensor:
    """Average each ``factor x factor`` block back to one token and write it in."""
    x0, y0, x1, y1 = plan.box
    gh, gw = plan.grid_hw
    grid = _to_grid(zoom_tokens.float(), gh, gw)
    down = F.avg_pool2d(grid, kernel_size=plan.factor).to(tokens.dtype)
    full = _to_grid(tokens, plan.token_h, plan.token_w).clone()
    full[:, :, y0:y1, x0:x1] = down
    return _to_tokens(full)


@torch.no_grad()
def dit_token_zoom_velocity(
    adapter,
    state_main,
    state_zoom,
    plan: ZoomPlan,
    *,
    layers: tuple[int, int] | None = None,
    bypass: bool = False,
    dump: list | None = None,
) -> torch.Tensor:
    """Run the transformer block by block with the face zoomed at every layer.

    ``layers`` restricts *where the zoomed result is written back*; the zoomed
    stream itself runs everywhere regardless, because its text and condition
    tokens have to reach block *i* through the same *i* blocks the frame did.
    """
    n_blocks = adapter.num_blocks()
    lo, hi = layers if layers is not None else (0, n_blocks)

    for idx in range(n_blocks):
        frame_tokens = adapter.get_noise_tokens(state_main)
        adapter.set_noise_tokens(state_zoom, crop_and_upsample(frame_tokens, plan))

        state_main = adapter.run_block(idx, state_main)
        if not bypass:
            state_zoom = adapter.run_block(idx, state_zoom)

        if lo <= idx < hi:
            # ``bypass`` pastes the frame's own tokens back through the same
            # resample round trip: the control that says whether an artefact
            # comes from the zoomed block or from the resampling around it.
            out = adapter.get_noise_tokens(state_main)
            zoomed = (crop_and_upsample(out, plan) if bypass
                      else adapter.get_noise_tokens(state_zoom))
            merged = downsample_and_paste(out, zoomed, plan)
            adapter.set_noise_tokens(state_main, merged)
            if dump is not None:
                dump.append(idx)

    return adapter.finalize_noise_pred(state_main)
