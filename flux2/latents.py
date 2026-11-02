"""FLUX.2 latent helpers (encode / decode packed latents, dimension rules).

Self-contained port of the pieces of ``flux2_face_pipeline.py`` the method
needs, so ``publish/`` has no dependency on the research repo.
"""

from __future__ import annotations

import math

import torch
from PIL import Image

FLUX2_MAX_INPUT_AREA = 1024 * 1024  # matches Flux2Pipeline.__call__


def compute_flux2_dimensions(
    pil_image: Image.Image, max_area: int = FLUX2_MAX_INPUT_AREA, multiple_of: int = 16,
) -> tuple[int, int]:
    """Mirror the resize/rounding rules inside ``Flux2Pipeline.__call__``.
    Returns ``(width, height)`` rounded down to multiples of ``multiple_of``."""
    width, height = pil_image.size
    if width * height > max_area:
        scale = math.sqrt(max_area / (width * height))
        width = int(width * scale)
        height = int(height * scale)
    width = (width // multiple_of) * multiple_of
    height = (height // multiple_of) * multiple_of
    return width, height


def _get_bn_stats(pipeline, dtype: torch.dtype, device: torch.device):
    bn = pipeline.vae.bn
    eps = pipeline.vae.config.batch_norm_eps
    mean = bn.running_mean.view(1, -1, 1, 1).to(device=device, dtype=dtype)
    std = torch.sqrt(bn.running_var.view(1, -1, 1, 1) + eps).to(device=device, dtype=dtype)
    return mean, std


def decode_packed_x0_flux2(pipeline, x0_packed: torch.Tensor, latent_ids: torch.Tensor) -> Image.Image:
    """Packed x0 -> unpack -> denormalize -> unpatchify -> VAE decode -> PIL."""
    device = pipeline._execution_device
    vae_dtype = pipeline.vae.dtype
    x0 = x0_packed.to(device=device, dtype=vae_dtype)
    ids = latent_ids.to(device=device)
    lat = pipeline._unpack_latents_with_ids(x0, ids)
    bn_mean, bn_std = _get_bn_stats(pipeline, dtype=lat.dtype, device=lat.device)
    lat = lat * bn_std + bn_mean
    lat = pipeline._unpatchify_latents(lat)
    pixels = pipeline.vae.decode(lat, return_dict=False)[0]
    return pipeline.image_processor.postprocess(pixels, output_type="pil")[0]


def encode_pil_to_packed_latents_flux2(
    pipeline, pil_image: Image.Image, width: int, height: int,
    device: torch.device, dtype: torch.dtype, generator,
) -> torch.Tensor:
    """PIL -> preprocess -> VAE encode -> patchify+normalize -> pack -> (1, seq, C*4)."""
    img_tensor = pipeline.image_processor.preprocess(
        pil_image, height=height, width=width, resize_mode="crop",
    ).to(device=device, dtype=dtype)
    latent = pipeline._encode_vae_image(img_tensor, generator=generator)
    return pipeline._pack_latents(latent)
