"""Head masks from BiSeNet face parsing, for swapping a whole head (face + hair).

The refine masks in :mod:`utils.face_mask` are ellipses sized from the eyes:
they never reach the hair, so a refine keeps the photo's hairstyle. A head swap
has to replace it, and an ellipse cannot follow a hairline, so here the mask is
a parsed silhouette instead. Two ways to combine the target's head with the
aligned reference's (the reference is already warped into the target's frame,
so the maps share coordinates):

* ``union`` -- everything of both heads: the old hair goes and the new one fits
  wherever it reaches.
* ``parsing`` -- face parts intersected (keeps the photo's head size and drops
  occluders that are not face on the target), hair unioned.

Both are dilated, then blurred once, by fractions of the target's inter-ocular
distance (IOD): the token grid the mask is consumed on is 16px at 1024, so a
fixed-pixel feather would vanish on a downscale.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms

from .bisenet import BiSeNet
from .compositing import square_crop_box
from .face_detection import FaceInfo, detect_face_robust

DEFAULT_PARSING_CKPT = os.environ.get(
    "FACE_PARSING_CKPT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "weights", "face_parsing_79999_iter.pth"),
)

# CelebAMask-HQ ids: skin, brows, eyes, glasses, ears, nose, mouth, lips.
# Glasses (6) matter: on a face wearing them BiSeNet emits no eye classes at
# all, so without 6 the mask has holes where the eyes are.
FACE_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13)
HAIR_ID = 17

MASK_MODES = ("union", "parsing")


def combine_head_swap_masks(face_t, hair_t, face_r, hair_r, include_hair=True):
    """``parsing`` mode: face parts intersect, hair unions."""
    mask = np.minimum(face_t, face_r)
    if include_hair:
        mask = np.maximum(mask, np.maximum(hair_t, hair_r))
    return mask


def _dilate_blur(mask: np.ndarray, face: FaceInfo | None,
                 dilate_frac: float, blur_sigma_frac: float) -> np.ndarray:
    """Dilate, then blur, both IOD-relative. A blur on a binary mask leaves 0.5
    on the old boundary and so pulls the edge inward; dilating by about the blur
    radius first puts the 0.5 contour back where the parser drew it."""
    if face is not None and dilate_frac > 0:
        radius = int(round(float(face.size) * float(dilate_frac)))
        if radius > 0:
            k = 2 * radius + 1
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if face is not None and blur_sigma_frac > 0:
        mask = gaussian_filter(mask, sigma=max(0.5, float(face.size) * float(blur_sigma_frac)))
    return np.clip(mask, 0.0, 1.0)


class HeadMaskBuilder:
    """BiSeNet on a square crop around the face, pasted back into the frame.

    The network is trained on 512x512 aligned face crops: a whole scene arrives
    stretched to 512x512 with the face too small to parse, while a square crop
    of ``max(bbox) * crop_scale`` keeps the aspect and puts the face at a size
    the parser knows.
    """

    def __init__(self, model_path: str = DEFAULT_PARSING_CKPT, device: str = "cuda"):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"face-parsing checkpoint not found: {model_path} "
                "(set FACE_PARSING_CKPT or pass --parsing-ckpt)")
        self.device = device
        self.net = BiSeNet(n_classes=19)
        self.net.load_state_dict(torch.load(model_path, map_location=device))
        self.net.to(device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def parse(self, image: Image.Image) -> np.ndarray:
        """19-class map at ``image``'s own resolution. Nearest on the way back:
        class ids cannot be interpolated."""
        w, h = image.size
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        parsing = self.net(x)[0].squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)
        return cv2.resize(parsing, (w, h), interpolation=cv2.INTER_NEAREST)

    def _crop_box(self, image: Image.Image, face: FaceInfo | None, crop_scale: float):
        if face is None or crop_scale <= 0:
            return None
        side = max(face.width, face.height) * float(crop_scale)
        return square_crop_box(float(face.center[0]), float(face.center[1]), side, *image.size)

    def parse_frame(self, image: Image.Image, box) -> np.ndarray:
        """Parse ``box`` (whole frame if None); outside it is background (0)."""
        image = image.convert("RGB")
        if box is None:
            return self.parse(image)
        x0, y0, x1, y1 = (int(v) for v in box)
        parsing = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        parsing[y0:y1, x0:x1] = self.parse(image.crop((x0, y0, x1, y1)))
        return parsing

    def head_mask(self, image: Image.Image, include_hair: bool = True,
                  crop_scale: float = 2.4) -> np.ndarray:
        """Hard 0/1 head of ``image``, parsed around its own face."""
        box = self._crop_box(image, detect_face_robust(image), crop_scale)
        ids = FACE_IDS + ((HAIR_ID,) if include_hair else ())
        return np.isin(self.parse_frame(image, box), ids).astype(np.float32)

    def union_mask(self, target: Image.Image, aligned_ref: Image.Image,
                   include_hair: bool = True, crop_scale: float = 2.4,
                   blur_sigma_frac: float = 0.15, dilate_frac: float = 0.06) -> np.ndarray:
        """Target head OR aligned reference head, each parsed around its own face."""
        mask = np.maximum(self.head_mask(target, include_hair, crop_scale),
                          self.head_mask(aligned_ref, include_hair, crop_scale))
        return _dilate_blur(mask, detect_face_robust(target), dilate_frac, blur_sigma_frac)

    def parsing_mask(self, target: Image.Image, aligned_ref: Image.Image,
                     include_hair: bool = True, crop_scale: float = 2.4,
                     blur_sigma_frac: float = 0.15, dilate_frac: float = 0.06) -> np.ndarray:
        """Faces intersected, hair unioned; both parsed on the target's crop box."""
        face = detect_face_robust(target)
        box = self._crop_box(target, face, crop_scale)
        maps = []
        for image in (target, aligned_ref):
            p = self.parse_frame(image, box)
            maps += [np.isin(p, FACE_IDS).astype(np.float32), (p == HAIR_ID).astype(np.float32)]
        mask = combine_head_swap_masks(*maps, include_hair=include_hair)
        return _dilate_blur(mask, face, dilate_frac, blur_sigma_frac)

    def build(self, mode: str, target: Image.Image, aligned_ref: Image.Image, **kw) -> np.ndarray:
        if mode == "union":
            return self.union_mask(target, aligned_ref, **kw)
        if mode == "parsing":
            return self.parsing_mask(target, aligned_ref, **kw)
        raise ValueError(f"mask mode {mode!r}: expected one of {MASK_MODES}")
