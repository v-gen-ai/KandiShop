"""Open-vocabulary object segmentation + placement, the object-swap counterpart
to ``face_detection`` / ``face_alignment`` / ``face_parsing_mask``.

The dual-stream method does not know it is looking at a face. It needs three
things and nothing else: WHERE the thing is in the target, a version of the
reference warped into that place, and a mask saying which stream wins where.
For faces those come from a landmark detector, a 5-point similarity fit and
BiSeNet. None of that generalises to a mug, so this module supplies the same
three answers for an arbitrary object named by TEXT:

    where   -- Grounding DINO turns "a mug." into boxes, SAM turns a box into a
               silhouette; the object is summarised as an :class:`ObjectInfo`
               (bbox, mask, centroid, characteristic size).
    warp    -- :func:`align_object_to_target` places the reference object,
               either by a transform fitted to real correspondences
               (:mod:`object_alignment`) or, failing that, at the target's
               centroid scaled by the size ratio times a user factor.
    mask    -- :func:`object_blend_mask` unions the two silhouettes.

Three things are deliberately different from the face path.

Correspondences, not landmarks. A similarity fit needs corresponding points,
and an object has no canonical ones, so they are found by appearance in
:mod:`object_alignment` -- which also decides whether they support a rotation
and a perspective tilt or nothing at all. When they support nothing (the two
photographs show different sides of the thing), the placement degrades to
scale + translate and the reference keeps the orientation it was shot in.

No size prior. Faces have one: the inter-ocular distance is the same physical
length on everybody, so matching it matches the head. Objects do not -- nothing
in two photographs says whether the incoming mug is bigger than the outgoing
one, and no segmentation can recover it. So the ratio is the caller's to state:
``size_ratio=1`` means "make them the same size", ``2`` means "the new object is
twice as big", ``0.5`` half. What "size" means is set by ``size_metric``
(default: sqrt of the mask area, so equal size == equal area, which behaves
sanely when the two classes have different aspect ratios).

Union, not intersection. The face mask intersects the two face regions because
the reference's head, aligned on inter-ocular distance, is often the larger of
the two and the union would grow the head. Here the mask must cover the old
object (it has to disappear) AND the new one (it has to appear), so the union is
the only correct choice -- with the same consequence the hair union has on the
face path: in "old minus new" the copy stream has nothing to show and falls back
to whatever the background canvas holds.

Weights are pulled from the Hub on first use (~700 MB + ~2.4 GB) and both nets
are lazily built singletons, so importing this module costs nothing.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
import torch
from PIL import Image

DEFAULT_DETECTOR = "IDEA-Research/grounding-dino-base"
DEFAULT_SEGMENTER = "facebook/sam-vit-huge"

SizeMetric = Literal["sqrt_area", "bbox_diag", "bbox_max", "bbox_width", "bbox_height"]
SelectMode = Literal["score", "area"]
AlignRung = Literal["matcher", "centroid"]


@dataclass
class ObjectInfo:
    """A single segmented object, in its own image's pixel coordinates.

    Mirrors :class:`face_detection.FaceInfo` (``center`` / ``size`` / ``bbox``)
    so the placement math reads the same, but carries the silhouette as well:
    unlike a face, an object has no canonical shape to fall back on, so the mask
    IS the geometry.
    """

    bbox: np.ndarray          # [x1, y1, x2, y2] in pixels
    mask: np.ndarray          # (H, W) float32 in {0, 1}
    score: float
    label: str
    center: np.ndarray        # (2,) centroid of the mask (x, y)
    size: float               # characteristic size, see size_metric
    size_metric: str = "sqrt_area"
    area: float = 0.0
    prompt: str = field(default="", repr=False)

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


def _characteristic_size(mask: np.ndarray, bbox: np.ndarray,
                         metric: SizeMetric) -> float:
    """One number standing in for "how big is this thing".

    ``sqrt_area`` (default) is the side of the square of equal area. It is the
    one that survives a class change: a book and a mug photographed to look
    "equally big" agree on area far better than on either bbox dimension, since
    those swing with aspect ratio alone. ``bbox_diag`` / ``bbox_max`` ignore the
    silhouette and are there for thin or hollow objects, where the filled area
    understates the visual footprint. ``bbox_width`` makes the incoming object
    exactly as wide as the outgoing one, whatever its height: the swap for an
    object held in a hand, where the grip fixes the width. ``bbox_height`` is
    the same for height: the new object covers the old one top to bottom.
    """
    w = float(bbox[2] - bbox[0])
    h = float(bbox[3] - bbox[1])
    if metric == "bbox_diag":
        return max(math.hypot(w, h), 1e-3)
    if metric == "bbox_max":
        return max(w, h, 1e-3)
    if metric == "bbox_width":
        return max(w, 1e-3)
    if metric == "bbox_height":
        return max(h, 1e-3)
    return max(math.sqrt(float(mask.sum())), 1e-3)


def _mask_centroid(mask: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Centre of mass of the silhouette, with the bbox centre as fallback.

    The centroid, not the bbox centre: it is where the object's *mass* is, so a
    mug with a handle sticking out to one side lands on its body rather than
    half a handle-width off, and it degrades gracefully when the two objects
    have different shapes.
    """
    total = float(mask.sum())
    if total < 1.0:
        return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
                        dtype=np.float64)
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()], dtype=np.float64)


def _restore_dino_box_head(det, detector: str) -> None:
    """Reload the shared box head from the checkpoint into every copy of it.

    The checkpoint stores the head once, as ``model.decoder.bbox_embed.0``, and
    the model ties the other eleven copies to it. transformers 5.x ties them the
    other way round and overwrites the checkpoint values with the fresh init of
    the top-level copy -- without reporting a missing key -- which leaves the
    boxes right but the scores collapsed (0.76 -> 0.11 on a plain tube). On 4.x
    this rewrites the values that are already there.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    prefix = "model.decoder.bbox_embed.0."
    with safe_open(hf_hub_download(detector, "model.safetensors"), "pt") as fh:
        head = {k[len(prefix):]: fh.get_tensor(k) for k in fh.keys()
                if k.startswith(prefix)}
    if not head:
        return
    with torch.no_grad():
        for copies in (det.bbox_embed, det.model.decoder.bbox_embed):
            for mod in copies:
                for name, p in mod.named_parameters():
                    p.copy_(head[name].to(p.device, p.dtype))


def _patch_dino_text_inputs(det) -> None:
    """Make transformers 5.x feed the text encoder what 4.x fed it.

    Two regressions, both in the phrase masking around BERT, and together worth
    a 0.76 -> 0.11 drop in score on a single-word prompt:

    * the vectorised ``generate_masks_with_special_tokens_and_transfer_map``
      counts positions from the closing delimiter itself, so the '.' ending a
      phrase gets position 0 instead of the phrase length;
    * the phrase mask reaches BERT as a 4D bool tensor, which the eager
      attention adds to the scores as +1/+0 rather than 0/-inf, so the phrase
      blocks are not masked at all;
    * ``encode_sinusoidal_position_embedding`` casts its output to the input
      dtype, and the text positions are integer ids, so every sin/cos of the
      text position embedding is truncated to an integer.

    On 4.x the mask arrives 3D and the hook leaves it alone, the position
    rewrite produces what the 4.x function already returns, and the sinusoid
    function does not exist.
    """
    import transformers.models.grounding_dino.modeling_grounding_dino as gd

    sine = getattr(gd, "encode_sinusoidal_position_embedding", None)
    if sine is not None and not getattr(sine, "_k6_float", False):
        def sine_float(pos_tensor, *args, **kwargs):
            if not pos_tensor.is_floating_point():
                pos_tensor = pos_tensor.float()
            return sine(pos_tensor, *args, **kwargs)

        sine_float._k6_float = True
        gd.encode_sinusoidal_position_embedding = sine_float

    if not getattr(gd, "_k6_v4_positions", False):
        original = gd.generate_masks_with_special_tokens_and_transfer_map

        def masks_v4_positions(input_ids):
            attention_mask, _ = original(input_ids)
            n = input_ids.shape[1]
            dev = input_ids.device
            special = torch.isin(input_ids, torch.tensor(gd.SPECIAL_TOKENS, device=dev))
            idx = torch.arange(n, device=dev).unsqueeze(0).expand_as(input_ids)
            prev = torch.where(special, idx, torch.full_like(idx, -1)).cummax(1)[0]
            prev_strict = torch.cat([torch.full_like(prev[:, :1], -1), prev[:, :-1]], 1)
            nxt = torch.where(special, idx, torch.full_like(idx, n))
            nxt = torch.flip(torch.cummin(torch.flip(nxt, [1]), 1)[0], [1])
            valid = (nxt != 0) & (nxt != n - 1) & (nxt != n)
            pos = torch.where(valid, idx - prev_strict - 1, torch.zeros_like(idx))
            return attention_mask, pos.clamp(min=0).long()

        gd.generate_masks_with_special_tokens_and_transfer_map = masks_v4_positions
        gd._k6_v4_positions = True

    def additive_mask(_mod, args):
        if len(args) < 2 or not torch.is_tensor(args[1]):
            return None
        mask = args[1]
        if mask.dtype != torch.bool or mask.ndim != 4:
            return None
        dtype = next(_mod.parameters()).dtype
        add = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
        add = add.masked_fill(~mask, torch.finfo(dtype).min)
        return (args[0], add) + tuple(args[2:])

    det.model.text_backbone.register_forward_pre_hook(additive_mask)


class ObjectSegmenter:
    """Grounding DINO (text -> boxes) + SAM (box -> silhouette).

    Split in two on purpose. An open-vocabulary detector is what lets the class
    be a string the caller types, but its output is a box, and a box is far too
    coarse a mask -- pasting a rectangle would hand the model the target's
    background around the object as something to replace. SAM has the opposite
    profile: no notion of classes at all, but given a box it returns the object's
    outline. The pair covers both halves and neither needs training here.
    """

    def __init__(self, detector: str = DEFAULT_DETECTOR,
                 segmenter: str = DEFAULT_SEGMENTER,
                 device: str = "cuda", dtype: torch.dtype = torch.float32):
        from transformers import (AutoProcessor, GroundingDinoForObjectDetection,
                                  SamModel, SamProcessor)

        self.device = device
        self.dtype = dtype
        # Eager attention, explicitly. The TP runners replace
        # ``transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS`` with a table of
        # Nones so the generator's own attention path is used, and any other
        # transformers model in the process that looks an implementation up in
        # that table gets None back and dies calling it. Eager never consults it.
        self.det_processor = AutoProcessor.from_pretrained(detector)
        self.det = GroundingDinoForObjectDetection.from_pretrained(
            detector, torch_dtype=dtype, attn_implementation="eager",
        ).to(device).eval()
        _restore_dino_box_head(self.det, detector)
        _patch_dino_text_inputs(self.det)
        self.sam_processor = SamProcessor.from_pretrained(segmenter)
        self.sam = SamModel.from_pretrained(
            segmenter, torch_dtype=dtype, attn_implementation="eager",
        ).to(device).eval()

    @staticmethod
    def _normalise_prompt(prompt: str) -> str:
        """Grounding DINO was trained on lowercase phrases terminated by '.';
        anything else quietly costs recall."""
        text = prompt.strip().lower()
        if not text.endswith("."):
            text += "."
        return text

    @torch.no_grad()
    def _boxes(self, pil: Image.Image, prompt: str, *,
               box_threshold: float, text_threshold: float):
        text = self._normalise_prompt(prompt)
        inputs = self.det_processor(images=pil, text=text, return_tensors="pt")
        inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                  for k, v in inputs.items()}
        outputs = self.det(**inputs)
        post = self.det_processor.post_process_grounded_object_detection
        kwargs = dict(
            input_ids=inputs["input_ids"],
            text_threshold=text_threshold,
            target_sizes=[(pil.height, pil.width)],
        )
        try:
            res = post(outputs, threshold=box_threshold, **kwargs)[0]
        except TypeError:
            # Older transformers spell the box threshold `box_threshold`.
            res = post(outputs, box_threshold=box_threshold, **kwargs)[0]
        labels = res.get("text_labels", res.get("labels", []))
        boxes = res["boxes"].detach().float().cpu().numpy()
        scores = res["scores"].detach().float().cpu().numpy()
        labels = [str(l) for l in labels]
        return boxes, scores, labels

    @torch.no_grad()
    def _mask_from_box(self, pil: Image.Image, box: np.ndarray) -> np.ndarray:
        inputs = self.sam_processor(
            pil, input_boxes=[[[float(v) for v in box]]], return_tensors="pt",
        )
        inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                  for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        out = self.sam(**inputs, multimask_output=True)
        masks = self.sam_processor.image_processor.post_process_masks(
            out.pred_masks.float().cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0][0]                                   # (3, H, W) bool
        # SAM answers a box with three nested readings (part / subpart / whole);
        # its own predicted IoU is the intended way to pick between them.
        best = int(out.iou_scores[0, 0].float().argmax().item())
        return masks[best].numpy().astype(np.float32)

    @torch.no_grad()
    def detect(self, pil: Image.Image, prompt: str, *,
               box_threshold: float = 0.30, text_threshold: float = 0.25,
               size_metric: SizeMetric = "sqrt_area",
               max_objects: int = 5) -> list[ObjectInfo]:
        """Every instance of *prompt* in *pil*, best score first."""
        pil = pil.convert("RGB")
        boxes, scores, labels = self._boxes(
            pil, prompt, box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        order = np.argsort(-scores)[:max_objects]
        found = []
        for i in order:
            box = boxes[i]
            mask = self._mask_from_box(pil, box)
            found.append(ObjectInfo(
                bbox=box.astype(np.float64),
                mask=mask,
                score=float(scores[i]),
                label=labels[i] if i < len(labels) else prompt,
                center=_mask_centroid(mask, box),
                size=_characteristic_size(mask, box, size_metric),
                size_metric=size_metric,
                area=float(mask.sum()),
                prompt=prompt,
            ))
        return found

    def segment_primary(self, pil: Image.Image, prompt: str, *,
                        select: SelectMode = "score",
                        **kwargs) -> ObjectInfo | None:
        """The one object the swap acts on, or ``None`` if the class is absent.

        ``select="area"`` takes the largest instance instead of the most
        confident one, which is what you want when a photo contains several of
        the same class and the subject is simply the big one in front.
        """
        found = self.detect(pil, prompt, **kwargs)
        if not found:
            return None
        if select == "area":
            return max(found, key=lambda o: o.area)
        return found[0]


_segmenter: ObjectSegmenter | None = None
_segmenter_key: tuple | None = None


def get_object_segmenter(detector: str = DEFAULT_DETECTOR,
                         segmenter: str = DEFAULT_SEGMENTER,
                         device: str = "cuda") -> ObjectSegmenter:
    """Lazily built singleton, rebuilt only if the checkpoints change.

    Falls back to CPU if the GPU cannot take the extra ~3 GB: under tensor
    parallelism the generator already owns most of the card, and a slow mask on
    rank 0 is better than an OOM that takes the whole run down with it.
    """
    global _segmenter, _segmenter_key
    device = os.environ.get("K6_OBJSEG_DEVICE", device)
    key = (detector, segmenter, device)
    if _segmenter is None or _segmenter_key != key:
        try:
            _segmenter = ObjectSegmenter(detector, segmenter, device=device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            _segmenter = ObjectSegmenter(detector, segmenter, device="cpu")
            key = (detector, segmenter, "cpu")
        _segmenter_key = key
    return _segmenter


def _anchor(obj: ObjectInfo, place_center: str) -> tuple[float, float]:
    """The point of ``obj`` that lands on the other object's: the mask centroid,
    the bbox centre (``"bbox"``) or the middle of the bbox's bottom edge
    (``"bottom"``, so a smaller object still stands where the old one stood)."""
    if place_center == "bbox":
        return (float(obj.bbox[0] + obj.bbox[2]) / 2.0, float(obj.bbox[1] + obj.bbox[3]) / 2.0)
    if place_center == "bottom":
        return float(obj.bbox[0] + obj.bbox[2]) / 2.0, float(obj.bbox[3])
    return float(obj.center[0]), float(obj.center[1])


def compute_object_transform(source_obj: ObjectInfo, target_obj: ObjectInfo,
                             size_ratio: float = 1.0,
                             place_center: str = "centroid") -> tuple[np.ndarray, float]:
    """2x3 affine mapping *source* image coords onto the target object's place.

    Scale + translate, no rotation (see the module docstring). ``size_ratio``
    is the user's size prior: the placed object comes out ``size_ratio`` times
    the size of the one it replaces. ``place_center`` picks the points that are
    made to coincide (see :func:`_anchor`).
    """
    scale = float(size_ratio) * float(target_obj.size) / float(source_obj.size)
    cx, cy = _anchor(source_obj, place_center)
    dx, dy = _anchor(target_obj, place_center)
    M = np.array([[scale, 0.0, dx - scale * cx], [0.0, scale, dy - scale * cy]], dtype=np.float64)
    return M, scale


def align_object_to_target(
    source: Image.Image,
    target_obj: ObjectInfo,
    source_obj: ObjectInfo,
    output_size: tuple[int, int],
    *,
    background: Image.Image | None = None,
    size_ratio: float = 1.0,
    scale_clamp: tuple[float, float] = (0.02, 50.0),
    isolate: bool = True,
    isolate_feather_frac: float = 0.01,
    matrix: np.ndarray | None = None,
    place_center: str = "centroid",
) -> tuple[Image.Image, np.ndarray, float]:
    """Place the reference object where the target's object is.

    Returns ``(aligned_pil, warped_source_mask, scale)``; the warped mask is in
    the output frame, so it can be unioned with the target's directly.

    ``matrix`` is a 3x3 from :mod:`object_alignment`, fitted to actual
    correspondences. Without one the placement falls back to the
    correspondence-free rule this module started with: equal areas, equal
    centroids, no rotation.

    ``isolate`` (default) cuts the reference down to its own silhouette before
    warping, leaving the rest of the canvas as ``background``. Without it the
    copy stream is handed the reference PHOTO -- and since the blend takes that
    stream inside the mask, wherever the mask reaches past the new object (the
    "old minus new" ring that the union always creates) the reference's own
    background is what gets reproduced there. Isolating puts flat canvas in that
    ring instead: still wrong, but wrong in a way that is visible and local
    rather than a plausible-looking piece of the wrong photograph.
    """
    if matrix is None:
        M, scale = compute_object_transform(source_obj, target_obj, size_ratio, place_center)
        if not (scale_clamp[0] <= scale <= scale_clamp[1]):
            clamped = float(max(scale_clamp[0], min(scale, scale_clamp[1])))
            (cx, cy), (dx, dy) = _anchor(source_obj, place_center), _anchor(target_obj, place_center)
            M[0, 0] = M[1, 1] = clamped
            M[0, 2] = dx - clamped * cx
            M[1, 2] = dy - clamped * cy
            scale = clamped
        H = np.vstack([M, [0.0, 0.0, 1.0]])
    else:
        H = np.asarray(matrix, dtype=np.float64)
        scale = float(math.hypot(H[0, 0], H[0, 1]))

    w, h = output_size
    src_rgb = np.array(source.convert("RGB"))

    if background is not None:
        canvas = np.array(background.convert("RGB").resize((w, h)))
    else:
        canvas = np.full((h, w, 3), 128, dtype=np.uint8)

    warped_mask = cv2.warpPerspective(
        source_obj.mask, H, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    warped_rgb = cv2.warpPerspective(
        src_rgb, H, (w, h), flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )

    if isolate:
        alpha = np.clip(warped_mask, 0.0, 1.0)
        if isolate_feather_frac > 0:
            # A hard cut leaves a stair-stepped rim that the model happily
            # reproduces as an outline around the pasted object.
            sigma = max(0.5, isolate_feather_frac * float(target_obj.size))
            alpha = cv2.GaussianBlur(alpha, ksize=(0, 0),
                                     sigmaX=sigma, sigmaY=sigma)
        a = alpha[:, :, None].astype(np.float32)
        out = a * warped_rgb.astype(np.float32) + (1.0 - a) * canvas.astype(np.float32)
        aligned = np.clip(out, 0, 255).astype(np.uint8)
    else:
        aligned = warped_rgb

    return Image.fromarray(aligned), np.clip(warped_mask, 0.0, 1.0), scale


@dataclass
class ObjectSwapSpec:
    """Everything the object path needs, in one object.

    Gathered rather than spread over a dozen keyword arguments because the swap
    is set up in three places (the crop box in the refine pass, the placement,
    the mask) and they must agree; passing one spec around is what keeps them
    from drifting apart, the same reason ``mask_kwargs`` exists on the face path.
    """

    target_class: str                       # what is being replaced, in the scene
    source_class: str | None = None         # what replaces it (defaults to same)
    size_ratio: float = 1.0                 # 1 == same size, 2 == twice as big
    size_metric: SizeMetric = "sqrt_area"
    select: SelectMode = "score"
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    isolate: bool = True
    dilate_frac: float = 0.04
    blur_sigma_frac: float = 0.03
    align: AlignRung = "matcher"            # "matcher" | "centroid"
    allow_homography: bool = True
    # 1.0 leaves parts with no matched features entirely to the scene. Anything
    # lower keeps a fraction of the reference there, which shows up as a ghost
    # rim on caps and lids: they are off the fitted plane, so they land wrong.
    conf_weight: float = 1.0                # how much inlier density shapes the mask
    # Keypoints cluster on the printing, so the gaps to cover inside one label
    # and the distance to the nearest unprinted part (a cap) are the same order.
    # The plateau covers the gaps; the falloff has to be short enough to die
    # before the cap, which on the Milk Bobbles jar starts ~0.17 sizes out.
    conf_radius_frac: float = 0.06          # full weight this far from an inlier
    conf_sigma_frac: float = 0.04           # and how fast it dies past that
    crop_auto: bool = True                  # size the refine crop from the fit scale
    # Well over 1, for two reasons. The crop is sized from the silhouette ratio
    # while what matters is the scale the matcher ends up with, and the two
    # differ by several percent. And the pass repaints rather than copies, so
    # canvas pixels per glyph keep buying legibility past the point where the
    # reference has any detail left to give: on Move Free, 1.36x is visibly
    # crisper in the fine print than 1.08x off the same reference.
    crop_target_scale: float = 1.35
    crop_min: float = 1.1
    crop_max: float = 2.4
    # "centroid" sits the crop and the placed reference on the object's mass;
    # "bbox" on its box, which a swap needs: on a wide body under a narrow cap
    # the centroid is low enough to cut the cap off the crop, and matching it
    # with a uniform object's centroid shifts the new one down.
    crop_center: str = "centroid"
    place_center: str | None = None         # placement anchor; None = crop_center
    detector: str = DEFAULT_DETECTOR
    segmenter: str = DEFAULT_SEGMENTER

    @property
    def source_prompt(self) -> str:
        return self.source_class or self.target_class

    def find(self, pil: Image.Image, prompt: str) -> ObjectInfo | None:
        return get_object_segmenter(self.detector, self.segmenter).segment_primary(
            pil, prompt, select=self.select, box_threshold=self.box_threshold,
            text_threshold=self.text_threshold, size_metric=self.size_metric,
        )

    def auto_crop_scale(self, source_pil: Image.Image, target_obj: ObjectInfo,
                        refine_size: int, fallback: float) -> tuple[float, str | None]:
        """Crop multiplier that lands the reference in the pass magnified.

        The refine crop is the only magnification available -- the canvas is
        pinned at ``refine_size`` by the model's token grid -- so a crop that is
        too wide shrinks the reference on its way in and throws away exactly the
        fine print the pass exists to restore. Given the object's size here and
        in the reference, the crop side that lands the reference at
        ``crop_target_scale`` is ``size_here * refine_size /
        (size_in_reference * crop_target_scale)``, and this returns it as a
        multiple of the object's longest side, clamped to keep some scene around
        the object and to stay inside anything sane.
        """
        if not self.crop_auto:
            return fallback, None
        source_obj = self.find(source_pil, self.source_prompt)
        if source_obj is None or source_obj.size <= 0:
            return fallback, "reference object not found; keeping the given crop"
        side = (float(target_obj.size) * float(refine_size)
                / (float(source_obj.size) * float(self.crop_target_scale)))
        want = side / max(float(target_obj.width), float(target_obj.height), 1.0)
        got = min(max(want, self.crop_min), self.crop_max)
        note = (f"auto crop {got:.2f} (wanted {want:.2f}) from object "
                f"{target_obj.size:.0f}px here vs {source_obj.size:.0f}px in the "
                f"reference")
        return got, note

    def plan(self, target_pil: Image.Image, source_pil: Image.Image,
             output_size: tuple[int, int], *, background=None):
        """The two entry points the pipeline calls -- ``find`` for the refine
        crop and ``plan`` for the placement + mask -- are the whole interface a
        swap mode has to implement."""
        return build_object_swap(target_pil, source_pil, self, output_size,
                                 background=background)


@dataclass
class ObjectSwapPlan:
    """Result of :func:`build_object_swap`: the placed reference and the mask."""

    aligned: Image.Image
    mask: np.ndarray
    target_obj: ObjectInfo
    source_obj: ObjectInfo
    scale: float
    fit: object | None = None       # AlignFit when the matcher placed it


def build_object_swap(
    target_pil: Image.Image,
    source_pil: Image.Image,
    spec: ObjectSwapSpec,
    output_size: tuple[int, int],
    *,
    background: Image.Image | None = None,
) -> ObjectSwapPlan | None:
    """Segment both objects, place the reference, union the silhouettes.

    Returns ``None`` if either class is absent, which the caller treats exactly
    like a failed face detection: no blending, single stream, image unchanged.
    """
    target_obj = spec.find(target_pil, spec.target_class)
    if target_obj is None:
        return None
    source_obj = spec.find(source_pil, spec.source_prompt)
    if source_obj is None:
        return None

    # The size prior and the matcher answer the same question, so they cannot
    # both be in charge: asking for a different size means asking for a
    # placement the correspondences do not support.
    fit = None
    if spec.align == "matcher" and float(spec.size_ratio) == 1.0:
        from .object_alignment import fit_object_transform

        fit = fit_object_transform(
            source_pil, target_pil, source_obj, target_obj, output_size,
            allow_homography=spec.allow_homography, verbose=True,
        )
    elif spec.align == "matcher":
        print(f"    [align] size_ratio={spec.size_ratio} overrides the matcher; "
              f"placing by centroid", flush=True)

    aligned, warped_mask, scale = align_object_to_target(
        source_pil, target_obj, source_obj, output_size,
        background=background, size_ratio=spec.size_ratio, isolate=spec.isolate,
        matrix=fit.matrix if fit is not None else None,
        place_center=spec.place_center or spec.crop_center,
    )
    # The target's silhouette was measured on target_pil, which the caller has
    # already sized to the output frame; resize defensively so the union cannot
    # be one pixel off when it has not.
    w, h = output_size
    target_mask = target_obj.mask
    if target_mask.shape != (h, w):
        target_mask = cv2.resize(target_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # A matcher fit means the two silhouettes are the same object seen twice, so
    # their disagreement is fit error rather than a shape to cover; a centroid
    # placement or a size override means a genuine swap, which needs the union.
    refining = fit is not None and fit.rung != "centroid"
    mask = object_blend_mask(
        target_mask, warped_mask, size=target_obj.size,
        dilate_frac=spec.dilate_frac, blur_sigma_frac=spec.blur_sigma_frac,
        combine="intersect" if refining else "union",
    )

    # Only when the matcher actually fitted something: a centroid placement has
    # no inliers, and with no evidence anywhere the weighting would delete the
    # mask instead of shaping it. That guard is also what keeps a genuine swap
    # (a mug replaced by a book, no correspondences by construction) on the full
    # union it needs to erase the old object.
    if refining and spec.conf_weight > 0:
        from .object_alignment import confidence_map

        sigma = spec.conf_sigma_frac * float(target_obj.size)
        radius = spec.conf_radius_frac * float(target_obj.size)
        conf = confidence_map(fit.inlier_xy, output_size, sigma=sigma,
                              radius=radius)
        cw = float(spec.conf_weight)
        mask = mask * ((1.0 - cw) + cw * conf)
        print(f"    [align] confidence weighting w={cw} radius={radius:.1f}px "
              f"sigma={sigma:.1f}px over {len(fit.inlier_xy)} inliers -> mask "
              f"mean {mask.mean():.3f}", flush=True)

    return ObjectSwapPlan(aligned=aligned, mask=mask, target_obj=target_obj,
                          source_obj=source_obj, scale=scale, fit=fit)


def object_blend_mask(
    target_mask: np.ndarray,
    warped_source_mask: np.ndarray,
    *,
    size: float,
    dilate_frac: float = 0.04,
    blur_sigma_frac: float = 0.03,
    combine: str = "union",
) -> np.ndarray:
    """The two silhouettes combined, dilated then feathered. Values in [0, 1].

    ``union`` is what a swap needs: the old object has to be covered wherever it
    sticks out past the new one, or its edge survives the blend. ``intersect``
    is what a refine of the same object needs: the two silhouettes are the same
    object and any disagreement between them is fit error, so painting the
    difference puts reference pixels on background -- a hard-edged bulge exactly
    where the fit is least trustworthy.

    Both radii are fractions of the target object's characteristic size, so the
    mask behaves the same on a mug filling the frame and on one across the room.
    Dilation happens before the blur for the reason spelled out in
    ``face_parsing_mask.blend_mask``: a Gaussian on a binary mask leaves 0.5 on
    the old boundary and so pulls the edge inward, and dilating by roughly the
    blur radius first puts it back where the segmentation drew it.
    """
    join = np.minimum if combine == "intersect" else np.maximum
    mask = join(
        np.clip(target_mask.astype(np.float32), 0.0, 1.0),
        np.clip(warped_source_mask.astype(np.float32), 0.0, 1.0),
    )
    if dilate_frac > 0:
        radius = int(round(float(size) * float(dilate_frac)))
        if radius > 0:
            k = 2 * radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask, kernel)
    if blur_sigma_frac > 0:
        sigma = max(0.5, float(size) * float(blur_sigma_frac))
        mask = cv2.GaussianBlur(mask, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(mask, 0.0, 1.0)
