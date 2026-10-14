"""Pad frames to the model's aligned canvas and crop them back exactly.

Design: docs/design/architecture.md ("Padding rules"). Never stretch, never crop silently: the source is placed in
an aligned canvas, the offsets are recorded, and ``crop_back`` removes exactly what was added, in space and time.
Core's empty latent floors sizes and its VAE encode centre-crops (docs/re-notes/core-ltx-geometry-and-schedule.md);
padding first avoids both.

IMAGE tensors are [F, H, W, C] floats in 0..1 (ComfyUI layout); MASK tensors are [F, H, W].
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .geometry import LTXV, LatentGeometry

PAD_MODES = ("reflect", "replicate", "constant")
FITS = ("pad", "fit-inside", "crop-to-fill")
PLACEMENTS = ("center", "top-left")

# Delivered frame size -> name. The sampled canvas is the aligned size (DCP 2K 1920x1088, DCP 4K 4096x2176).
TARGETS = {
    "DCP 2K (1920x1080)": (1920, 1080),
    "DCP 4K (4096x2160)": (4096, 2160),
    "UHD (3840x2160)": (3840, 2160),
    "source size": None,
    "custom": None,
}


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class PadPlan:
    src_w: int
    src_h: int
    src_frames: int
    fit: str
    scale: float            # resize applied to the source before placement (1.0 for fit="pad")
    scaled_w: int
    scaled_h: int
    crop: Box               # part of the scaled source that is kept (whole source unless fit="crop-to-fill")
    deliver_w: int
    deliver_h: int
    sample_w: int
    sample_h: int
    sample_frames: int
    content: Box            # where the kept source sits in the sampled canvas
    delivery: Box           # where the delivered frame sits in the sampled canvas
    mode: str
    value: float
    placement: str

    @property
    def frames_after(self) -> int:
        return self.sample_frames - self.src_frames

    @property
    def pad_left(self) -> int:
        return self.content.x

    @property
    def pad_top(self) -> int:
        return self.content.y

    @property
    def pad_right(self) -> int:
        return self.sample_w - self.content.x - self.content.w

    @property
    def pad_bottom(self) -> int:
        return self.sample_h - self.content.y - self.content.h

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(frames_after=self.frames_after, pad_left=self.pad_left, pad_top=self.pad_top,
                 pad_right=self.pad_right, pad_bottom=self.pad_bottom)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PadPlan":
        keep = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        for k in ("crop", "content", "delivery"):
            if isinstance(keep.get(k), dict):
                keep[k] = Box(**keep[k])
        return cls(**keep)


def _place(outer: int, inner: int, placement: str) -> int:
    return (outer - inner) // 2 if placement == "center" else 0


def make_plan(src_w: int, src_h: int, src_frames: int, deliver_w: int | None = None, deliver_h: int | None = None,
              fit: str = "pad", mode: str = "reflect", value: float = 0.0, placement: str = "center",
              geometry: LatentGeometry = LTXV) -> PadPlan:
    """Plan how a ``src_w`` x ``src_h`` x ``src_frames`` clip goes into the model's aligned canvas.

    ``deliver_w`` / ``deliver_h`` is the frame size that is delivered after crop-back (None = the source size).
    The sampled canvas is that size rounded up to the geometry's alignment, and the frame count is rounded up to
    the next 8k+1.
    """
    if fit not in FITS:
        raise ValueError(f"fit must be one of {', '.join(FITS)}; got {fit!r}")
    if mode not in PAD_MODES:
        raise ValueError(f"pad mode must be one of {', '.join(PAD_MODES)}; got {mode!r}")
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {', '.join(PLACEMENTS)}; got {placement!r}")
    if min(src_w, src_h, src_frames) < 1:
        raise ValueError(f"the source must have a size and at least one frame, got {src_w}x{src_h}x{src_frames}")
    deliver_w = int(deliver_w or src_w)
    deliver_h = int(deliver_h or src_h)

    if fit == "pad":
        if src_w > deliver_w or src_h > deliver_h:
            raise ValueError(
                f"The source ({src_w}x{src_h}) is larger than the target frame ({deliver_w}x{deliver_h}). "
                "Choose fit 'fit-inside' to scale it down with its aspect ratio kept, or 'crop-to-fill' to crop it, "
                "or pick a larger target.")
        scale = 1.0
    elif fit == "fit-inside":
        scale = min(deliver_w / src_w, deliver_h / src_h)
    else:
        scale = max(deliver_w / src_w, deliver_h / src_h)
    scaled_w = src_w if scale == 1.0 else max(1, round(src_w * scale))
    scaled_h = src_h if scale == 1.0 else max(1, round(src_h * scale))
    if fit == "fit-inside":     # rounding must never push the scaled source past the frame
        scaled_w, scaled_h = min(scaled_w, deliver_w), min(scaled_h, deliver_h)
    if fit == "crop-to-fill":
        scaled_w, scaled_h = max(scaled_w, deliver_w), max(scaled_h, deliver_h)
        crop = Box(_place(scaled_w, deliver_w, placement), _place(scaled_h, deliver_h, placement), deliver_w, deliver_h)
    else:
        crop = Box(0, 0, scaled_w, scaled_h)

    sample_w, sample_h = geometry.aligned_size(deliver_w), geometry.aligned_size(deliver_h)
    delivery = Box(_place(sample_w, deliver_w, placement), _place(sample_h, deliver_h, placement), deliver_w, deliver_h)
    content = Box(delivery.x + _place(deliver_w, crop.w, placement), delivery.y + _place(deliver_h, crop.h, placement),
                  crop.w, crop.h)
    return PadPlan(src_w=int(src_w), src_h=int(src_h), src_frames=int(src_frames), fit=fit, scale=float(scale),
                   scaled_w=scaled_w, scaled_h=scaled_h, crop=crop, deliver_w=deliver_w, deliver_h=deliver_h,
                   sample_w=sample_w, sample_h=sample_h, sample_frames=geometry.aligned_frames(int(src_frames)),
                   content=content, delivery=delivery, mode=mode, value=float(value), placement=placement)


def _index_map(n: int, before: int, after: int, mode: str, device) -> torch.Tensor:
    """Source index for every output position of an axis padded by ``before`` / ``after`` (any pad size)."""
    idx = torch.arange(-before, n + after, device=device)
    if mode == "replicate" or n == 1:
        return idx.clamp(0, n - 1)
    period = 2 * (n - 1)                       # reflect without repeating the edge sample, for any pad length
    m = idx.remainder(period)
    return torch.where(m < n, m, period - m)


def _pad_spatial(x: torch.Tensor, plan: PadPlan, h_axis: int, w_axis: int, mode: str, value: float) -> torch.Tensor:
    top, bottom, left, right = plan.pad_top, plan.pad_bottom, plan.pad_left, plan.pad_right
    if mode == "constant":
        shape = list(x.shape)
        shape[h_axis], shape[w_axis] = plan.sample_h, plan.sample_w
        out = torch.full(shape, float(value), dtype=x.dtype, device=x.device)
        sl = [slice(None)] * x.ndim
        sl[h_axis], sl[w_axis] = slice(top, top + x.shape[h_axis]), slice(left, left + x.shape[w_axis])
        out[tuple(sl)] = x
        return out
    hi = _index_map(x.shape[h_axis], top, bottom, mode, x.device)
    wi = _index_map(x.shape[w_axis], left, right, mode, x.device)
    return x.index_select(h_axis, hi).index_select(w_axis, wi)


def _resize(x: torch.Tensor, w: int, h: int, channels_last: bool) -> torch.Tensor:
    """Resize [F,H,W,C] (channels_last) or [F,H,W] to w x h; bicubic with antialias when shrinking."""
    if x.shape[-2 if not channels_last else -3] == h and x.shape[-1 if not channels_last else -2] == w:
        return x
    t = x.movedim(-1, 1) if channels_last else x.unsqueeze(1)
    shrink = h < t.shape[-2] or w < t.shape[-1]
    t = F.interpolate(t.float(), size=(h, w), mode="bicubic", align_corners=False, antialias=shrink)
    t = t.clamp(0.0, 1.0).to(x.dtype)
    return t.movedim(1, -1) if channels_last else t.squeeze(1)


def _check_source(x: torch.Tensor, plan: PadPlan, what: str, h_axis: int, w_axis: int) -> None:
    if x.shape[0] != plan.src_frames or x.shape[h_axis] != plan.src_h or x.shape[w_axis] != plan.src_w:
        raise ValueError(f"The {what} is {x.shape[w_axis]}x{x.shape[h_axis]} with {x.shape[0]} frames, but the plan "
                         f"was made for {plan.src_w}x{plan.src_h} with {plan.src_frames} frames. Make the plan from "
                         f"the same {what}.")


def pad_images(images: torch.Tensor, plan: PadPlan) -> torch.Tensor:
    """[F,H,W,C] source frames -> [sample_frames, sample_h, sample_w, C] canvas frames."""
    if images.ndim != 4:
        raise ValueError(f"expected an IMAGE batch [frames, height, width, channels], got shape {tuple(images.shape)}")
    _check_source(images, plan, "image batch", 1, 2)
    x = _resize(images, plan.scaled_w, plan.scaled_h, channels_last=True)
    c = plan.crop
    x = x[:, c.y:c.y + c.h, c.x:c.x + c.w, :]
    x = _pad_spatial(x, plan, 1, 2, plan.mode, plan.value)
    if plan.frames_after:   # repeat the last frame: no reversed motion; dropped again by crop_back
        x = torch.cat([x, x[-1:].expand(plan.frames_after, *x.shape[1:])], dim=0)
    return x


def pad_masks(masks: torch.Tensor, plan: PadPlan, mode: str = "replicate", value: float = 0.0) -> torch.Tensor:
    """[F,H,W] (or [H,W]) masks through the same transform as the frames. Default: replicate the edge."""
    if mode not in PAD_MODES:
        raise ValueError(f"mask pad mode must be one of {', '.join(PAD_MODES)}; got {mode!r}")
    if masks.ndim == 2:
        masks = masks.unsqueeze(0).expand(plan.src_frames, *masks.shape)
    if masks.ndim != 3:
        raise ValueError(f"expected a MASK [frames, height, width], got shape {tuple(masks.shape)}")
    if masks.shape[0] == 1 and plan.src_frames > 1:
        masks = masks.expand(plan.src_frames, *masks.shape[1:])
    _check_source(masks, plan, "mask", 1, 2)
    x = _resize(masks, plan.scaled_w, plan.scaled_h, channels_last=False)
    c = plan.crop
    x = x[:, c.y:c.y + c.h, c.x:c.x + c.w]
    x = _pad_spatial(x, plan, 1, 2, mode, value)
    if plan.frames_after:
        x = torch.cat([x, x[-1:].expand(plan.frames_after, *x.shape[1:])], dim=0)
    return x


def crop_back(images: torch.Tensor, plan: PadPlan, region: str = "content") -> torch.Tensor:
    """Remove the padding. ``region``: 'content' = the source area, 'delivery' = the delivered frame.

    Frames decoded at an integer or fractional multiple of the sampled size (for example after a 2x latent
    upscale) are cropped at the same relative position.
    """
    if region not in ("content", "delivery"):
        raise ValueError(f"region must be 'content' or 'delivery'; got {region!r}")
    if images.ndim != 4:
        raise ValueError(f"expected an IMAGE batch [frames, height, width, channels], got shape {tuple(images.shape)}")
    f, h, w = images.shape[:3]
    rh, rw = h / plan.sample_h, w / plan.sample_w
    if abs(rh - rw) > 1e-6:
        raise ValueError(f"The decoded frames are {w}x{h}, which is not a uniform scale of the planned canvas "
                         f"{plan.sample_w}x{plan.sample_h}. Crop back needs the frames the plan was made for.")
    if f < plan.src_frames:
        raise ValueError(f"The decoded clip has {f} frames, fewer than the source's {plan.src_frames}. "
                         "Something upstream dropped frames.")
    b = plan.content if region == "content" else plan.delivery
    x0, y0 = round(b.x * rw), round(b.y * rh)
    x1, y1 = round((b.x + b.w) * rw), round((b.y + b.h) * rh)
    return images[:plan.src_frames, y0:y1, x0:x1, :]


def summary(plan: PadPlan) -> str:
    lines = [f"source      {plan.src_w}x{plan.src_h}, {plan.src_frames} frames"]
    if plan.fit != "pad":
        lines.append(f"fit         {plan.fit}: scale {plan.scale:.4f} -> {plan.scaled_w}x{plan.scaled_h}"
                     + (f", keep {plan.crop.w}x{plan.crop.h} at ({plan.crop.x},{plan.crop.y})"
                        if plan.fit == "crop-to-fill" else ""))
    lines += [f"delivered   {plan.deliver_w}x{plan.deliver_h}",
              f"sampled     {plan.sample_w}x{plan.sample_h}, {plan.sample_frames} frames "
              f"(+{plan.frames_after} repeated at the end, dropped after decode)",
              f"padding     left {plan.pad_left}, top {plan.pad_top}, right {plan.pad_right}, bottom {plan.pad_bottom} "
              f"px ({plan.mode}{'' if plan.mode != 'constant' else f' {plan.value:g}'}, {plan.placement})"]
    return "\n".join(lines)
