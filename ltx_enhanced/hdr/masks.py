"""Highlight masks and the source-preservation composite (owner A23 spec 4.5, 4.6, 4.9; design: HDR_ARCHITECTURE 6b).

No ComfyUI import. IMAGE tensors are [B, H, W, C] (C >= 3), MASK tensors [B, H, W], float. The model never sees a
mask: these run after generation and decide which generated pixels reach the final image.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from . import colorspace

CHANNEL_POLICIES = ("any", "all", "luma")
USER_MODES = ("add", "subtract", "replace")
_LUMA = (0.2126, 0.7152, 0.0722)        # Rec.709 weights on the display-encoded values (a ranking, not a colour metric)


def luma(images: torch.Tensor) -> torch.Tensor:
    w = torch.tensor(_LUMA, dtype=torch.float32, device=images.device)
    return (images[..., :3].float() * w).sum(-1)


# --- mask operators -------------------------------------------------------------------------------------------------
def grow(mask: torch.Tensor, px: int) -> torch.Tensor:
    """Signed grow: px > 0 dilates (max filter), px < 0 erodes (min filter), 0 = unchanged."""
    px = int(px)
    if px == 0:
        return mask
    k = 2 * abs(px) + 1
    m = mask.float().unsqueeze(1)
    if px > 0:
        out = F.max_pool2d(m, k, stride=1, padding=abs(px))
    else:
        out = -F.max_pool2d(-m, k, stride=1, padding=abs(px))
    return out.squeeze(1).clamp(0.0, 1.0)


def feather(mask: torch.Tensor, px: int) -> torch.Tensor:
    """Box blur of radius px (two passes = a soft tent edge); replicate padding keeps the frame border unchanged."""
    px = int(px)
    if px <= 0:
        return mask
    m = mask.float().unsqueeze(1)
    k = 2 * px + 1
    for _ in range(2):
        m = F.avg_pool2d(F.pad(m, (px, px, px, px), mode="replicate"), k, stride=1)
    return m.squeeze(1).clamp(0.0, 1.0)


def temporal_smooth(mask: torch.Tensor, frames: int) -> torch.Tensor:
    """Centred mean over `frames` frames (odd; 1 = off); the clip ends use the frames that exist."""
    n = int(frames)
    if n <= 1 or mask.shape[0] <= 1:
        return mask
    half = n // 2
    m = mask.float()
    c = torch.cumsum(F.pad(m, (0, 0, 0, 0, 1, 0)), dim=0)       # c[i] = sum of frames < i
    B = m.shape[0]
    idx = torch.arange(B, device=m.device)
    lo = (idx - half).clamp(min=0)
    hi = (idx + half + 1).clamp(max=B)
    return ((c[hi] - c[lo]) / (hi - lo).view(-1, 1, 1).float()).clamp(0.0, 1.0)


def apply_user(mask: torch.Tensor, user, mode: str) -> torch.Tensor:
    if user is None:
        return mask
    if mode not in USER_MODES:
        raise ValueError(f"user mask mode {mode!r}: use one of {USER_MODES}.")
    u = match_frames(user.float(), mask.shape[0], "user mask")
    if u.shape[-2:] != mask.shape[-2:]:
        u = F.interpolate(u.unsqueeze(1), size=mask.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    if mode == "add":
        return torch.maximum(mask, u).clamp(0, 1)
    if mode == "subtract":
        return (mask * (1.0 - u)).clamp(0, 1)
    return u.clamp(0, 1)


def match_frames(t: torch.Tensor, frames: int, what: str) -> torch.Tensor:
    """A single frame is used for every frame; otherwise the counts must agree (never silently stretched)."""
    if t.shape[0] == frames:
        return t
    if t.shape[0] == 1:
        return t.expand(frames, *t.shape[1:])
    raise ValueError(f"The {what} has {t.shape[0]} frames but {frames} are needed; use one frame or the same count.")


# --- 4.5 highlight analyzer -----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HighlightSettings:
    hard_clip: float = 0.998
    near_white: float = 0.92
    channel_policy: str = "any"
    expand_px: int = 0
    feather_px: int = 0
    temporal_frames: int = 1


def clip_mask(images: torch.Tensor, hard_clip: float, policy: str) -> torch.Tensor:
    """1 where the display values sit at the code ceiling (clipped); a bright but unclipped pixel is never included."""
    if policy not in CHANNEL_POLICIES:
        raise ValueError(f"channel_policy {policy!r}: use one of {CHANNEL_POLICIES}.")
    rgb = images[..., :3].float()
    if policy == "any":
        return (rgb >= hard_clip).any(-1).float()
    if policy == "all":
        return (rgb >= hard_clip).all(-1).float()
    return (luma(rgb) >= hard_clip).float()


def near_white_map(images: torch.Tensor, near_white: float, hard_clip: float) -> torch.Tensor:
    """Soft 0..1 ramp from `near_white` to `hard_clip` on luma (display values)."""
    lo, hi = float(near_white), max(float(hard_clip), float(near_white) + 1e-6)
    return ((luma(images) - lo) / (hi - lo)).clamp(0.0, 1.0)


def analyze_highlights(images: torch.Tensor, s: HighlightSettings, user=None, user_mode: str = "add"):
    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError(f"Highlight Analyzer expects IMAGE [frames, H, W, 3]; got {tuple(images.shape)}.")
    raw_clip = clip_mask(images, s.hard_clip, s.channel_policy)
    nw = near_white_map(images, s.near_white, s.hard_clip)
    clip = temporal_smooth(feather(grow(raw_clip, s.expand_px), s.feather_px), s.temporal_frames)
    clip = apply_user(clip, user, user_mode)
    hmap = torch.maximum(nw, raw_clip)
    change = [0.0] + [float((clip[i] - clip[i - 1]).abs().mean()) for i in range(1, clip.shape[0])]
    report = {
        "settings": asdict(s), "user_mask": None if user is None else user_mode,
        "frames": int(images.shape[0]),
        "clipped_fraction": [round(float(v), 6) for v in raw_clip.flatten(1).mean(1)],
        "near_white_fraction": [round(float(v), 6) for v in (nw > 0).float().flatten(1).mean(1)],
        "frames_with_clipping": int((raw_clip.flatten(1).amax(1) > 0).sum()),
        "mask_change": [round(v, 6) for v in change],
    }
    return clip, hmap, report


# --- 4.6 synth mask -------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SynthSettings:
    use_clip: bool = True
    include_near_white: bool = False
    near_white_weight: float = 1.0
    grow_px: int = 4
    feather_px: int = 6
    temporal_frames: int = 3


def synth_mask(clip, highlight_map, s: SynthSettings, user=None, user_mode: str = "add"):
    if clip is None and highlight_map is None and user is None:
        raise ValueError("Synth Mask needs a clip mask, a highlight map or a user mask.")
    ref = clip if clip is not None else (highlight_map if highlight_map is not None else user)
    m = torch.zeros_like(ref.float())
    if s.use_clip and clip is not None:
        m = torch.maximum(m, clip.float())
    if s.include_near_white and highlight_map is not None:
        m = torch.maximum(m, (highlight_map.float() * float(s.near_white_weight)).clamp(0, 1))
    m = temporal_smooth(feather(grow(m, s.grow_px), s.feather_px), s.temporal_frames)
    m = apply_user(m, user, user_mode)
    cov = m.flatten(1).mean(1)
    flick = [0.0] + [float((m[i] - m[i - 1]).abs().mean()) for i in range(1, m.shape[0])]
    qc = {"settings": asdict(s), "coverage": [round(float(v), 6) for v in cov],
          "mean_coverage": round(float(cov.mean()), 6), "flicker": [round(v, 6) for v in flick],
          "max_flicker": round(max(flick), 6)}
    return m, qc


def overlay(images: torch.Tensor, mask: torch.Tensor, strength: float = 0.6) -> torch.Tensor:
    """Display preview: the source with the mask tinted red."""
    rgb = images[..., :3].float().clamp(0, 1)
    m = match_frames(mask.float(), rgb.shape[0], "mask").unsqueeze(-1) * float(strength)
    red = torch.tensor((1.0, 0.1, 0.1), dtype=rgb.dtype, device=rgb.device)
    return rgb * (1 - m) + red * m


# --- 4.9 source-preservation composite ------------------------------------------------------------------------------
TRANSFERS = ("acescct", "acescg")
COLOR_MATCH = ("off", "per_channel_gain")


def to_linear(images: torch.Tensor, transfer: str) -> torch.Tensor:
    if transfer == "acescct":
        return colorspace.acescct_decode(images[..., :3])
    if transfer == "acescg":
        return images[..., :3].float()
    raise ValueError(f"transfer {transfer!r}: use one of {TRANSFERS}.")


def from_linear(lin: torch.Tensor, transfer: str) -> torch.Tensor:
    return colorspace.acescct_encode(lin) if transfer == "acescct" else lin


def estimate_gain(src_lin, gen_lin, mask, trusted_max_mask: float = 0.05, floor: float = 0.005):
    """Per-channel median of gen / src over trusted pixels (mask low, source not crushed, not near the top)."""
    m = mask.float().unsqueeze(-1).expand_as(src_lin)
    ok = (m < trusted_max_mask) & (src_lin > floor) & (src_lin < 0.9) & (gen_lin > 0)
    gains, counts = [], []
    for c in range(3):
        sel = ok[..., c]
        n = int(sel.sum())
        counts.append(n)
        if n < 64:
            gains.append(1.0)
            continue
        gains.append(float((gen_lin[..., c][sel] / src_lin[..., c][sel]).median()))
    return gains, counts


@dataclass(frozen=True)
class CompositeSettings:
    synth_mix: float = 1.0
    color_match: str = "per_channel_gain"
    source_transfer: str = "acescct"
    generated_transfer: str = "acescct"


def composite(source, generated, synth, s: CompositeSettings, shadow=None, custom=None):
    if source.shape[:3] != generated.shape[:3]:
        raise ValueError(f"Source {tuple(source.shape[:3])} and generated {tuple(generated.shape[:3])} differ in "
                         "frames or size: feed the same window (same plan, same index) at the same resolution.")
    if s.color_match not in COLOR_MATCH:
        raise ValueError(f"color_match {s.color_match!r}: use one of {COLOR_MATCH}.")
    B = source.shape[0]
    m = match_frames(synth.float(), B, "synth mask")
    if m.shape[-2:] != source.shape[1:3]:
        raise ValueError(f"The synth mask is {tuple(m.shape[-2:])} but the frames are {tuple(source.shape[1:3])}; "
                         "build it from the resized source window.")
    for extra in (shadow, custom):
        if extra is not None:
            m = torch.maximum(m, match_frames(extra.float(), B, "extra mask"))
    m = (m * float(s.synth_mix)).clamp(0.0, 1.0)
    src = to_linear(source, s.source_transfer)
    gen = to_linear(generated, s.generated_transfer)
    gains, counts = [1.0, 1.0, 1.0], [0, 0, 0]
    if s.color_match == "per_channel_gain":
        gains, counts = estimate_gain(src, gen, m)
        gen = gen / torch.tensor(gains, dtype=gen.dtype, device=gen.device).clamp(min=1e-6)
    m3 = m.unsqueeze(-1)
    final = src * (1.0 - m3) + gen * m3
    src_part, gen_part = src * (1.0 - m3), gen * m3
    keep = m3.expand_as(final) == 0
    in_gen = m3.expand_as(final) > 0.5
    out = from_linear(final, s.generated_transfer)
    back = to_linear(out, s.generated_transfer)
    report = {
        "settings": asdict(s), "frames": int(B), "working_space": "ACEScg (AP1 linear)",
        "gains": [round(g, 6) for g in gains], "gain_samples": counts,
        "mask_coverage": round(float(m.mean()), 6),
        "preserved_max_abs_diff_working": float((final - src).abs()[keep].max()) if keep.any() else 0.0,
        "preserved_max_abs_diff_after_encode": float((back - src).abs()[keep].max()) if keep.any() else 0.0,
        "generated_region_mean_abs_diff": float((final - src).abs()[in_gen].mean()) if in_gen.any() else 0.0,
        "note": ("preserved = exactly the source in the working space where the mask is 0; across encodings the "
                 "difference is float round-off, not bit-identical"),
    }
    alpha = source[..., 3:] if source.shape[-1] > 3 else None
    if alpha is not None:
        out = torch.cat([out, alpha.to(out.dtype)], dim=-1)
    return out, m, src_part, gen_part, report


def report_json(report: dict) -> str:
    return json.dumps(report, indent=1)
