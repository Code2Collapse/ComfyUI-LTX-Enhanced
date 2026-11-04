"""8-bit cleanup before HDR reconstruction (owner A23 spec 4.3). Independent design, no ComfyUI import.

Banding in an 8-bit source is the loss of the values between code steps. This fills them in while keeping every
pixel inside its own source quantisation cell, so re-quantising the result gives back exactly the source codes:

    repeat n times:  y = smooth(y);  y = clamp(y, x - t * h, x + t * h)        (h = half a code = 0.5 / 255)

``smooth`` is a separable box blur (wider in the shadows by ``shadow_sensitivity``); the clamp is the projection onto
the set of images that quantise to the source. Edges survive because a blur across an edge leaves the cell and is
clamped back. ``t`` (tolerance, default 0.98) stays just under half a code so rounding ties cannot flip a code.
Only an 8-bit source is changed; 10-bit, float and unknown sources pass through unchanged (spec: never modify them).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

CODE = 1.0 / 255.0


@dataclass(frozen=True)
class CleanupSettings:
    enabled: bool = True
    strength: float = 1.0           # 0 = off, 1 = the full projected result (a linear blend in between)
    radius: int = 16                # box-blur radius in pixels (measured: 16 with 8 passes; docs/evidence/L9.132)
    iterations: int = 8
    shadow_sensitivity: float = 0.5  # extra blur width in the shadows (0 = none)
    tolerance: float = 0.98         # fraction of half a code the result may move


def _box(x: torch.Tensor, r: int) -> torch.Tensor:
    """Separable box blur on [B, C, H, W] with replicate padding."""
    if r <= 0:
        return x
    k = 2 * r + 1
    x = F.avg_pool2d(F.pad(x, (r, r, 0, 0), mode="replicate"), (1, k), stride=1)
    return F.avg_pool2d(F.pad(x, (0, 0, r, r), mode="replicate"), (k, 1), stride=1)


def _clean_chunk(x: torch.Tensor, s: CleanupSettings) -> torch.Tensor:
    """x: [B, H, W, C] display values that sit on the 8-bit grid."""
    t = x.permute(0, 3, 1, 2).float()
    lo = (t - s.tolerance * 0.5 * CODE).clamp(0.0, 1.0)
    hi = (t + s.tolerance * 0.5 * CODE).clamp(0.0, 1.0)
    w = None
    if s.shadow_sensitivity > 0:
        luma = (0.2126 * t[:, 0:1] + 0.7152 * t[:, 1:2] + 0.0722 * t[:, 2:3])
        w = ((0.15 - luma) / 0.15).clamp(0.0, 1.0) * min(1.0, float(s.shadow_sensitivity))
    y = t
    for _ in range(max(1, int(s.iterations))):
        sm = _box(y, s.radius)
        if w is not None:
            sm = sm * (1 - w) + _box(y, 2 * s.radius) * w
        y = torch.minimum(torch.maximum(sm, lo), hi)
    y = t + (y - t) * float(max(0.0, min(1.0, s.strength)))
    return y.permute(0, 2, 3, 1)


def on_8bit_grid(images: torch.Tensor, sample: int = 65536) -> bool:
    """True when the values sit on k/255 (within float round-off)."""
    v = images[..., :3].reshape(-1)
    if v.numel() > sample:
        v = v[torch.linspace(0, v.numel() - 1, sample).long()]
    return bool(((v * 255.0 - (v * 255.0).round()).abs() < 1e-3).all())


def cleanup(images: torch.Tensor, bit_depth: str, s: CleanupSettings, chunk: int = 8):
    """Returns (prepared frames, report). Frame count, size and alpha are kept; only RGB of an 8-bit source moves."""
    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError(f"8-bit Cleanup expects IMAGE [frames, H, W, 3]; got {tuple(images.shape)}.")
    rep = {"settings": asdict(s), "bit_depth": str(bit_depth), "frames": int(images.shape[0])}
    if not s.enabled or s.strength <= 0:
        return images, {**rep, "applied": False, "reason": "disabled"}
    if str(bit_depth).strip().lower() not in ("8", "8-bit", "8bit"):
        return images, {**rep, "applied": False,
                        "reason": f"source bit depth is {bit_depth!r}: only 8-bit sources are changed"}
    if not on_8bit_grid(images):
        return images, {**rep, "applied": False,
                        "reason": "the values are not on the 8-bit grid (already processed or not 8-bit); unchanged"}
    rgb = images[..., :3]
    out = torch.empty_like(rgb, dtype=torch.float32)
    for i in range(0, rgb.shape[0], chunk):
        out[i:i + chunk] = _clean_chunk(rgb[i:i + chunk], s)
    codes_in = (rgb.float() * 255.0).round()
    codes_out = (out * 255.0).round()
    dev = (out - rgb.float()).abs() / CODE
    rep.update({"applied": True,
                "requantizes_to_source": bool(torch.equal(codes_in, codes_out)),
                "max_shift_codes": round(float(dev.max()), 4),
                "mean_shift_codes": round(float(dev.mean()), 4),
                "changed_fraction": round(float((dev > 1e-6).float().mean()), 6)})
    res = out.to(images.dtype)
    if images.shape[-1] > 3:
        res = torch.cat([res, images[..., 3:]], dim=-1)
    return res, rep
