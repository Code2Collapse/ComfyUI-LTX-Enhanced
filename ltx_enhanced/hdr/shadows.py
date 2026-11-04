"""Optional shadow recovery (owner A23 spec 4.4). Independent design, no ComfyUI import.

Where the SDR source is crushed to black, the source has no detail; the generated HDR may have some. Inside a soft
crushed mask the generated shadows are brought to the SOURCE's local exposure and used; everywhere else the result is
exactly the source.

- Crushed mask: luma of the display-encoded source <= ``black_threshold``, feathered by ``transition_px``.
- Local exposure: in a ring of source pixels just above black (``black_threshold`` < luma <= ``ring_factor`` x it),
  gain = blur(source * ring) / blur(generated * ring) over ``local_radius``; where no ring pixel is near, the global
  ring gain is used. Explicit and measured per run (report), no hidden formula.
- Protected levels: inside the mask the result never drops below the source (black stays the floor).
- ``recovered = source * (1 - m) + max(source, generated * gain) * m`` with ``m = strength * mask``, in linear ACEScg.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from . import masks


@dataclass(frozen=True)
class ShadowSettings:
    enabled: bool = False
    black_threshold: float = 0.004   # display luma at or below this = crushed
    strength: float = 1.0
    transition_px: int = 8
    local_radius: int = 32
    ring_factor: float = 4.0


def crushed_mask(display: torch.Tensor, black_threshold: float, transition_px: int) -> torch.Tensor:
    raw = (masks.luma(display) <= float(black_threshold)).float()
    return masks.feather(raw, transition_px) if transition_px > 0 else raw


def _blur_hw(x: torch.Tensor, r: int) -> torch.Tensor:
    """Box blur of [B, H, W, C] over H and W."""
    t = x.permute(0, 3, 1, 2)
    k = 2 * r + 1
    t = F.avg_pool2d(F.pad(t, (r, r, 0, 0), mode="replicate"), (1, k), stride=1)
    t = F.avg_pool2d(F.pad(t, (0, 0, r, r), mode="replicate"), (k, 1), stride=1)
    return t.permute(0, 2, 3, 1)


def local_gain(display, src_lin, gen_lin, s: ShadowSettings):
    lum = masks.luma(display)
    ring = ((lum > s.black_threshold) & (lum <= s.black_threshold * s.ring_factor)).float().unsqueeze(-1)
    n = int(ring.sum())
    eps = 1e-8
    if n == 0:
        return torch.ones_like(gen_lin), {"ring_pixels": 0, "global_gain": [1.0, 1.0, 1.0], "note": "no ring: gain 1"}
    g_src = (src_lin * ring).sum(dim=(0, 1, 2))
    g_gen = (gen_lin * ring).sum(dim=(0, 1, 2)).clamp(min=eps)
    global_gain = (g_src / g_gen).clamp(0.0, 100.0)
    r = max(1, int(s.local_radius))
    num = _blur_hw(src_lin * ring, r)
    den = _blur_hw(gen_lin * ring, r)
    w = _blur_hw(ring.expand_as(gen_lin), r)
    gain = torch.where(w > 1e-4, num / den.clamp(min=eps), global_gain.view(1, 1, 1, 3).expand_as(gen_lin))
    return gain.clamp(0.0, 100.0), {"ring_pixels": n, "global_gain": [round(float(v), 6) for v in global_gain]}


def recover(source, generated, display, s: ShadowSettings, source_transfer="acescct", generated_transfer="acescct"):
    """Returns (recovered image in the SOURCE encoding, mask, report)."""
    if source.shape[:3] != generated.shape[:3] or display.shape[:3] != source.shape[:3]:
        raise ValueError("Shadow Recovery needs the source, generated and display frames of the same window and size.")
    rep = {"settings": asdict(s), "frames": int(source.shape[0])}
    if not s.enabled or s.strength <= 0:
        return source, torch.zeros(source.shape[:3]), {**rep, "applied": False, "reason": "disabled"}
    src = masks.to_linear(source, source_transfer)
    gen = masks.to_linear(generated, generated_transfer)
    m = (crushed_mask(display, s.black_threshold, s.transition_px) * float(min(1.0, s.strength))).clamp(0, 1)
    gain, ginfo = local_gain(display, src, gen, s)
    m3 = m.unsqueeze(-1)
    rec = torch.maximum(src, gen * gain)
    out = src * (1 - m3) + rec * m3
    keep = (m3.expand_as(out) == 0)
    rep.update({"applied": True, **ginfo, "mask_coverage": round(float(m.mean()), 6),
                "outside_mask_max_abs_diff": float((out - src).abs()[keep].max()) if keep.any() else 0.0,
                "inside_never_below_source": bool((out + 1e-7 >= src).all()),
                "mean_lift_in_mask": float((out - src)[m3.expand_as(out) > 0.5].mean()) if (m > 0.5).any() else 0.0})
    res = masks.from_linear(out, source_transfer)
    if source.shape[-1] > 3:
        res = torch.cat([res, source[..., 3:].to(res.dtype)], dim=-1)
    return res, m, rep
