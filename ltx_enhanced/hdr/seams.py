"""Seam exposure match between neighbouring windows (owner A23 spec 4.10). Independent design, no ComfyUI import.

Windows run one per queue. Window k+1 starts ``overlap_head`` frames before its first delivered frame; those frames
were already DELIVERED by window k (EXR on disk). They are two generations of the same source frames, so they measure
the exposure step at the seam directly.

- Measure on the overlap: ``ratio = prev / cur`` (luma: one number; per_channel: three; local: a low-resolution,
  blurred per-channel gain map), excluding protected highlights (linear above ``protect_above``) and near-black.
- Correct window k+1 with ``gain(t) = ratio ** w(t)``, ``w`` = 1 on the overlap and the first delivered frame,
  falling smoothly to 0 over ``seam_width`` frames, so the change is local to the seam (never a whole-window shift).
- Report the overlap mismatch before and after (mean |log2 ratio|, in stops).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from . import masks

MODES = ("luma", "per_channel", "local")
_W = torch.tensor((0.2722287, 0.6740818, 0.0536895))      # AP1 luminance weights (ACEScg Y row)


@dataclass(frozen=True)
class SeamSettings:
    enabled: bool = True
    mode: str = "per_channel"
    seam_width: int = 24
    protect_above: float = 1.0      # linear ACEScg: highlights above this are not used to measure
    floor: float = 0.002
    max_stops: float = 2.0          # refuse to apply a correction larger than this (a real mismatch, not a seam)


def _weights(prev, cur, s: SeamSettings):
    ok = (prev[..., :3] > s.floor) & (cur[..., :3] > s.floor) & (prev[..., :3] < s.protect_above) \
        & (cur[..., :3] < s.protect_above)
    return ok.all(-1, keepdim=True).float()


def measure(prev, cur, s: SeamSettings):
    """prev, cur: [n, H, W, 3] linear ACEScg of the same source frames. Returns a gain tensor broadcastable to cur."""
    w = _weights(prev, cur, s)
    n = float(w.sum())
    if n < 64:
        return None, {"samples": int(n), "note": "too few usable pixels in the overlap"}
    if s.mode == "luma":
        y = _W.to(prev.device)
        pv, cv = (prev[..., :3] * y).sum(-1, keepdim=True), (cur[..., :3] * y).sum(-1, keepdim=True)
        g = (pv * w).sum() / (cv * w).sum().clamp(min=1e-8)
        return g.view(1, 1, 1, 1).expand(1, 1, 1, 3), {"samples": int(n), "gain": [round(float(g), 6)] * 3}
    if s.mode == "per_channel":
        g = (prev[..., :3] * w).sum(dim=(0, 1, 2)) / (cur[..., :3] * w).sum(dim=(0, 1, 2)).clamp(min=1e-8)
        return g.view(1, 1, 1, 3), {"samples": int(n), "gain": [round(float(v), 6) for v in g]}
    if s.mode == "local":
        H, W = prev.shape[1:3]
        gh, gw = max(1, H // 32), max(1, W // 32)
        num = F.adaptive_avg_pool2d((prev[..., :3] * w).sum(0, keepdim=True).permute(0, 3, 1, 2), (gh, gw))
        den = F.adaptive_avg_pool2d((cur[..., :3] * w).sum(0, keepdim=True).permute(0, 3, 1, 2), (gh, gw))
        g = (num / den.clamp(min=1e-8))
        glob = (prev[..., :3] * w).sum(dim=(0, 1, 2)) / (cur[..., :3] * w).sum(dim=(0, 1, 2)).clamp(min=1e-8)
        cov = F.adaptive_avg_pool2d(w.sum(0, keepdim=True).permute(0, 3, 1, 2), (gh, gw))
        g = torch.where(cov > 1e-3, g, glob.view(1, 3, 1, 1))
        g = F.interpolate(g, size=(H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        return g, {"samples": int(n), "gain_global": [round(float(v), 6) for v in glob], "grid": [gh, gw]}
    raise ValueError(f"mode {s.mode!r}: use one of {MODES}.")


def mismatch_stops(prev, cur, s: SeamSettings) -> float:
    w = _weights(prev, cur, s)
    if float(w.sum()) < 1:
        return 0.0
    r = (prev[..., :3].clamp(min=1e-8) / cur[..., :3].clamp(min=1e-8)).log2().abs()
    return float((r * w).sum() / (w.sum() * 3))


def ramp(n_frames: int, start: int, width: int) -> torch.Tensor:
    """w(t): 1 for t <= start, smoothstep down to 0 over `width` frames after `start`."""
    t = torch.arange(n_frames, dtype=torch.float32)
    if width <= 0:
        return (t <= start).float()
    x = ((t - start) / float(width)).clamp(0.0, 1.0)
    return 1.0 - (x * x * (3 - 2 * x))


def match(cur_window, prev_overlap, overlap_n: int, keep_offset: int, s: SeamSettings, transfer: str = "acescct"):
    """cur_window: the whole window (encoded); prev_overlap: [overlap_n, H, W, 3] linear ACEScg delivered by the
    previous window for the first overlap_n frames of this one. Returns (window in the same encoding, report)."""
    rep = {"settings": asdict(s), "overlap_frames": int(overlap_n)}
    if not s.enabled:
        return cur_window, {**rep, "applied": False, "reason": "disabled"}
    if overlap_n <= 0 or prev_overlap is None:
        return cur_window, {**rep, "applied": False, "reason": "no overlap with a delivered previous window"}
    lin = masks.to_linear(cur_window, transfer)
    cur_ov = lin[:overlap_n]
    if prev_overlap.shape[1:3] != cur_ov.shape[1:3]:
        raise ValueError(f"The previous window's frames are {tuple(prev_overlap.shape[1:3])} but this window is "
                         f"{tuple(cur_ov.shape[1:3])}; the windows must share one resolution.")
    before = mismatch_stops(prev_overlap, cur_ov, s)
    g, ginfo = measure(prev_overlap, cur_ov, s)
    rep.update(ginfo)
    if g is None:
        return cur_window, {**rep, "applied": False, "reason": ginfo["note"], "mismatch_stops_before": before}
    stops = float(g.clamp(min=1e-8).log2().abs().max())
    if stops > s.max_stops:
        return cur_window, {**rep, "applied": False, "mismatch_stops_before": before,
                            "reason": f"the windows differ by {stops:.2f} stops (> max_stops {s.max_stops}): not a "
                                      "seam to hide - check the windows"}
    w = ramp(lin.shape[0], max(overlap_n, keep_offset), s.seam_width).to(lin.device).view(-1, 1, 1, 1)
    gain = torch.exp(torch.log(g.clamp(min=1e-8)) * w)
    out = lin * gain
    after = mismatch_stops(prev_overlap, out[:overlap_n], s)
    res = masks.from_linear(out, transfer)
    if cur_window.shape[-1] > 3:
        res = torch.cat([res, cur_window[..., 3:].to(res.dtype)], dim=-1)
    curve = [round(float(math.exp(math.log(max(1e-8, float(g.mean()))) * float(v))), 6) for v in w.flatten()]
    rep.update({"applied": True, "mismatch_stops_before": round(before, 6), "mismatch_stops_after": round(after, 6),
                "max_correction_stops": round(stops, 6), "gain_curve_mean": curve})
    return res, rep
