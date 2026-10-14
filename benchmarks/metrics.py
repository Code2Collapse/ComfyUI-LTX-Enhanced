"""Output metrics for tiled-sampling benchmarks (no ComfyUI import; torch only).

Frames are float tensors [F, H, W, C] in 0..1 (ComfyUI IMAGE layout). Every metric is a plain number with a stated
meaning, so results files can be compared across runs and machines:

* ``seam_score`` - how much stronger the edges are on known tile boundaries than elsewhere. ~1.0 = no visible seam.
* ``grid_score`` - the same idea without a plan: the periodic peak of the edge profile at a given stride.
* ``flicker`` - mean frame-to-frame change of fine detail (high-pass luminance), lower = steadier.
* ``motion_profile`` - mean frame-to-frame change per frame, to spot a frozen start (upstream issue #384).
* ``bad_frames`` - frames that are NaN, infinite, or blank (near-constant), which a pipeline must never pass on.
* ``psnr`` - against a reference clip of the same size.

Measure the content area (LTXE Crop Back region "content"): letterbox bars are flat, and a flat area has no edges to
compare against. Ratios use a floor of ``EDGE_FLOOR`` (0.001 luminance per pixel, below what is visible) in the
denominator, so flat frames give finite numbers instead of infinities.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

EDGE_FLOOR = 1e-3


def _luma(frames: torch.Tensor) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"expected frames [F, H, W, C], got {tuple(frames.shape)}")
    f = frames.float()
    if f.shape[-1] >= 3:
        return 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]
    return f[..., 0]


def edge_profiles(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean absolute horizontal / vertical neighbour difference per column / row: [W-1], [H-1]."""
    y = _luma(frames)
    cols = (y[:, :, 1:] - y[:, :, :-1]).abs().mean(dim=(0, 1))
    rows = (y[:, 1:, :] - y[:, :-1, :]).abs().mean(dim=(0, 2))
    return cols, rows


def _ratio(profile: torch.Tensor, positions: Sequence[int], halo: int = 2) -> float:
    """Profile value at the given boundary positions over the median of the profile away from them."""
    n = profile.numel()
    pos = sorted({int(p) for p in positions if 0 <= int(p) < n})
    if not pos or n < 4:
        return 1.0
    mask = torch.ones(n, dtype=torch.bool)
    for p in pos:
        mask[max(0, p - halo):p + halo + 1] = False
    base = profile[mask].median() if bool(mask.any()) else profile.median()
    at = torch.stack([profile[max(0, p - 1):p + 2].max() for p in pos]).mean()
    return float(at / base.clamp_min(EDGE_FLOOR))


def seam_score(frames: torch.Tensor, x_boundaries: Sequence[int] = (), y_boundaries: Sequence[int] = ()) -> dict:
    """Edge strength on the given tile boundaries (pixel columns / rows, the first pixel of the next tile or the
    edge of an overlap) relative to the rest of the frame. ~1.0 means the boundaries look like any other column."""
    cols, rows = edge_profiles(frames)
    xs = [b - 1 for b in x_boundaries]    # a boundary at pixel b shows in the difference between b-1 and b
    ys = [b - 1 for b in y_boundaries]
    sx, sy = _ratio(cols, xs), _ratio(rows, ys)
    return {"x": sx, "y": sy, "max": max(sx, sy)}


def boundaries_from_plan(plan: dict, scale: float = 1.0) -> tuple[list[int], list[int]]:
    """Both edges of every overlap of an LTXE plan (``Plan.to_dict()``), in output pixels (``scale`` after an
    upscale), shifted into the cropped content area."""
    s = plan["geometry"]["spatial"] * scale
    pad = plan["pad"]
    ox, oy = pad["content"]["x"] * scale, pad["content"]["y"] * scale

    def edges(starts_sizes, origin):
        out = set()
        for start, size in starts_sizes[1:]:
            out.add(round(start * s - origin))
        for start, size in starts_sizes[:-1]:
            out.add(round((start + size) * s - origin))
        return sorted(out)

    return edges(plan["tiles"]["cols"], ox), edges(plan["tiles"]["rows"], oy)


def grid_score(frames: torch.Tensor, stride_px: int) -> float:
    """Strength of an edge pattern that repeats every ``stride_px`` pixels (a fixed tile grid), ~1.0 = none."""
    cols, rows = edge_profiles(frames)
    out = []
    for prof in (cols, rows):
        n = prof.numel()
        if stride_px < 2 or n < 2 * stride_px:
            continue
        phase = torch.stack([prof[p::stride_px].mean() for p in range(stride_px)])
        out.append(float(phase.max() / phase.median().clamp_min(EDGE_FLOOR)))
    return max(out) if out else 1.0


def _highpass(y: torch.Tensor) -> torch.Tensor:
    blur = F.avg_pool2d(y.unsqueeze(1), 5, stride=1, padding=2, count_include_pad=False).squeeze(1)
    return y - blur


def flicker(frames: torch.Tensor) -> float:
    """Mean absolute frame-to-frame change of the high-pass luminance (fine detail), in 0..1 units."""
    y = _luma(frames)
    if y.shape[0] < 2:
        return 0.0
    hp = _highpass(y)
    return float((hp[1:] - hp[:-1]).abs().mean())


def motion_profile(frames: torch.Tensor) -> list[float]:
    """Mean absolute luminance change between consecutive frames, one value per frame step."""
    y = _luma(frames)
    return [float((y[i + 1] - y[i]).abs().mean()) for i in range(y.shape[0] - 1)]


def frozen_start(frames: torch.Tensor, fps: float, seconds: float = 1.0, ratio: float = 0.2) -> bool:
    """True when the first ``seconds`` move less than ``ratio`` of the rest of the clip (a stalled start)."""
    m = motion_profile(frames)
    k = max(1, int(round(fps * seconds)))
    if len(m) <= k + 1:
        return False
    head, rest = sum(m[:k]) / k, sum(m[k:]) / len(m[k:])
    return rest > 1e-6 and head < ratio * rest


def bad_frames(frames: torch.Tensor, blank_std: float = 1e-3) -> dict:
    f = frames.float()
    nonfinite = [int(i) for i in range(f.shape[0]) if not bool(torch.isfinite(f[i]).all())]
    blank = [int(i) for i in range(f.shape[0]) if i not in nonfinite and float(f[i].std()) < blank_std]
    return {"nonfinite": nonfinite, "blank": blank, "ok": not nonfinite and not blank}


def psnr(frames: torch.Tensor, reference: torch.Tensor) -> float:
    if frames.shape != reference.shape:
        raise ValueError(f"PSNR needs the same shape: {tuple(frames.shape)} vs {tuple(reference.shape)}")
    mse = float(((frames.float() - reference.float()) ** 2).mean())
    return math.inf if mse == 0 else 10 * math.log10(1.0 / mse)


def all_metrics(frames: torch.Tensor, plan: dict | None = None, scale: float = 1.0, fps: float = 24.0,
                reference: torch.Tensor | None = None) -> dict:
    out = {"frames": int(frames.shape[0]), "size": [int(frames.shape[2]), int(frames.shape[1])],
           "flicker": flicker(frames), "frozen_start": frozen_start(frames, fps), "bad_frames": bad_frames(frames)}
    if plan is not None:
        xb, yb = boundaries_from_plan(plan, scale)
        out["seam_score"] = seam_score(frames, xb, yb)
        out["seam_boundaries_px"] = {"x": xb, "y": yb}
    if reference is not None:
        out["psnr"] = psnr(frames, reference)
    return out
