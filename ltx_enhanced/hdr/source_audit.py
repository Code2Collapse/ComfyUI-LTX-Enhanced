"""Source audit before SDR -> HDR (A23 L9.128). No ComfyUI import; CPU; samples frames, never loads a model.

Everything here is measured from the frames (plus optional container metadata the caller passes in): sizes, frame
rate, an estimate of the source bit depth from how the values sit on the 8 / 10-bit code grids, clipped highlights,
near-white areas, crushed shadows, duplicate frames, cuts, softness and an "upscaled" hint. Values above 1.0 mean the
input is already scene-linear / HDR - it belongs in the Native HDR workflow, not in SDR -> HDR reconstruction.
Each finding is a plain sentence in `warnings`; the thresholds are inputs, and the report records which were used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

LUMA = torch.tensor([0.2126, 0.7152, 0.0722])           # Rec.709 luma weights (display-referred SDR)


@dataclass(frozen=True)
class AuditSettings:
    scan_every: int = 1             # look at every n-th frame
    clip_level: float = 0.998       # a channel at or above this is clipped
    near_white: float = 0.95        # luma at or above this is near-white
    crush_level: float = 0.004      # luma at or below this is crushed
    clip_warn_pct: float = 1.0      # warn when this % of pixels (mean over frames) clip
    crush_warn_pct: float = 5.0
    duplicate_diff: float = 1e-4    # mean |frame - previous| below this = duplicate
    cut_diff: float = 0.25          # mean |frame - previous| above this = likely cut
    soft_threshold: float = 1e-4    # mean Laplacian energy below this = soft


def _luma(x: torch.Tensor) -> torch.Tensor:
    return (x[..., :3].float() * LUMA).sum(-1)


def _on_grid(x: torch.Tensor, levels: int, tol: float) -> float:
    q = x * levels
    return float(((q - q.round()).abs() <= tol * levels).float().mean())


def estimate_bit_depth(sample: torch.Tensor) -> str:
    """'8', '10', 'float' or 'unknown', from where the values sit. A float source (or anything off both grids) is
    not called 8-bit, so 8-bit-only processing never touches it."""
    x = sample[..., :3].float().flatten()
    if x.numel() > 400_000:
        x = x[torch.randperm(x.numel(), generator=torch.Generator().manual_seed(0))[:400_000]]
    if float(x.max()) > 1.0 + 1e-3 or float(x.min()) < -1e-3:
        return "float"
    if _on_grid(x, 255, 1e-5) > 0.995:
        return "8"
    if _on_grid(x, 1023, 1e-5) > 0.995:
        return "10"
    return "unknown"


def _laplacian_energy(y: torch.Tensor) -> torch.Tensor:
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])[None, None]
    return torch.nn.functional.conv2d(y[:, None], k).pow(2).mean(dim=(1, 2, 3))


UPSCALE_HINT_BELOW = 0.12


def _upscale_hint(y: torch.Tensor) -> float:
    """Spectral energy above half the Nyquist frequency divided by the energy in the band below it (1/16-1/4 cycles
    per pixel). Scale-free, so contrast does not matter. Measured 2026-11-04 on synthetic frames: 2x-4x bilinear /
    bicubic upscales 0.012-0.087, blurred noise >= 0.16, a 1/f "natural" image 0.56 - hence the 0.12 cut. Only a hint."""
    if min(y.shape[-2:]) < 16:
        return float("nan")
    f = torch.fft.rfft2(y - y.mean(dim=(-2, -1), keepdim=True)).abs().pow(2)
    fy = torch.fft.fftfreq(y.shape[-2]).abs()[:, None]
    fx = torch.fft.rfftfreq(y.shape[-1])[None, :]
    r = torch.maximum(fy, fx)
    hi, mid = f[..., r > 0.25].sum(), f[..., (r > 0.0625) & (r <= 0.25)].sum()
    return float(hi / (mid + 1e-12))


def audit(images: torch.Tensor, fps: float = 0.0, metadata: dict | None = None,
          settings: AuditSettings = AuditSettings()) -> dict:
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError("The source audit needs an IMAGE batch [frames, height, width, 3].")
    n, h, w = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    idx = list(range(0, n, max(1, int(settings.scan_every))))
    s = images[idx, ..., :3].float().cpu()
    y = _luma(s)
    warnings: list[str] = []
    depth = estimate_bit_depth(s)
    hdr_input = depth == "float"
    if hdr_input:
        warnings.append("Values go above 1.0 (or below 0): this is already scene-linear / HDR material. Use the Native "
                        "HDR workflow; SDR -> HDR reconstruction expects display-referred SDR frames.")
    clip_pct = (s >= settings.clip_level).any(-1).float().mean(dim=(1, 2)) * 100.0
    white_pct = (y >= settings.near_white).float().mean(dim=(1, 2)) * 100.0
    crush_pct = (y <= settings.crush_level).float().mean(dim=(1, 2)) * 100.0
    if float(clip_pct.mean()) >= settings.clip_warn_pct:
        warnings.append(f"{float(clip_pct.mean()):.1f}% of the pixels clip on average (peak {float(clip_pct.max()):.1f}% "
                        "in one frame): these are the highlights SDR -> HDR reconstructs.")
    if float(crush_pct.mean()) >= settings.crush_warn_pct:
        warnings.append(f"{float(crush_pct.mean()):.1f}% of the pixels are crushed to black on average: consider "
                        "shadow recovery.")
    diffs = (s[1:] - s[:-1]).abs().mean(dim=(1, 2, 3)) if s.shape[0] > 1 else torch.zeros(0)
    dup = [idx[i + 1] for i, d in enumerate(diffs.tolist()) if d < settings.duplicate_diff]
    cuts = [idx[i + 1] for i, d in enumerate(diffs.tolist()) if d > settings.cut_diff]
    if dup and settings.scan_every == 1:
        warnings.append(f"{len(dup)} frame(s) repeat the previous one (first: {dup[:8]}): check the frame rate or a "
                        "frame-rate conversion upstream.")
    sharp = _laplacian_energy(y)
    soft = float(sharp.mean()) < settings.soft_threshold
    if soft:
        warnings.append("The footage is very soft (little fine detail).")
    up = _upscale_hint(y)
    if up == up and up < UPSCALE_HINT_BELOW:
        warnings.append("Almost no detail above half the resolution: the source may be upscaled.")
    if depth == "8":
        warnings.append("8-bit source: banding in smooth gradients and highlights is likely (8-bit cleanup applies).")
    md = dict(metadata or {})
    report = {
        "schema": 1, "frames": n, "width": w, "height": h, "aspect": round(w / h, 6), "fps": float(fps) or None,
        "duration_s": round(n / fps, 4) if fps else None, "bit_depth_estimate": depth, "already_hdr": hdr_input,
        "scanned_frames": len(idx), "clip_pct_mean": round(float(clip_pct.mean()), 4),
        "clip_pct_max": round(float(clip_pct.max()), 4), "near_white_pct_mean": round(float(white_pct.mean()), 4),
        "crushed_pct_mean": round(float(crush_pct.mean()), 4), "duplicate_frames": dup, "likely_cuts": cuts,
        "laplacian_energy_mean": float(sharp.mean()), "upscale_hf_ratio": up, "soft": soft,
        "colour": {k: md.get(k) for k in ("primaries", "transfer", "matrix", "range", "chroma", "pix_fmt")},
        "metadata": md, "settings": asdict(settings), "warnings": warnings,
    }
    return report


def report_text(r: dict) -> str:
    c = r["colour"]
    tags = ", ".join(f"{k} {v}" for k, v in c.items() if v) or "no colour tags given"
    lines = [f"{r['frames']} frames, {r['width']}x{r['height']} (aspect {r['aspect']})"
             + (f", {r['fps']:g} fps, {r['duration_s']} s" if r["fps"] else ", frame rate not given"),
             f"bit depth (estimated from the values): {r['bit_depth_estimate']}; {tags}",
             f"clipped {r['clip_pct_mean']}% (max {r['clip_pct_max']}%), near-white {r['near_white_pct_mean']}%, "
             f"crushed {r['crushed_pct_mean']}%; {len(r['duplicate_frames'])} duplicate frame(s), "
             f"{len(r['likely_cuts'])} likely cut(s)"]
    lines += [f"- {w}" for w in r["warnings"]] or ["- no warnings"]
    return "\n".join(lines)
