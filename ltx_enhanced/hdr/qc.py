"""HDR delivery QC (owner A23 spec 4.12, output side). Reads what LTXE HDR Delivery wrote; never regenerates.

Works from the output folder alone: the per-window sidecars (metadata/<shot>_segNNN.json) give the shot's frame range,
the window count, fps, colour space and precision; the EXR frames give the pixels.

Checks (each a pass / warning / failure with the frames involved):
- frame numbering and count: every frame of the shot range present once (missing frames, missing windows);
- frame integrity: NaN / Inf, unreadable files;
- EXR metadata: precision, compression, AP1 chromaticities for ACEScg, frames per second vs the sidecar;
- exposure consistency: a jump in mean log2 luminance between consecutive frames above ``jump_stops`` (seams);
- temporal flicker: the mean absolute frame-to-frame change of mean luminance, in stops;
- highlight reconstruction: the share of pixels above 1.0 linear per frame (reported, not judged).
Depth / normals / vectors / matte are P3 passes: reported as not delivered.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
from dataclasses import asdict, dataclass

import numpy as np

from . import exr_writer

AP1 = exr_writer.AP1_CHROMATICITIES
_W = np.array((0.2722287, 0.6740818, 0.0536895), dtype=np.float32)


@dataclass(frozen=True)
class QCSettings:
    sample_every: int = 1           # read every n-th frame for the pixel statistics (numbering is always complete)
    stride: int = 4                 # pixel stride for the statistics
    jump_stops: float = 0.25        # consecutive-frame exposure jump that is flagged
    flicker_stops: float = 0.05     # mean frame-to-frame change above this is a warning
    contact_frames: int = 8


def _sidecars(folder: str, shot: str):
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "metadata", f"{shot}_seg*.json"))):
        try:
            with open(p, encoding="utf-8") as f:
                out.append(json.load(f))
        except (OSError, json.JSONDecodeError) as exc:
            out.append({"_error": f"{os.path.basename(p)}: {exc}"})
    return out


def _header(path: str) -> dict:
    import OpenImageIO as oiio  # Apache-2.0; optional
    inp = oiio.ImageInput.open(path)
    if inp is None:
        return {"_error": oiio.geterror()}
    s = inp.spec()
    inp.close()
    chroma = s.getattribute("chromaticities")
    return {"format": str(s.format), "compression": s.getattribute("compression"),
            "chromaticities": [round(float(v), 4) for v in chroma] if chroma else None,
            "fps": list(s.getattribute("FramesPerSecond") or []), "size": [s.width, s.height]}


def run_qc(folder: str, shot: str, s: QCSettings):
    sides = [d for d in _sidecars(folder, shot) if "_error" not in d]
    errors = [d["_error"] for d in _sidecars(folder, shot) if "_error" in d]
    rep = {"settings": asdict(s), "shot": shot, "checks": {}, "failed_frames": [], "warnings": []}
    if not sides:
        rep["checks"]["delivery_found"] = {"status": "fail", "detail": f"no sidecars for {shot!r} in {folder}"}
        rep["recommendation"] = "hold"
        return rep, None
    shot_range = next((d.get("shot_frame_range") for d in sides if d.get("shot_frame_range")), None)
    n_windows = max(int(d.get("segments", 0)) for d in sides)
    got_windows = sorted({int(d["segment"]) for d in sides})
    rep["windows"] = {"expected": n_windows, "delivered": got_windows}
    if shot_range is None:
        lo = min(d["frame_range"][0] for d in sides)
        hi = max(d["frame_range"][1] for d in sides)
        shot_range = [lo, hi]
        rep["warnings"].append("sidecars without shot_frame_range (older Delivery): range taken from the windows found")
    first, last = int(shot_range[0]), int(shot_range[1])
    digits = exr_writer.frame_digits(last)
    hdr_dir = os.path.join(folder, "hdr")
    pat = re.compile(rf"^{re.escape(shot)}_hdr_(\d+)\.exr$")
    found = {}
    for name in (os.listdir(hdr_dir) if os.path.isdir(hdr_dir) else []):
        m = pat.match(name)
        if m:
            found.setdefault(int(m.group(1)), []).append(name)
    expected = set(range(first, last + 1))
    missing = sorted(expected - set(found))
    extra = sorted(set(found) - expected)
    bad_pad = sorted(n for n, names in found.items() if any(len(x.rsplit("_", 1)[1].split(".")[0]) != digits
                                                             for x in names))
    rep["frame_range"] = [first, last]
    rep["checks"]["frame_numbering"] = {
        "status": "fail" if missing or extra or bad_pad else "pass",
        "expected": len(expected), "found": len(set(found) & expected), "missing": _ranges(missing),
        "extra": _ranges(extra), "wrong_padding": _ranges(bad_pad)}
    rep["checks"]["windows"] = {"status": "pass" if len(got_windows) == n_windows else "fail",
                                "missing": sorted(set(range(n_windows)) - set(got_windows))}
    if errors:
        rep["warnings"] += errors
    # metadata (first and last present frame)
    present = sorted(set(found) & expected)
    meta = {}
    if present:
        side0 = sides[0]
        for n in {present[0], present[-1]}:
            meta[n] = _header(os.path.join(hdr_dir, found[n][0]))
        hd = meta[present[0]]
        want_fmt = "half" if side0.get("precision") == "half" else "float"
        fps_ok = True
        if side0.get("source_fps"):
            num, den = exr_writer.fps_fraction(side0["source_fps"])
            fps_ok = hd.get("fps") == [num, den]
        chroma_ok = side0.get("output_color_space") != "acescg" or (hd.get("chromaticities") or [None])[:2] == \
            [round(AP1[0], 4), round(AP1[1], 4)]
        ok = hd.get("format") == want_fmt and hd.get("compression") == side0.get("compression") and fps_ok and chroma_ok
        rep["checks"]["exr_metadata"] = {"status": "pass" if ok else "fail", "header": hd,
                                         "expected": {"format": want_fmt, "compression": side0.get("compression"),
                                                      "fps": side0.get("source_fps"),
                                                      "color_space": side0.get("output_color_space")}}
    # pixel statistics
    stats, thumbs = [], []
    sample = present[::max(1, int(s.sample_every))]
    pick = set(sample[:: max(1, len(sample) // max(1, s.contact_frames))][: s.contact_frames]) if sample else set()
    bad = []
    for n in sample:
        path = os.path.join(hdr_dir, found[n][0])
        try:
            px = exr_writer.read_exr(path)
        except Exception as exc:  # noqa: BLE001
            bad.append(n)
            rep["warnings"].append(f"frame {n}: unreadable ({exc})")
            continue
        sub = px[:: s.stride, :: s.stride]
        finite = np.isfinite(sub).all()
        if not finite:
            bad.append(n)
        y = (np.nan_to_num(sub, nan=0.0, posinf=0.0, neginf=0.0) * _W).sum(-1)
        stats.append({"frame": n, "mean_log2_lum": float(np.log2(np.clip(y, 1e-6, None)).mean()),
                      "above_1": float((y > 1.0).mean()), "max": float(np.nanmax(sub)) if finite else None})
        if n in pick:
            thumbs.append(px)
    rep["checks"]["frame_integrity"] = {"status": "fail" if bad else "pass", "bad_frames": _ranges(bad),
                                        "sampled": len(sample)}
    jumps = []
    for a, b in zip(stats, stats[1:]):
        if b["frame"] - a["frame"] == max(1, int(s.sample_every)):
            d = abs(b["mean_log2_lum"] - a["mean_log2_lum"])
            if d > s.jump_stops:
                jumps.append({"between": [a["frame"], b["frame"]], "stops": round(d, 4)})
    steps = [abs(b["mean_log2_lum"] - a["mean_log2_lum"]) for a, b in zip(stats, stats[1:])]
    flicker = float(np.mean(steps)) if steps else 0.0
    rep["checks"]["exposure_consistency"] = {"status": "warning" if jumps else "pass", "jumps": jumps[:50],
                                             "threshold_stops": s.jump_stops}
    rep["checks"]["temporal_flicker"] = {"status": "warning" if flicker > s.flicker_stops else "pass",
                                         "mean_step_stops": round(flicker, 5), "threshold_stops": s.flicker_stops}
    rep["checks"]["highlight_reconstruction"] = {
        "status": "info", "frames_with_values_above_1": sum(1 for x in stats if x["above_1"] > 0),
        "max_share_above_1": round(max((x["above_1"] for x in stats), default=0.0), 6)}
    for p in ("depth", "normals", "vectors", "matte"):
        rep["checks"][f"{p}_pass"] = {"status": "n/a", "detail": "not delivered (A23 P3)"}
    rep["failed_frames"] = sorted(set(missing) | set(bad))
    rep["warning_intervals"] = [j["between"] for j in jumps]
    statuses = [c["status"] for c in rep["checks"].values()]
    rep["recommendation"] = "hold" if "fail" in statuses else ("review" if "warning" in statuses else "accept")
    rep["stats"] = stats if len(stats) <= 400 else stats[::max(1, len(stats) // 400)]
    return rep, thumbs


def _ranges(nums):
    out, start, prev = [], None, None
    for n in sorted(nums):
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append([start, prev])
            start = prev = n
    if start is not None:
        out.append([start, prev])
    return out


def contact_sheet(thumbs, exposure: float = 0.0, cols: int = 4, width: int = 256):
    """Tonemapped (Reinhard on luminance-preserving RGB, then sRGB) grid of the sampled frames; display only."""
    if not thumbs:
        return np.zeros((1, 64, 64, 3), np.float32)
    tiles = []
    for px in thumbs:
        h, w = px.shape[:2]
        step = max(1, int(math.ceil(w / width)))
        t = np.nan_to_num(px[::step, ::step], nan=0.0, posinf=0.0, neginf=0.0).clip(min=0) * (2.0 ** exposure)
        t = t / (1.0 + t)
        t = np.where(t <= 0.0031308, 12.92 * t, 1.055 * np.power(np.clip(t, 0.0031308, None), 1 / 2.4) - 0.055)
        tiles.append(t.clip(0, 1).astype(np.float32))
    th = max(t.shape[0] for t in tiles)
    tw = max(t.shape[1] for t in tiles)
    rows = int(math.ceil(len(tiles) / cols))
    sheet = np.zeros((rows * th, cols * tw, 3), np.float32)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * th:r * th + t.shape[0], c * tw:c * tw + t.shape[1]] = t
    return sheet[None]
