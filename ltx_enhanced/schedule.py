"""Checks on a sigma schedule before tiled sampling.

ComfyUI core's LTX schedulers derive their shift from the latent's token count; above about 2K the schedule
saturates and at DCP 4K it is NaN (crash_repros/001-ltxv-schedule-at-4k). The model in a tiled run sees a tile, so
the token count that matters is the tile's. These helpers only inspect numbers; they never build a schedule.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

SATURATED = 0.999


def tokens(latent_frames: int, rows: int, cols: int) -> int:
    """Video tokens the model attends over (LTX patchifies one token per latent cell)."""
    return int(latent_frames) * int(rows) * int(cols)


def check_sigmas(sigmas: Sequence[float]) -> dict:
    vals = [float(s) for s in sigmas]
    nan = sum(1 for v in vals if math.isnan(v) or math.isinf(v))
    finite = [v for v in vals if not (math.isnan(v) or math.isinf(v))]
    rising = sum(1 for a, b in zip(vals, vals[1:]) if not (math.isnan(a) or math.isnan(b)) and b > a)
    steps = max(0, len(vals) - 1)
    saturated = sum(1 for v in vals[:-1] if not math.isnan(v) and v >= SATURATED)
    flat = steps > 2 and len({round(v, 6) for v in finite[:-1]}) == 1
    problems = []
    if nan:
        problems.append(f"{nan} of {len(vals)} sigmas are NaN or infinite")
    if rising:
        problems.append(f"the schedule rises {rising} time(s); it must go down")
    if steps and saturated > steps / 2:
        problems.append(f"{saturated} of {steps} steps sit at sigma >= {SATURATED}, so almost no denoising happens "
                        "until the last step")
    if flat:
        problems.append("every step has the same sigma")
    if problems:
        message = ("The sigma schedule is unusable: " + "; ".join(problems) + ". This happens when an LTX scheduler "
                   "is given the full high-resolution latent: its shift grows with the token count. Give the "
                   "scheduler a latent of the tile size instead, or use manual sigmas.")
    else:
        message = "The sigma schedule looks usable."
    return {"steps": steps, "nan_count": nan, "rising": rising, "saturated_steps": saturated, "flat": flat,
            "ok": not problems, "message": message}
