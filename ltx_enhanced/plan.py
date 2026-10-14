"""One plan for a run: padding (space and time) plus the spatial tile grid, with a readable summary.

The plan is JSON-serialisable so it can travel between nodes and be saved with results.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .geometry import LTXV, LatentGeometry
from .padding import PadPlan, make_plan, summary as pad_summary
from .schedule import tokens
from .tiles import TilePlan, plan_tiles

# Token counts at which ComfyUI core's LTX schedules were MEASURED to break (97 frames, 20 steps, default shifts;
# crash_repros/001-ltxv-schedule-at-4k): ModelSamplingLTXV + a generic scheduler keeps every step at sigma >= 0.999
# from 26 520 tokens (1920x1088); LTXVScheduler turns into a staircase at 46 800 tokens (2560x1440), a flat 0.1 at
# 106 080 and NaN at 113 152 (4096x2176).
MEASURED_MODEL_SAMPLING_SATURATES = 26_520
MEASURED_SCHEDULER_BREAKS = 46_800


@dataclass(frozen=True)
class Plan:
    pad: PadPlan
    tiles: TilePlan
    latent_frames: int
    geometry: LatentGeometry

    @property
    def canvas_tokens(self) -> int:
        return tokens(self.latent_frames, self.tiles.height, self.tiles.width)

    @property
    def tile_tokens(self) -> int:
        return tokens(self.latent_frames, self.tiles.tile_h, self.tiles.tile_w)

    def warnings(self) -> list[str]:
        out = []
        for name, n in (("whole canvas", self.canvas_tokens), ("one tile", self.tile_tokens)):
            if n >= MEASURED_SCHEDULER_BREAKS:
                out.append(f"{name}: {n:,} tokens. Core's LTX schedulers fed a latent this big were measured to give a "
                           "staircase, flat or NaN schedule. Give the scheduler a tile-sized latent or use manual sigmas.")
            elif n >= MEASURED_MODEL_SAMPLING_SATURATES:
                out.append(f"{name}: {n:,} tokens. ModelSamplingLTXV with a generic scheduler was measured to keep every "
                           "step at sigma >= 0.999 at this size. Prefer LTXVScheduler on a tile-sized latent, or manual sigmas.")
        if self.tiles.count > 1 and self.tile_tokens >= MEASURED_MODEL_SAMPLING_SATURATES:
            out.append("The tiles themselves are large; smaller tiles keep the schedule in its normal range.")
        return out

    def to_dict(self) -> dict:
        return {"schema": 1, "pad": self.pad.to_dict(), "tiles": self.tiles.to_dict(), "latent_frames": self.latent_frames,
                "geometry": {"spatial": self.geometry.spatial, "temporal": self.geometry.temporal,
                             "channels": self.geometry.channels, "causal_first": self.geometry.causal_first},
                "canvas_tokens": self.canvas_tokens, "tile_tokens": self.tile_tokens, "warnings": self.warnings()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)

    @classmethod
    def from_dict(cls, d: dict) -> "Plan":
        g = LatentGeometry(**d["geometry"])
        pad = PadPlan.from_dict(d["pad"])
        t = d["tiles"]
        tiles = plan_tiles(t["height"], t["width"], t["tile_h"], t["tile_w"], t["min_overlap"])
        return cls(pad=pad, tiles=tiles, latent_frames=int(d["latent_frames"]), geometry=g)

    def summary(self) -> str:
        t = self.tiles
        lines = [pad_summary(self.pad),
                 f"latent      {t.width}x{t.height} cells x {self.latent_frames} frames = {self.canvas_tokens:,} tokens"]
        if t.count == 1:
            lines.append("tiles       one tile covers the canvas (no tiling)")
        else:
            s = self.geometry.spatial
            lines.append(f"tiles       {len(t.cols)} x {len(t.rows)} = {t.count} tiles of {t.tile_w * s}x{t.tile_h * s} px, "
                         f"overlap >= {t.realised_min_overlap * s} px, up to {t.max_coverage} tiles per cell; "
                         f"{self.tile_tokens:,} tokens per tile")
        lines += [f"warning     {w}" for w in self.warnings()]
        return "\n".join(lines)


def make_run_plan(src_w: int, src_h: int, src_frames: int, deliver_w: int | None = None, deliver_h: int | None = None,
                  tile_w_px: int | None = None, tile_h_px: int | None = None, min_overlap_px: int | None = None,
                  fit: str = "pad", mode: str = "reflect", value: float = 0.0, placement: str = "center",
                  geometry: LatentGeometry = LTXV) -> Plan:
    pad = make_plan(src_w, src_h, src_frames, deliver_w, deliver_h, fit=fit, mode=mode, value=value,
                    placement=placement, geometry=geometry)
    s = geometry.spatial
    rows, cols = pad.sample_h // s, pad.sample_w // s
    th = rows if not tile_h_px else max(1, geometry.aligned_size(tile_h_px) // s)
    tw = cols if not tile_w_px else max(1, geometry.aligned_size(tile_w_px) // s)
    ov = None if min_overlap_px is None else -(-max(0, min_overlap_px) // s)
    tiles = plan_tiles(rows, cols, th, tw, ov)
    return Plan(pad=pad, tiles=tiles, latent_frames=geometry.latent_frames(pad.sample_frames), geometry=geometry)
