"""Temporal segment plan for SDR -> HDR on LTX-2.5 (A23 L9.130). No ComfyUI import.

Every generated window is 8k+1 frames (the video VAE's frame rule, geometry.py). Windows overlap; the seam between two
windows sits in the middle of their overlap, and the accepted ranges tile the clip EXACTLY (no frame lost, none
twice). The last window is padded up to 8k+1 with copies of the final frame; that padding is generated and discarded
(`pad_tail`). Conditioning runs at min(source fps, 30) - what the official graph does - while the delivered frames
keep the source frame rate and numbering.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from fractions import Fraction

from ..geometry import LTXAV, LatentGeometry

MODEL_FAMILY = "ltx-2.5-sdr-to-hdr"
MAX_CONDITIONING_FPS = 30


@dataclass(frozen=True)
class Segment:
    index: int
    gen_start: int          # first source frame (0-based) of the generated window
    gen_frames: int         # frames generated (8k+1), including pad_tail
    pad_tail: int           # copies of the last source frame appended to reach 8k+1 (discarded)
    keep_start: int         # first source frame this window delivers
    keep_end: int           # one past the last delivered source frame
    overlap_head: int       # frames shared with the previous window (seam context)

    @property
    def gen_end(self) -> int:
        """One past the last REAL source frame in the window."""
        return self.gen_start + self.gen_frames - self.pad_tail


@dataclass(frozen=True)
class SegmentPlan:
    total_frames: int
    segment_frames: int
    overlap: int
    source_fps: str
    conditioning_fps: str
    first_frame_number: int
    width: int
    height: int
    segments: tuple = field(default_factory=tuple)
    model_family: str = MODEL_FAMILY

    def to_dict(self) -> dict:
        d = asdict(self)
        d["segments"] = [asdict(s) for s in self.segments]
        d["schema"] = 1
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentPlan":
        if d.get("model_family") != MODEL_FAMILY:
            raise ValueError(f"This plan is for '{d.get('model_family')}', not {MODEL_FAMILY}; plan the clip again.")
        segs = tuple(Segment(**s) for s in d["segments"])
        keys = {k: d[k] for k in ("total_frames", "segment_frames", "overlap", "source_fps", "conditioning_fps",
                                  "first_frame_number", "width", "height")}
        return cls(segments=segs, **keys)

    def tokens_per_segment(self, geometry: LatentGeometry = LTXAV) -> int:
        lat_f = geometry.latent_frames(self.segment_frames)
        return lat_f * (geometry.aligned_size(self.height) // geometry.spatial) * \
            (geometry.aligned_size(self.width) // geometry.spatial)

    def pixel_buffer_bytes(self) -> int:
        """ESTIMATE: one float32 RGB window in memory (not a measurement of the model's peak use)."""
        return self.segment_frames * self.height * self.width * 3 * 4

    def summary(self) -> str:
        lines = [f"clip        {self.total_frames} frames at {self.source_fps} fps "
                 f"(numbered {self.first_frame_number}-{self.first_frame_number + self.total_frames - 1}), "
                 f"{self.width}x{self.height}",
                 f"timing      conditioning at {self.conditioning_fps} fps; delivered frames keep {self.source_fps} fps",
                 f"windows     {len(self.segments)} x up to {self.segment_frames} frames (8k+1), overlap "
                 f"{self.overlap}; seam in the middle of each overlap",
                 f"estimate    {self.tokens_per_segment():,} tokens and ~{self.pixel_buffer_bytes() / 2**30:.2f} GB of "
                 "float32 frames per window (ESTIMATE, not measured)"]
        for s in self.segments:
            lines.append(f"  #{s.index}: generate {s.gen_start}-{s.gen_end - 1}"
                         + (f" (+{s.pad_tail} padding)" if s.pad_tail else "")
                         + f", deliver {s.keep_start}-{s.keep_end - 1}")
        return "\n".join(lines)


NTSC_RATES = tuple(Fraction(n * 1000, 1001) for n in (24, 30, 48, 60, 120))


def _fps(fps) -> Fraction:
    """Exact rate. A float typed into a widget (23.976, 29.97, 59.94) means the NTSC fraction, not 2997/100."""
    if isinstance(fps, Fraction):
        f = fps
    else:
        f = Fraction(fps).limit_denominator(1001)
        for ntsc in NTSC_RATES:
            if abs(float(f) - float(ntsc)) < 1e-3:
                f = ntsc
                break
    if f <= 0:
        raise ValueError(f"The frame rate must be above 0, got {fps}.")
    return f


def plan_segments(total_frames: int, source_fps, width: int, height: int, segment_frames: int = 97,
                  overlap: int = 16, first_frame_number: int = 0,
                  geometry: LatentGeometry = LTXAV) -> SegmentPlan:
    total, seg, ov = int(total_frames), int(segment_frames), int(overlap)
    if total < 1:
        raise ValueError(f"The clip has {total} frames; it needs at least 1.")
    if seg < 9 or (seg - 1) % geometry.temporal:
        raise ValueError(f"segment_frames must be 8k+1 and at least 9 (9, 17, ..., 97, 121); got {seg}.")
    if ov < 0 or ov > seg - 9:
        raise ValueError(f"overlap must be 0..{seg - 9} for {seg}-frame windows (at least 8 new frames per window); "
                         f"got {ov}.")
    fps = _fps(source_fps)
    cond = min(fps, Fraction(MAX_CONDITIONING_FPS))
    step = seg - ov
    starts = []
    start = 0
    while True:
        starts.append(start)
        if start + seg >= total:
            break
        start += step
    segs = []
    for i, s in enumerate(starts):
        real = min(seg, total - s)
        gen = geometry.aligned_frames(real)
        keep_start = 0 if i == 0 else s + ov // 2
        keep_end = total if i == len(starts) - 1 else starts[i + 1] + ov // 2
        segs.append(Segment(index=i, gen_start=s, gen_frames=gen, pad_tail=gen - real, keep_start=keep_start,
                            keep_end=keep_end, overlap_head=0 if i == 0 else ov))
    plan = SegmentPlan(total_frames=total, segment_frames=seg, overlap=ov, source_fps=str(fps),
                       conditioning_fps=str(cond), first_frame_number=int(first_frame_number), width=int(width),
                       height=int(height), segments=tuple(segs))
    check_exact_cover(plan)
    return plan


def check_exact_cover(plan: SegmentPlan) -> None:
    """Every source frame is delivered by exactly one window, from inside that window's real frames."""
    expect = 0
    for s in plan.segments:
        if s.keep_start != expect or s.keep_end <= s.keep_start:
            raise AssertionError(f"window {s.index} delivers {s.keep_start}-{s.keep_end}, expected to start at {expect}")
        if not (s.gen_start <= s.keep_start and s.keep_end <= s.gen_end):
            raise AssertionError(f"window {s.index} delivers frames it does not generate")
        expect = s.keep_end
    if expect != plan.total_frames:
        raise AssertionError(f"the windows deliver {expect} of {plan.total_frames} frames")
