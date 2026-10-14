"""Tile planning and blend weights for tiled denoising.

Design: docs/design/architecture.md ("Tile rules"). Per axis: the fewest tiles whose overlap is at least
``min_overlap``, first and last pinned to the edges, the rest evenly spread (the MultiDiffusion placement, arXiv
2302.08113). The requested overlap is capped at half a tile. Weights are trapezoids whose ramps exclude both end
points, so even a one-cell overlap blends 50/50 (the hard-seam failure of endpoint-inclusive ramps is described in
docs/re-notes/ltxvideo-tiled-looping-vae.md). Even spacing can still make the realised overlap larger than half a tile
(for example 50 cells in 32-cell tiles: starts 0, 9, 18), so a cell may sit under three tiles of one axis; the blend
always divides by the summed weights, and ``TilePlan.max_coverage`` reports it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

WEIGHTS = ("trapezoid", "gaussian")


@dataclass(frozen=True)
class AxisTile:
    start: int
    size: int
    left_ramp: int      # overlap with the previous tile (0 for the first)
    right_ramp: int     # overlap with the next tile (0 for the last)

    @property
    def end(self) -> int:
        return self.start + self.size


def split_axis(length: int, tile: int, min_overlap: int) -> list[AxisTile]:
    """Cover [0, length) with fixed-size tiles overlapping by at least ``min_overlap`` cells."""
    if length < 1 or tile < 1:
        raise ValueError(f"length and tile must be positive, got length={length}, tile={tile}")
    tile = min(tile, length)
    if tile == length:
        return [AxisTile(0, length, 0, 0)]
    if min_overlap < 0 or 2 * min_overlap > tile:
        raise ValueError(f"The overlap ({min_overlap}) must be between 0 and half the tile ({tile // 2}): "
                         "a larger overlap makes a tile overlap the one after its neighbour.")
    n = 2
    while (n * tile - length) / (n - 1) < min_overlap:
        n += 1
    starts = [round(i * (length - tile) / (n - 1)) for i in range(n)]
    out = []
    for i, s in enumerate(starts):
        left = starts[i - 1] + tile - s if i else 0
        right = s + tile - starts[i + 1] if i < n - 1 else 0
        out.append(AxisTile(s, tile, left, right))
    return out


def trapezoid_1d(size: int, left_ramp: int, right_ramp: int) -> torch.Tensor:
    """1 in the middle, linear ramps over the overlaps; a ramp of r cells takes the r interior points of a 0..1 line
    with r + 2 points, so neighbouring ramps sum to exactly 1."""
    w = torch.ones(size, dtype=torch.float64)
    if left_ramp:
        w[:left_ramp] = torch.linspace(0.0, 1.0, left_ramp + 2, dtype=torch.float64)[1:-1]
    if right_ramp:
        w[size - right_ramp:] = torch.minimum(w[size - right_ramp:],
                                              torch.linspace(1.0, 0.0, right_ramp + 2, dtype=torch.float64)[1:-1])
    return w


def gaussian_1d(size: int, var: float = 0.05, floor: float = 0.0) -> torch.Tensor:
    """Gaussian bump over the tile (Mixture of Diffusers style, arXiv 2302.02412), optional floor."""
    x = torch.arange(size, dtype=torch.float64)
    mid = (size - 1) / 2
    g = torch.exp(-((x - mid) ** 2) / (2 * var * size * size))
    return floor + (1 - floor) * g


def axis_weights(tile: AxisTile, kind: str = "trapezoid", var: float = 0.05, floor: float = 0.0) -> torch.Tensor:
    if kind == "trapezoid":
        return trapezoid_1d(tile.size, tile.left_ramp, tile.right_ramp)
    if kind == "gaussian":
        return gaussian_1d(tile.size, var, floor)
    raise ValueError(f"weight kind must be one of {', '.join(WEIGHTS)}; got {kind!r}")


@dataclass(frozen=True)
class Tile2D:
    y: AxisTile
    x: AxisTile

    def slices(self) -> tuple[slice, slice]:
        return slice(self.y.start, self.y.end), slice(self.x.start, self.x.end)


@dataclass(frozen=True)
class TilePlan:
    height: int             # latent rows
    width: int              # latent columns
    tile_h: int
    tile_w: int
    min_overlap: int | None    # requested; None = half of each axis's tile
    rows: tuple[AxisTile, ...]
    cols: tuple[AxisTile, ...]

    @property
    def tiles(self) -> list[Tile2D]:
        return [Tile2D(r, c) for r in self.rows for c in self.cols]

    @property
    def count(self) -> int:
        return len(self.rows) * len(self.cols)

    @property
    def realised_min_overlap(self) -> int:
        ov = [t.left_ramp for t in self.rows[1:]] + [t.left_ramp for t in self.cols[1:]]
        return min(ov) if ov else 0

    @property
    def axis_coverage(self) -> tuple[int, int]:
        """Most tiles covering one cell along the rows axis and along the columns axis."""
        def axis(ts):
            cover = [0] * ts[-1].end
            for t in ts:
                for i in range(t.start, t.end):
                    cover[i] += 1
            return max(cover)
        return axis(self.rows), axis(self.cols)

    @property
    def max_coverage(self) -> int:
        """Most tiles covering one cell (1 = no overlap, 4 = an interior corner of a 2-D grid)."""
        r, c = self.axis_coverage
        return r * c

    def weight(self, tile: Tile2D, kind: str = "trapezoid", var: float = 0.05, floor: float = 0.0) -> torch.Tensor:
        return axis_weights(tile.y, kind, var, floor)[:, None] * axis_weights(tile.x, kind, var, floor)[None, :]

    def weight_sum(self, kind: str = "trapezoid", var: float = 0.05, floor: float = 0.0) -> torch.Tensor:
        s = torch.zeros(self.height, self.width, dtype=torch.float64)
        for t in self.tiles:
            ys, xs = t.slices()
            s[ys, xs] += self.weight(t, kind, var, floor)
        return s

    def to_dict(self) -> dict:
        return {"height": self.height, "width": self.width, "tile_h": self.tile_h, "tile_w": self.tile_w,
                "min_overlap": self.min_overlap, "count": self.count, "grid": [len(self.rows), len(self.cols)],
                "rows": [[t.start, t.size] for t in self.rows], "cols": [[t.start, t.size] for t in self.cols],
                "realised_min_overlap": self.realised_min_overlap, "max_coverage": self.max_coverage}


def plan_tiles(height: int, width: int, tile_h: int, tile_w: int, min_overlap: int | None = None) -> TilePlan:
    """2-D latent tile plan. ``min_overlap`` (cells, both axes) defaults to half of each axis's tile, the LTX-2
    default; a requested overlap above half a tile is capped at half."""
    th, tw = min(tile_h, height), min(tile_w, width)
    ov_h = th // 2 if min_overlap is None else min(int(min_overlap), th // 2)
    ov_w = tw // 2 if min_overlap is None else min(int(min_overlap), tw // 2)
    rows = split_axis(height, th, ov_h) if th < height else [AxisTile(0, height, 0, 0)]
    cols = split_axis(width, tw, ov_w) if tw < width else [AxisTile(0, width, 0, 0)]
    return TilePlan(height, width, th, tw, min_overlap, tuple(rows), tuple(cols))


def overlap_cells(px_overlap: int, spatial: int = 32) -> int:
    """Pixels of overlap -> latent cells (rounded up so a requested overlap is never undershot)."""
    return math.ceil(max(0, px_overlap) / spatial)
