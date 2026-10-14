"""Accumulate per-tile predictions into one canvas and normalise by the summed weights.

Tiles are cut along the last two dimensions (latent rows, columns); every leading dimension (batch, channels,
frames) is carried whole. Blending x0 predictions with weights that are normalised to sum to 1 is the convex
combination MultiDiffusion solves for (arXiv 2302.08113).
"""
from __future__ import annotations

import torch

from .tiles import Tile2D, TilePlan


class TileAccumulator:
    def __init__(self, shape: torch.Size | tuple, plan: TilePlan, device=None, dtype=torch.float32,
                 kind: str = "trapezoid", var: float = 0.05, floor: float = 0.0):
        if tuple(shape[-2:]) != (plan.height, plan.width):
            raise ValueError(f"canvas {tuple(shape[-2:])} does not match the tile plan {(plan.height, plan.width)}")
        self.plan = plan
        self.kind, self.var, self.floor = kind, var, floor
        self.acc = torch.zeros(shape, device=device, dtype=dtype)
        self.wsum = torch.zeros(shape[-2:], device=device, dtype=dtype)
        self._weights: dict[tuple, torch.Tensor] = {}

    def weight(self, tile: Tile2D) -> torch.Tensor:
        key = (tile.y, tile.x)
        w = self._weights.get(key)
        if w is None:
            w = self.plan.weight(tile, self.kind, self.var, self.floor).to(self.acc.device, self.acc.dtype)
            self._weights[key] = w
        return w

    def add(self, tile: Tile2D, value: torch.Tensor) -> None:
        ys, xs = tile.slices()
        if tuple(value.shape[-2:]) != (tile.y.size, tile.x.size):
            raise ValueError(f"tile output {tuple(value.shape[-2:])} does not match the tile {(tile.y.size, tile.x.size)}")
        w = self.weight(tile)
        self.acc[..., ys, xs] += value.to(self.acc.dtype) * w
        self.wsum[ys, xs] += w

    def result(self) -> torch.Tensor:
        if bool((self.wsum <= 0).any()):
            raise RuntimeError("Some latent cells were not covered by any tile; the tile plan has a gap.")
        return self.acc / self.wsum
