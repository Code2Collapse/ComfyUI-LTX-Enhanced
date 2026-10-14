"""Latent geometry of LTX video models: how pixels and frames map to latent cells.

Facts from the LTX-Video and LTX-2 papers (arXiv 2501.00103, 2601.03233) and from ComfyUI core's latent format
(docs/re-notes/core-ltx-geometry-and-schedule.md): 32 pixels per latent cell in each spatial axis, 8 pixel frames
per latent frame after a causal first frame that encodes one pixel frame, 128 latent channels.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LatentGeometry:
    spatial: int = 32       # pixels per latent cell (height and width)
    temporal: int = 8       # pixel frames per latent frame after the first
    channels: int = 128
    causal_first: bool = True   # latent frame 0 encodes exactly one pixel frame

    def aligned_size(self, px: int) -> int:
        """Smallest multiple of the spatial factor that holds ``px`` pixels."""
        if px < 1:
            raise ValueError(f"a size must be at least 1 pixel, got {px}")
        return -(-int(px) // self.spatial) * self.spatial

    def latent_size(self, px: int) -> int:
        """Latent cells for an aligned pixel size."""
        if px % self.spatial:
            raise ValueError(f"{px} px is not a multiple of {self.spatial}; align it first")
        return px // self.spatial

    def latent_frames(self, frames: int) -> int:
        """Latent frames for a frame count of the form 8k+1."""
        if frames < 1:
            raise ValueError(f"a clip needs at least 1 frame, got {frames}")
        if self.causal_first and (frames - 1) % self.temporal:
            raise ValueError(f"{frames} frames is not {self.temporal}k+1; align it first")
        return (frames - 1) // self.temporal + 1 if self.causal_first else -(-frames // self.temporal)

    def aligned_frames(self, frames: int) -> int:
        """Smallest frame count >= ``frames`` that the VAE encodes without dropping frames (8k+1)."""
        if frames < 1:
            raise ValueError(f"a clip needs at least 1 frame, got {frames}")
        if not self.causal_first:
            return -(-frames // self.temporal) * self.temporal
        return -(-(frames - 1) // self.temporal) * self.temporal + 1

    def pixel_frames(self, latent_frames: int) -> int:
        """Pixel frames decoded from ``latent_frames`` latent frames."""
        if latent_frames < 1:
            raise ValueError(f"need at least 1 latent frame, got {latent_frames}")
        return (latent_frames - 1) * self.temporal + 1 if self.causal_first else latent_frames * self.temporal


LTXV = LatentGeometry()     # LTX-Video 0.9.x
LTXAV = LatentGeometry()    # LTX-2 / 2.3 / 2.5: same video-stream geometry; audio is a separate stream
PRESETS = {"LTXV": LTXV, "LTXAV": LTXAV}
