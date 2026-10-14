"""Evaluate one model call tile by tile and blend the x0 predictions (no ComfyUI import).

This is the body of the APPLY_MODEL wrapper (docs/design/architecture.md, "LTXE Tiled Model"). The architecture
follows the behaviour described in docs/re-notes/ltx2-official-tiled-diffusion.md: the sampler keeps one
full-canvas latent and its own loop; each model call is split into spatial tiles, each tile's denoised (x0)
prediction is blended with trapezoid weights, audio (when the latent is a packed audio-video latent) is shared by
every tile and averaged.

The caller passes ``executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs) -> x0`` (ComfyUI
core's next APPLY_MODEL executor) and the keyword arguments core hands to the model.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Callable

import torch

from .blend import TileAccumulator
from .tiles import TilePlan, plan_tiles

# Keyword arguments the LTX models receive at apply_model (comfy/model_base.py LTXV / LTXAV extra_conds).
SHARED_KEYS = frozenset({"frame_rate", "attention_mask", "audio_denoise_mask", "ref_audio"})
CROPPED_KEYS = frozenset({"denoise_mask"})
PACKING_KEY = "latent_shapes"
GUIDE_KEYS = ("keyframe_idxs", "guide_attention_entries", "generated_keyframes")


class TilingUnsupported(ValueError):
    """Raised with a plain-English reason when a call cannot be tiled safely."""


@dataclass(frozen=True)
class TileSettings:
    tile_w: int                 # latent cells
    tile_h: int
    min_overlap: int | None = None
    weight: str = "trapezoid"
    var: float = 0.05
    floor: float = 0.0


def disabled() -> bool:
    return os.environ.get("LTXE_DISABLE_TILING", "").strip() not in ("", "0")


def pack(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Size]]:
    """Flatten [B, ...] tensors into one [B, 1, N] latent (the packed audio-video layout)."""
    shapes = [t.shape for t in tensors]
    return torch.cat([t.reshape(t.shape[0], 1, -1) for t in tensors], dim=-1), shapes


def unpack(packed: torch.Tensor, shapes: list) -> list[torch.Tensor]:
    out, rest = [], packed
    for s in shapes:
        n = math.prod(s[1:])
        out.append(rest[:, :, :n].reshape([rest.shape[0]] + list(s)[1:]))
        rest = rest[:, :, n:]
    return out


def _has_guides(kwargs: dict) -> bool:
    kf = kwargs.get("keyframe_idxs")
    if torch.is_tensor(kf) and kf.ndim >= 3 and kf.shape[2] > 0:
        return True
    return bool(kwargs.get("guide_attention_entries")) or kwargs.get("generated_keyframes") is not None


def _check_kwargs(kwargs: dict, c_concat, control) -> None:
    if _has_guides(kwargs):
        raise TilingUnsupported(
            "LTXE Tiled Model does not tile guide or keyframe tokens yet (IC-LoRA guides, LTXV Add Guide). "
            "Use it for text-to-video, image-to-video with an in-place first frame, or a low-denoise refine of an "
            "upscaled latent; for IC-LoRA guides use Lightricks' LTXV Tiled Fusion Sampler.")
    if control is not None:
        raise TilingUnsupported("LTXE Tiled Model cannot tile a ControlNet; remove the ControlNet or disable tiling.")
    if c_concat is not None:
        raise TilingUnsupported("LTXE Tiled Model cannot tile concat conditioning (c_concat) yet.")
    unknown = [k for k in kwargs if k not in SHARED_KEYS | CROPPED_KEYS | {PACKING_KEY, *GUIDE_KEYS}]
    if unknown:
        raise TilingUnsupported(
            f"LTXE Tiled Model got model inputs it does not know how to tile: {', '.join(sorted(unknown))}. "
            "Tiling them blindly could feed whole-frame data into one tile, so it stops here. Disable tiling, or "
            "report the node that adds them.")


def _crop(name: str, t: Any, ys: slice, xs: slice, h: int, w: int) -> Any:
    """Crop a per-cell model input to the tile; refuse one whose grid does not match the latent."""
    if t is None:
        return None
    if torch.is_tensor(t) and t.ndim >= 2 and tuple(t.shape[-2:]) == (h, w):
        return t[..., ys, xs]
    shape = tuple(t.shape) if torch.is_tensor(t) else type(t).__name__
    raise TilingUnsupported(f"LTXE Tiled Model cannot tile '{name}': its shape {shape} does not match the "
                            f"latent grid {h}x{w}.")


class TiledCall:
    """Holds the settings and a small plan cache; ``__call__`` runs one tiled model evaluation."""

    def __init__(self, settings: TileSettings):
        self.settings = settings
        self._plans: dict[tuple[int, int], TilePlan] = {}
        self.calls = 0            # model evaluations made (tiles), for tests and the log line
        self.last_plan: TilePlan | None = None

    def plan_for(self, h: int, w: int) -> TilePlan:
        p = self._plans.get((h, w))
        if p is None:
            s = self.settings
            p = plan_tiles(h, w, s.tile_h, s.tile_w, s.min_overlap)
            self._plans[(h, w)] = p
        return p

    def __call__(self, executor: Callable, x: torch.Tensor, t, c_concat=None, c_crossattn=None, control=None,
                 transformer_options: dict | None = None, **kwargs) -> torch.Tensor:
        transformer_options = {} if transformer_options is None else transformer_options
        if disabled():
            return executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

        shapes = kwargs.get(PACKING_KEY)
        packed = shapes is not None and len(shapes) > 1
        if packed:
            parts = unpack(x, shapes)
            video, others = parts[0], parts[1:]
        else:
            video, others = x, []
        if video.ndim != 5:
            raise TilingUnsupported(f"LTXE Tiled Model expects a 5-D video latent [batch, channels, frames, "
                                    f"height, width]; got shape {tuple(video.shape)}.")
        h, w = video.shape[-2:]
        plan = self.plan_for(h, w)
        self.last_plan = plan
        if plan.count == 1:
            self.calls += 1
            return executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
        _check_kwargs(kwargs, c_concat, control)

        s = self.settings
        acc = None
        other_sums: list[torch.Tensor] | None = None
        out_dtype = None
        for i, tile in enumerate(plan.tiles):
            ys, xs = tile.slices()
            v_tile = video[..., ys, xs]
            kw = {k: (_crop(k, v, ys, xs, h, w) if k in CROPPED_KEYS else v) for k, v in kwargs.items()}
            if packed:
                x_tile, tile_shapes = pack([v_tile, *others])
                kw[PACKING_KEY] = tile_shapes
            else:
                x_tile = v_tile
            opts = dict(transformer_options)
            opts["ltxe_tile"] = (i, plan.count)
            out = executor(x_tile, t, c_concat, c_crossattn, control, opts, **kw)
            self.calls += 1
            out_dtype = out.dtype
            if packed:
                outs = unpack(out, kw[PACKING_KEY])
                v_out, o_outs = outs[0], outs[1:]
            else:
                v_out, o_outs = out, []
            if acc is None:
                acc = TileAccumulator(video.shape[:-2] + (h, w), plan, device=v_out.device, dtype=torch.float32,
                                      kind=s.weight, var=s.var, floor=s.floor)
                other_sums = [torch.zeros_like(o, dtype=torch.float32) for o in o_outs]
            acc.add(tile, v_out)
            for k, o in enumerate(o_outs):
                other_sums[k] += o.to(torch.float32)
        video_x0 = acc.result().to(out_dtype)
        if not packed:
            return video_x0
        others_x0 = [(o / plan.count).to(out_dtype) for o in other_sums]
        return pack([video_x0, *others_x0])[0]
