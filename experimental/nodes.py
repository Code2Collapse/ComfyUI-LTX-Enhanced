"""ComfyUI nodes over the planning core. Stability label: experimental (docs/stability.md).

Imports ComfyUI only inside functions that run in ComfyUI, so the module can be imported (and its plan logic tested)
without core. Integration: core's APPLY_MODEL wrapper chain for tiling (docs/design/architecture.md); upstream
ComfyUI-LTXVideo is never imported (graph-level interop only, docs/licensing/register.md D-1).
"""
from __future__ import annotations

import json
import logging

import torch

from ..ltx_enhanced import padding
from ..ltx_enhanced.geometry import LTXV
from ..ltx_enhanced.plan import Plan, make_run_plan
from ..ltx_enhanced.schedule import check_sigmas
from ..ltx_enhanced.tiled_call import TiledCall, TileSettings, disabled

LOG = logging.getLogger("ltx_enhanced")
CATEGORY = "LTX Enhanced"
PLAN_TYPE = "LTXE_PLAN"
WRAPPER_KEY = "ltx_enhanced.tiles"
LABEL = "Experimental: implemented and tested on CPU, not yet benchmarked on real LTX weights."

TARGET_CHOICES = list(padding.TARGETS)


def _deliver(target: str, custom_w: int, custom_h: int, src_w: int, src_h: int) -> tuple[int, int]:
    if target == "custom":
        return int(custom_w), int(custom_h)
    size = padding.TARGETS[target]
    return size if size else (int(src_w), int(src_h))


def _outputs(plan: Plan):
    return (plan.to_dict(), plan.pad.sample_w, plan.pad.sample_h, plan.pad.sample_frames, plan.summary())


def _common_inputs() -> dict:
    return {
        "target": (TARGET_CHOICES, {"default": TARGET_CHOICES[0], "tooltip":
                   "Delivered frame size. The sampled canvas is that size rounded up to a multiple of 32 "
                   "(DCP 2K 1920x1080 is sampled at 1920x1088, DCP 4K 4096x2160 at 4096x2176)."}),
        "custom_width": ("INT", {"default": 1920, "min": 32, "max": 16384, "step": 1,
                                 "tooltip": "Delivered width when target is 'custom'."}),
        "custom_height": ("INT", {"default": 1080, "min": 32, "max": 16384, "step": 1,
                                  "tooltip": "Delivered height when target is 'custom'."}),
        "fit": (list(padding.FITS), {"default": "pad", "tooltip":
                "pad: place the source as it is (refuses a source larger than the target). fit-inside: scale "
                "down or up keeping the aspect ratio, then pad. crop-to-fill: scale to cover, crop the overflow."}),
        "pad_mode": (list(padding.PAD_MODES), {"default": "reflect", "tooltip":
                     "How the added border is filled: reflect (mirror the edge), replicate (repeat the edge "
                     "pixels) or constant (pad_value). Padding happens in pixels, before encoding."}),
        "pad_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                "tooltip": "Fill value for pad_mode 'constant'."}),
        "placement": (list(padding.PLACEMENTS), {"default": "center", "tooltip":
                      "center: split the border evenly (the odd pixel goes right or bottom). top-left: keep the "
                      "source at (0, 0)."}),
        "tile_width": ("INT", {"default": 1024, "min": 0, "max": 16384, "step": 32, "tooltip":
                       "Tile width in pixels for the plan summary and LTXE Tiled Model (0 = no tiling)."}),
        "tile_height": ("INT", {"default": 576, "min": 0, "max": 16384, "step": 32, "tooltip":
                        "Tile height in pixels (0 = no tiling)."}),
    }


class LTXEResolutionPlan:
    DESCRIPTION = ("Plans an LTX canvas for a size and length: aligned width, height and 8k+1 frame count, the "
                   "padding to remove afterwards, the tile grid and token counts. For text-to-video. " + LABEL)
    CATEGORY = f"{CATEGORY}/Planning"
    FUNCTION = "run"
    RETURN_TYPES = (PLAN_TYPE, "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("plan", "width", "height", "length", "summary")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width": ("INT", {"default": 1920, "min": 1, "max": 16384, "tooltip": "Source width in pixels."}),
            "height": ("INT", {"default": 1080, "min": 1, "max": 16384, "tooltip": "Source height in pixels."}),
            "frames": ("INT", {"default": 97, "min": 1, "max": 100000, "tooltip": "Frames to deliver."}),
            **_common_inputs()}}

    def run(self, width, height, frames, target, custom_width, custom_height, fit, pad_mode, pad_value, placement,
            tile_width, tile_height):
        dw, dh = _deliver(target, custom_width, custom_height, width, height)
        plan = make_run_plan(width, height, frames, dw, dh, tile_w_px=tile_width or None,
                             tile_h_px=tile_height or None, fit=fit, mode=pad_mode, value=pad_value,
                             placement=placement)
        out = _outputs(plan)
        return {"ui": {"text": [out[-1]]}, "result": out}


class LTXEPadFrames:
    DESCRIPTION = ("Pads source frames (and an optional mask) into the aligned LTX canvas without stretching or "
                   "cropping, and returns the plan that LTXE Crop Back uses to restore the exact source size and "
                   "length. " + LABEL)
    CATEGORY = f"{CATEGORY}/Planning"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MASK", PLAN_TYPE, "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "mask", "plan", "width", "height", "length", "summary")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE", {"tooltip": "Source frames."}), **_common_inputs()},
                "optional": {"mask": ("MASK", {"tooltip": "Optional mask; padded with the same transform "
                                                          "(edge replicated)."})}}

    def run(self, images, target, custom_width, custom_height, fit, pad_mode, pad_value, placement, tile_width,
            tile_height, mask=None):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        f, h, w = images.shape[:3]
        dw, dh = _deliver(target, custom_width, custom_height, w, h)
        plan = make_run_plan(w, h, f, dw, dh, tile_w_px=tile_width or None, tile_h_px=tile_height or None,
                             fit=fit, mode=pad_mode, value=pad_value, placement=placement)
        padded = padding.pad_images(images, plan.pad)
        if mask is None:
            out_mask = torch.zeros(padded.shape[:3], dtype=torch.float32)
        else:
            out_mask = padding.pad_masks(mask.float(), plan.pad)
        d, sw, sh, sf, text = _outputs(plan)
        return {"ui": {"text": [text]}, "result": (padded, out_mask, d, sw, sh, sf, text)}


class LTXECropBack:
    DESCRIPTION = ("Removes the padding an LTXE plan added: the extra frames at the end and the border, so the "
                   "frames come back at the source size (or the delivered frame). Works after a 2x upscale too. "
                   + LABEL)
    CATEGORY = f"{CATEGORY}/Planning"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "Decoded frames of the padded canvas (any uniform scale of it)."}),
            "plan": (PLAN_TYPE, {"tooltip": "The plan from LTXE Pad Frames or LTXE Resolution Plan."}),
            "region": (["content", "delivery"], {"default": "content", "tooltip":
                       "content: the source area. delivery: the full delivered frame (with any letterbox from "
                       "fit-inside)."})}}

    def run(self, images, plan, region):
        p = Plan.from_dict(plan if isinstance(plan, dict) else json.loads(plan))
        return (padding.crop_back(images, p.pad, region=region),)


class LTXETiledModel:
    DESCRIPTION = ("Makes every model call run tile by tile and blends the tiles' predictions with trapezoid weights, "
                   "while your sampler, noise and schedule stay as they are. Any sampler works, including ancestral "
                   "and multi-step ones. Audio in an LTX-2 audio-video latent is shared by all tiles. Not yet: IC-LoRA "
                   "guide or keyframe tokens, ControlNet. " + LABEL)
    CATEGORY = f"{CATEGORY}/Sampling"
    FUNCTION = "run"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "An LTX model (LTX-Video 0.9.x or LTX-2 / 2.3 / 2.5)."}),
            "tile_width": ("INT", {"default": 1024, "min": 32, "max": 16384, "step": 32, "tooltip":
                           "Tile width in pixels. Match the model's or the LoRA's trained window."}),
            "tile_height": ("INT", {"default": 576, "min": 32, "max": 16384, "step": 32,
                                    "tooltip": "Tile height in pixels."}),
            "min_overlap": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 32, "tooltip":
                            "Minimum overlap between tiles in pixels; 0 = half a tile (the LTX-2 default). "
                            "Capped at half a tile."}),
            "weight": (["trapezoid", "gaussian"], {"default": "trapezoid", "tooltip":
                       "trapezoid: flat inside, linear ramps over the overlaps (default). gaussian: experimental."}),
            "enabled": ("BOOLEAN", {"default": True, "tooltip": "Off = return the model unchanged."})}}

    def run(self, model, tile_width, tile_height, min_overlap, weight, enabled):
        if not enabled or disabled():
            if disabled():
                LOG.info("LTXE Tiled Model: LTXE_DISABLE_TILING is set; the model is returned unchanged")
            return (model,)
        import comfy.model_base
        import comfy.patcher_extension

        if not hasattr(comfy.patcher_extension, "WrappersMP") or not hasattr(comfy.patcher_extension.WrappersMP,
                                                                              "APPLY_MODEL"):
            raise RuntimeError("This ComfyUI is too old for LTXE Tiled Model: it has no APPLY_MODEL wrapper hook. "
                               "Update ComfyUI.")
        inner = getattr(model, "model", None)
        ltx = tuple(c for c in (getattr(comfy.model_base, "LTXV", None), getattr(comfy.model_base, "LTXAV", None)) if c)
        if not isinstance(inner, ltx):
            raise ValueError(f"LTXE Tiled Model needs an LTX model; this one is {type(inner).__name__}.")
        s = LTXV.spatial
        settings = TileSettings(tile_w=max(1, tile_width // s), tile_h=max(1, tile_height // s),
                                min_overlap=(-(-min_overlap // s) if min_overlap else None), weight=weight)
        call = TiledCall(settings)

        def wrapper(executor, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
            return call(executor, x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

        wrapper.ltxe_call = call      # the tiler's state (plan, call count), for tests and diagnostics
        m = model.clone()
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, WRAPPER_KEY, wrapper)
        return (m,)


class LTXESigmaCheck:
    DESCRIPTION = ("Passes a sigma schedule through and stops the run with a plain-English message when it is NaN, "
                   "rising, flat or saturated, the failures ComfyUI's LTX schedulers show on large latents. " + LABEL)
    CATEGORY = f"{CATEGORY}/Sampling"
    FUNCTION = "run"
    RETURN_TYPES = ("SIGMAS", "STRING")
    RETURN_NAMES = ("sigmas", "report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sigmas": ("SIGMAS",),
                             "stop_on_problem": ("BOOLEAN", {"default": True, "tooltip":
                                                 "On: stop the run when the schedule is unusable. Off: report only."})}}

    def run(self, sigmas, stop_on_problem):
        r = check_sigmas(sigmas.tolist() if torch.is_tensor(sigmas) else list(sigmas))
        text = (f"{r['steps']} steps; NaN {r['nan_count']}; rising {r['rising']}; steps at sigma >= 0.999: "
                f"{r['saturated_steps']}; flat {r['flat']}. {r['message']}")
        if not r["ok"] and stop_on_problem:
            raise ValueError(r["message"])
        return {"ui": {"text": [text]}, "result": (sigmas, text)}


NODE_CLASS_MAPPINGS = {
    "LTXE_ResolutionPlan": LTXEResolutionPlan,
    "LTXE_PadFrames": LTXEPadFrames,
    "LTXE_CropBack": LTXECropBack,
    "LTXE_TiledModel": LTXETiledModel,
    "LTXE_SigmaCheck": LTXESigmaCheck,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXE_ResolutionPlan": "LTXE Resolution Plan (experimental)",
    "LTXE_PadFrames": "LTXE Pad Frames (experimental)",
    "LTXE_CropBack": "LTXE Crop Back (experimental)",
    "LTXE_TiledModel": "LTXE Tiled Model (experimental)",
    "LTXE_SigmaCheck": "LTXE Sigma Check (experimental)",
}
