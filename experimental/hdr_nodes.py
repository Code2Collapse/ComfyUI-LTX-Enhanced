"""SDR -> HDR helper nodes (owner A23). Stability label: experimental (docs/stability.md, docs/HDR_ARCHITECTURE.md).

These nodes do the planning, auditing and colour dispatch the official LTX-2.5 SDR-to-HDR graph lacks. The model work
itself (scene embeddings, IC-LoRA, guide, sampling, float32 decode, EXR, HLG) stays with the official ComfyUI-LTXVideo
and ComfyUI core nodes, wired in the example workflows; this module never imports upstream code (register D-1).
ComfyUI is imported only inside run() so the pack still loads without it. Kill switch: LTXE_DISABLE_HDR=1.
"""
from __future__ import annotations

import json
import os

import torch

from ..ltx_enhanced.hdr import chunk_plan, colorspace, exr_writer, source_audit

CATEGORY = "LTX Enhanced/HDR"
PLAN_TYPE = "LTXE_HDR_PLAN"
LABEL = ("Experimental: implemented and tested on CPU; the HDR model run itself is the official LTX-2.5 graph and "
         "has not been benchmarked by this pack.")


class LTXEHDRSourceAudit:
    DESCRIPTION = ("Checks an SDR clip before any model is loaded: size, frame rate, bit depth (estimated from the "
                   "values), clipped highlights, near-white areas, crushed shadows, duplicate frames, likely cuts, "
                   "softness, an upscale hint, and whether the input is already HDR (then use the Native HDR "
                   "workflow). " + LABEL)
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "BOOLEAN", "INT")
    RETURN_NAMES = ("report", "report_json", "bit_depth", "already_hdr", "frames")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        d = source_audit.AuditSettings()
        return {"required": {
            "images": ("IMAGE", {"tooltip": "The SDR frames, as loaded (display-referred 0-1)."}),
            "fps": ("FLOAT", {"default": 24.0, "min": 0.0, "max": 1000.0, "step": 0.001,
                              "tooltip": "The source frame rate (0 = not known)."}),
            "scan_every": ("INT", {"default": d.scan_every, "min": 1, "max": 1000,
                                   "tooltip": "Look at every n-th frame (1 = all; duplicates are only checked at 1)."}),
            "clip_level": ("FLOAT", {"default": d.clip_level, "min": 0.5, "max": 1.0, "step": 0.001,
                                     "tooltip": "A channel at or above this value counts as clipped."}),
            "crush_level": ("FLOAT", {"default": d.crush_level, "min": 0.0, "max": 0.2, "step": 0.001,
                                      "tooltip": "Luma at or below this value counts as crushed to black."}),
        }, "optional": {
            "metadata_json": ("STRING", {"default": "", "multiline": True, "tooltip": (
                "Optional container facts as JSON, e.g. {\"primaries\": \"bt709\", \"transfer\": \"bt709\", "
                "\"range\": \"tv\", \"pix_fmt\": \"yuv420p\"}. Reported as given.")}),
        }}

    def run(self, images, fps, scan_every, clip_level, crush_level, metadata_json=""):
        try:
            md = json.loads(metadata_json) if str(metadata_json or "").strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"metadata_json is not valid JSON ({exc.msg}); leave it empty or fix it.") from exc
        s = source_audit.AuditSettings(scan_every=scan_every, clip_level=clip_level, crush_level=crush_level)
        r = source_audit.audit(images, fps=fps, metadata=md, settings=s)
        text = source_audit.report_text(r)
        return {"ui": {"text": [text]},
                "result": (text, json.dumps(r, indent=1), r["bit_depth_estimate"], bool(r["already_hdr"]), r["frames"])}


class LTXEHDRInputColor:
    DESCRIPTION = ("Maps the source into ACEScct, the working space of the LTX-2.5 SDR-to-HDR IC-LoRA, before any "
                   "resize. srgb_gamma (ordinary video) and srgb (scene-linear Rec.709) use ComfyUI core's Convert "
                   "Image Color Space, the node the official graph uses; acescg uses this pack's ACEScct curve (core "
                   "has no ACEScg input); acescct passes through. The untouched source is passed on for later "
                   "source-preserving compositing. " + LABEL)
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("acescct", "source", "transform")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "Source frames at their own size (convert first, resize after)."}),
            "input_space": (list(colorspace.INPUT_SPACES), {"default": "srgb_gamma", "tooltip": (
                "srgb_gamma: ordinary display-encoded video. srgb: scene-linear Rec.709 / sRGB. acescg: linear AP1. "
                "acescct: already ACEScct (passed through).")}),
        }}

    def run(self, images, input_space):
        if input_space not in colorspace.INPUT_SPACES:
            raise ValueError(f"Unknown input_space '{input_space}'; choose one of {', '.join(colorspace.INPUT_SPACES)}.")
        if input_space == "srgb_gamma" and float(images[..., :3].max()) > 1.0 + 1e-3:
            raise ValueError("The frames go above 1.0, so they are not display-encoded sRGB video. Choose srgb "
                             "(scene-linear) or acescg, or use the Native HDR workflow for HDR material.")
        if input_space == "acescct":
            out = images
        elif input_space == "acescg":
            out = colorspace.acescg_to_acescct(images)
        else:
            from comfy_extras.nodes_images import ImageColorSpace   # core, imported only when it runs
            res = ImageColorSpace.execute(images, colorspace.CORE_SOURCE_NAME[input_space], "HDR ACEScct")
            out = (res.result if hasattr(res, "result") else res)[0]
        note = {"input_space": input_space, "working_space": "ACEScct",
                "by": "passthrough" if input_space == "acescct" else
                      "ltx_enhanced ACEScct curve" if input_space == "acescg" else "ComfyUI core ImageColorSpace"}
        return (out, images, json.dumps(note))


class LTXEHDRChunkPlanner:
    DESCRIPTION = ("Plans the temporal windows for SDR -> HDR: 8k+1-frame windows that overlap, a seam in the middle "
                   "of each overlap, delivered ranges that cover the clip exactly once, padding of the last window "
                   "(discarded later), conditioning at min(source, 30) fps while the output keeps the source rate, "
                   "and token / memory estimates. " + LABEL)
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = (PLAN_TYPE, "INT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("plan", "segments", "conditioning_fps", "output_fps", "summary")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames": ("INT", {"default": 121, "min": 1, "max": 1_000_000, "tooltip": "Frames in the source clip (ignored when images is connected)."}),
            "fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 1000.0, "step": 0.001, "tooltip": (
                "Source frame rate (23.976 / 29.97 / 59.94 are read as the NTSC fractions).")}),
            "width": ("INT", {"default": 1920, "min": 32, "max": 16384, "tooltip": "Working width in pixels (ignored when images is connected)."}),
            "height": ("INT", {"default": 1080, "min": 32, "max": 16384, "tooltip": "Working height in pixels (ignored when images is connected)."}),
            "segment_frames": ("INT", {"default": 97, "min": 9, "max": 1001, "step": 8, "tooltip": (
                "Frames per window, 8k+1 (97 is the official SDR-to-HDR graphs' window).")}),
            "overlap": ("INT", {"default": 16, "min": 0, "max": 992, "tooltip": (
                "Frames shared by neighbouring windows; the seam sits in the middle. Not yet tuned on real runs.")}),
            "first_frame_number": ("INT", {"default": 0, "min": 0, "max": 10_000_000, "tooltip": (
                "Number of the first source frame (e.g. 1001), kept for the delivered frames.")}),
        }, "optional": {"images": ("IMAGE", {"tooltip": "Optional: frames, width and height are read from these."})}}

    def run(self, frames, fps, width, height, segment_frames, overlap, first_frame_number, images=None):
        if images is not None:
            frames, height, width = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
        plan = chunk_plan.plan_segments(frames, fps, width, height, segment_frames, overlap, first_frame_number)
        from fractions import Fraction
        text = plan.summary()
        return {"ui": {"text": [text]}, "result": (plan.to_dict(), len(plan.segments),
                                                    float(Fraction(plan.conditioning_fps)),
                                                    float(Fraction(plan.source_fps)), text)}


class LTXEHDRTakeSegment:
    DESCRIPTION = ("Cuts one planned window out of the source frames, padded to its 8k+1 length with copies of the "
                   "last frame, and says which of its frames are delivered (the rest is seam context or padding). "
                   + LABEL)
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("window", "keep_offset", "keep_count", "frames")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "The whole source clip (the same frames the plan was made for)."}),
            "plan": (PLAN_TYPE, {"tooltip": "From LTXE HDR Chunk Planner."}),
            "index": ("INT", {"default": 0, "min": 0, "max": 100_000, "tooltip": "Which window (0 = the first)."}),
        }}

    def run(self, images, plan, index):
        p = chunk_plan.SegmentPlan.from_dict(plan if isinstance(plan, dict) else json.loads(plan))
        if int(images.shape[0]) != p.total_frames:
            raise ValueError(f"The plan is for {p.total_frames} frames but {int(images.shape[0])} came in; plan this "
                             "clip again.")
        if not 0 <= index < len(p.segments):
            raise ValueError(f"The plan has windows 0-{len(p.segments) - 1}; index {index} does not exist.")
        s = p.segments[index]
        win = images[s.gen_start:s.gen_end]
        if s.pad_tail:
            win = torch.cat([win, images[s.gen_end - 1:s.gen_end].expand(s.pad_tail, *images.shape[1:])], dim=0)
        return (win.contiguous(), s.keep_start - s.gen_start, s.keep_end - s.keep_start, s.gen_frames)


class LTXEHDRDelivery:
    DESCRIPTION = ("Writes one decoded window as scene-linear EXR frames named after the source frames "
                   "(<shot>_hdr_1001.exr ...): only the frames the plan delivers (seam context and padding dropped), "
                   "converted from the ACEScct decode to ACEScg without clamping, half float + ZIP by default, plus a "
                   "JSON sidecar in metadata/. Files go into ComfyUI's output folder. " + LABEL)
    CATEGORY = CATEGORY
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("delivered", "first_frame", "last_frame", "summary")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "The decoded window (VAE Decode output: ACEScct codes, 0-1), every "
                                            "generated frame."}),
            "plan": (PLAN_TYPE, {"tooltip": "From LTXE HDR Chunk Planner (the same plan the window was cut with)."}),
            "index": ("INT", {"default": 0, "min": 0, "max": 100_000,
                              "tooltip": "The window index (the same as on LTXE HDR Take Segment)."}),
            "shot_name": ("STRING", {"default": "shot", "tooltip": "File name stem: <shot>_hdr_<frame>.exr."}),
            "output_folder": ("STRING", {"default": "ltxe_hdr", "tooltip": (
                "Folder inside ComfyUI's output folder; layers go in sub-folders (hdr/, metadata/).")}),
            "input_transfer": (("acescct", "acescg"), {"default": "acescct", "tooltip": (
                "acescct = the LTX-2.5 HDR VAE decode. acescg = frames that are already scene-linear ACEScg.")}),
            "output_space": (exr_writer.OUTPUT_SPACES, {"default": "acescg", "tooltip": (
                "acescg = scene-linear ACEScg (AP1), for comp and grade. acescct = the log codes.")}),
            "precision": (exr_writer.PRECISIONS, {"default": "half", "tooltip": "half = 16-bit float; float = 32."}),
            "compression": (exr_writer.COMPRESSIONS, {"default": "zip", "tooltip": "All lossless."}),
        }, "optional": {
            "source_report_json": ("STRING", {"forceInput": True, "tooltip": (
                "Optional: LTXE HDR Source Audit report_json, copied into the sidecar.")}),
        }}

    def run(self, images, plan, index, shot_name, output_folder, input_transfer, output_space, precision,
            compression, source_report_json=None):
        import folder_paths  # ComfyUI; imported here so the module loads without it

        p = chunk_plan.SegmentPlan.from_dict(plan if isinstance(plan, dict) else json.loads(plan))
        d = exr_writer.delivery_for(p, index)
        seg = p.segments[d.index]
        if int(images.shape[0]) != seg.gen_frames:
            raise ValueError(f"Window {d.index} was planned as {seg.gen_frames} frames but {int(images.shape[0])} "
                             "came in. Feed the decode of THIS window (same plan, same index).")
        shot = exr_writer.safe_name(shot_name, "shot_name")
        folder = exr_writer.safe_folder(folder_paths.get_output_directory(), output_folder)
        kept = images[d.keep_offset:d.keep_offset + d.keep_count]
        out = exr_writer.to_output_space(kept, input_transfer, output_space)
        digits = exr_writer.frame_digits(d.last_number)
        fps = exr_writer.fps_fraction(p.source_fps)
        try:
            src = json.loads(source_report_json) if str(source_report_json or "").strip() else {}
        except json.JSONDecodeError:
            src = {"note": "source_report_json was not valid JSON"}
        paths = []
        arr = out.detach().cpu().float().numpy()
        for i in range(d.keep_count):
            path = exr_writer.frame_path(folder, shot, "hdr", d.first_number + i, digits)
            exr_writer.write_exr(path, arr[i], precision=precision, compression=compression,
                                 output_space=output_space, fps=fps,
                                 attrs={"shot": shot, "segment": d.index, "frame": d.first_number + i,
                                        "model_family": p.model_family})
            paths.append(path)
        side = exr_writer.write_sidecar(folder, shot, d, exr_writer.sidecar(
            plan=p, delivery=d, shot=shot, layer="hdr", output_space=output_space, precision=precision,
            compression=compression, input_transfer=input_transfer, paths=paths, frames=out, source=src))
        summary = (f"window {d.index + 1} of {len(p.segments)}: wrote {d.keep_count} {output_space} {precision} "
                   f"{compression} EXR frames {d.first_number}-{d.last_number} to "
                   f"{os.path.relpath(os.path.dirname(paths[0]), folder_paths.get_output_directory())} "
                   f"(sidecar {os.path.basename(side)}).")
        return {"ui": {"text": [summary]}, "result": (out.contiguous(), d.first_number, d.last_number, summary)}


NODE_CLASS_MAPPINGS = {
    "LTXE_HDRSourceAudit": LTXEHDRSourceAudit,
    "LTXE_HDRInputColor": LTXEHDRInputColor,
    "LTXE_HDRChunkPlanner": LTXEHDRChunkPlanner,
    "LTXE_HDRTakeSegment": LTXEHDRTakeSegment,
    "LTXE_HDRDelivery": LTXEHDRDelivery,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXE_HDRSourceAudit": "LTXE HDR Source Audit (experimental)",
    "LTXE_HDRInputColor": "LTXE HDR Input Color (experimental)",
    "LTXE_HDRChunkPlanner": "LTXE HDR Chunk Planner (experimental)",
    "LTXE_HDRTakeSegment": "LTXE HDR Take Segment (experimental)",
    "LTXE_HDRDelivery": "LTXE HDR Delivery (experimental)",
}
