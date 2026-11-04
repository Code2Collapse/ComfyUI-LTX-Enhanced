"""HDR delivery: scene-linear EXR frames with source frame numbers, plus a JSON sidecar (owner A23 section 4.14, P1).

Model-independent and written from the ACES / OpenEXR definitions; OpenImageIO (Apache-2.0) writes the files and
is imported only when writing, so the pack loads without it.

- Frame accounting comes from the chunk plan: a window delivers source frames [keep_start, keep_end); the file
  number is ``first_frame_number + source frame`` (e.g. 1001 onwards), so segments never overwrite each other and
  the whole shot is one contiguous sequence.
- ``acescct`` decoded frames are converted with ``colorspace.acescct_decode`` (no clamp: the small negative footroom
  and highlights above 1.0 are kept; the sidecar records ``clamp: none``).
- Data written per frame: R, G, B, half (default) or float, ZIP (default) / PIZ / none - all lossless. Lossy DWAA is
  not offered (A23: never on data layers without approval; beauty-only DWAA is a later, explicit option).
- File names: ``<shot>_<layer>_<frame>.exr`` in ``<output>/<folder>/<layer>/`` (A23 folder-per-layer layout).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from fractions import Fraction

import torch

from . import colorspace

COMPRESSIONS = ("zip", "piz", "none")
OUTPUT_SPACES = ("acescg", "acescct")
PRECISIONS = ("half", "float")
# CIE xy of AP1 (ACEScg) primaries and the ACES white point: Rx Ry Gx Gy Bx By Wx Wy (EXR "chromaticities")
AP1_CHROMATICITIES = (0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767)
_NAME_OK = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class Delivery:
    index: int
    keep_offset: int        # first delivered frame inside the generated window
    keep_count: int         # delivered frames
    first_number: int       # file number of the first delivered frame
    last_number: int


def delivery_for(plan, index: int) -> Delivery:
    """Which frames of window ``index`` are delivered, and their file numbers."""
    segs = plan.segments
    if not 0 <= int(index) < len(segs):
        raise ValueError(f"The plan has windows 0-{len(segs) - 1}; index {index} does not exist.")
    s = segs[int(index)]
    first = int(plan.first_frame_number) + s.keep_start
    return Delivery(index=int(index), keep_offset=s.keep_start - s.gen_start, keep_count=s.keep_end - s.keep_start,
                    first_number=first, last_number=first + (s.keep_end - s.keep_start) - 1)


def safe_name(text: str, what: str) -> str:
    t = str(text or "").strip()
    if not t or not _NAME_OK.match(t) or t in (".", ".."):
        raise ValueError(f"{what} {text!r}: use letters, digits, '.', '_' or '-' only (no spaces, no slashes).")
    return t


def safe_folder(output_root: str, folder: str) -> str:
    """``folder`` inside ComfyUI's output directory; absolute paths and '..' are refused (D0.35: writes stay in the
    output root)."""
    rel = str(folder or "").strip().replace("\\", "/").strip("/")
    if not rel:
        raise ValueError("Give an output folder name (it is created inside ComfyUI's output folder).")
    if os.path.isabs(folder) or re.match(r"^[A-Za-z]:", rel) or any(p in ("", ".", "..") for p in rel.split("/")):
        raise ValueError(f"Output folder {folder!r} must be a plain relative folder inside ComfyUI's output folder "
                         "(no drive letter, no leading slash, no '..').")
    root = os.path.realpath(output_root)
    full = os.path.realpath(os.path.join(root, rel))
    if os.path.commonpath([root, full]) != root:
        raise ValueError(f"Output folder {folder!r} resolves outside ComfyUI's output folder.")
    return full


def frame_digits(last_number: int) -> int:
    return max(4, len(str(int(last_number))))


def shot_digits(plan) -> int:
    """Padding for the WHOLE shot (its last source frame), so every window uses the same file names."""
    return frame_digits(int(plan.first_frame_number) + int(plan.total_frames) - 1)


def read_exr(path: str):
    """One RGB EXR frame as a (H, W, 3) float32 array."""
    try:
        import OpenImageIO as oiio  # Apache-2.0; optional
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Reading EXR needs OpenImageIO (pip package 'OpenImageIO').") from exc
    inp = oiio.ImageInput.open(path)
    if inp is None:
        raise FileNotFoundError(f"Could not open {path}: {oiio.geterror()}")
    try:
        px = inp.read_image(oiio.FLOAT)
    finally:
        inp.close()
    return px[..., :3]


def frame_path(folder: str, shot: str, layer: str, number: int, digits: int) -> str:
    return os.path.join(folder, layer, f"{shot}_{layer}_{int(number):0{digits}d}.exr")


def to_output_space(frames: torch.Tensor, input_transfer: str, output_space: str) -> torch.Tensor:
    """Decoded frames -> the delivered encoding. ``acescct`` input is the LTX-2.5 HDR decode (ACEScct codes)."""
    x = frames[..., :3].float()        # RGB; an alpha channel is not part of the beauty layer
    if input_transfer == "acescct":
        if output_space == "acescct":
            return x
        return colorspace.acescct_decode(x)
    if input_transfer == "acescg":
        return x if output_space == "acescg" else colorspace.acescct_encode(x)
    raise ValueError(f"input_transfer {input_transfer!r}: use 'acescct' (LTX-2.5 HDR decode) or 'acescg'.")


def fps_fraction(fps_text: str) -> tuple[int, int]:
    f = Fraction(str(fps_text)).limit_denominator(1001) if fps_text not in (None, "", "0") else Fraction(0)
    return int(f.numerator), int(f.denominator or 1)


def write_exr(path: str, rgb, *, precision: str = "half", compression: str = "zip", output_space: str = "acescg",
              fps: tuple[int, int] = (0, 1), attrs: dict | None = None) -> None:
    """One RGB EXR frame (H, W, 3 float array). Raises with the reason on any failure."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision {precision!r}: use one of {PRECISIONS}.")
    if compression not in COMPRESSIONS:
        raise ValueError(f"compression {compression!r}: use one of {COMPRESSIONS} (all lossless).")
    try:
        import OpenImageIO as oiio  # Apache-2.0; optional
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Writing EXR needs OpenImageIO (pip package 'OpenImageIO'); it is not installed in this "
                           "Python.") from exc
    import numpy as np

    a = np.ascontiguousarray(np.asarray(rgb, dtype=np.float32))
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"write_exr expects (H, W, 3); got {a.shape}.")
    h, w = int(a.shape[0]), int(a.shape[1])
    spec = oiio.ImageSpec(w, h, 3, oiio.HALF if precision == "half" else oiio.FLOAT)
    spec.channelnames = ("R", "G", "B")
    spec.attribute("compression", compression)
    if output_space == "acescg":
        spec.attribute("chromaticities", oiio.TypeDesc("float[8]"), AP1_CHROMATICITIES)
        spec.attribute("oiio:ColorSpace", "ACEScg")
    else:
        spec.attribute("oiio:ColorSpace", "ACEScct")
    if fps[0] > 0:
        spec.attribute("FramesPerSecond", oiio.TypeDesc("rational"), (int(fps[0]), int(fps[1])))
    for k, v in (attrs or {}).items():
        spec.attribute(f"ltxe:{k}", str(v))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part.exr"
    out = oiio.ImageOutput.create(tmp)
    if out is None or not out.open(tmp, spec):
        raise RuntimeError(f"Could not open {tmp} for writing: {oiio.geterror()}")
    ok = out.write_image(a)
    out.close()
    if not ok:
        raise RuntimeError(f"Writing {tmp} failed: {oiio.geterror()}")
    os.replace(tmp, path)          # never leave a half-written frame under the real name


def fingerprint(frames: torch.Tensor) -> str:
    """Short content fingerprint of the delivered frames (sidecar)."""
    h = hashlib.sha1()
    h.update(str(tuple(frames.shape)).encode())
    step = max(1, int(frames.shape[0]) // 8)
    h.update(frames[::step].detach().cpu().contiguous().float().numpy().tobytes())
    return h.hexdigest()[:16]


def sidecar(*, plan, delivery: Delivery, shot: str, layer: str, output_space: str, precision: str, compression: str,
            input_transfer: str, paths: list, frames: torch.Tensor, source: dict | None = None) -> dict:
    return {
        "schema": 1,
        "shot": shot,
        "model_family": plan.model_family,
        "segment": delivery.index,
        "segments": len(plan.segments),
        "frame_range": [delivery.first_number, delivery.last_number],
        "shot_frame_range": [int(plan.first_frame_number), int(plan.first_frame_number) + int(plan.total_frames) - 1],
        "frames": delivery.keep_count,
        "source_fps": plan.source_fps,
        "conditioning_fps": plan.conditioning_fps,
        "output_fps": plan.source_fps,
        "resolution": [int(frames.shape[2]), int(frames.shape[1])],
        "layer": layer,
        "input_transfer": input_transfer,
        "output_color_space": output_space,
        "chromaticities": list(AP1_CHROMATICITIES) if output_space == "acescg" else None,
        "precision": precision,
        "compression": compression,
        "clamp": "none",
        "files": [os.path.basename(p) for p in paths],
        "content_fingerprint": fingerprint(frames),
        "source": source or {},
        "stability": "experimental",
    }


def write_sidecar(folder: str, shot: str, delivery: Delivery, data: dict) -> str:
    path = os.path.join(folder, "metadata", f"{shot}_seg{delivery.index:03d}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)
    return path
