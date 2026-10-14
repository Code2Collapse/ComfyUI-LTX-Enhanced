"""Crash repro 001 (ledger L7.106): run core's LTXVScheduler and ModelSamplingLTXV (CPU, no weights) on empty latents of several sizes.
Question: does the token-count shift degenerate at 1080p / 4K, and what does a tile-sized latent give instead?
Usage: python -I ltx_sched_probe.py <ComfyUI core dir> <out.json>"""
import json
import math
import sys
from pathlib import Path

CORE = Path(sys.argv[1])
OUT = Path(sys.argv[2])
sys.path.insert(0, str(CORE))
sys.argv = [sys.argv[0], "--cpu"]
import torch  # noqa: E402
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import comfy.samplers  # noqa: E402
from comfy_extras import nodes_lt  # noqa: E402

CASES = [  # name, width, height, frames
    ("768x512 (LTX reference)", 768, 512, 97),
    ("1280x704 HD", 1280, 704, 97),
    ("1920x1088 DCP 2K", 1920, 1088, 97),
    ("2560x1440 QHD", 2560, 1440, 97),
    ("3840x2176 UHD", 3840, 2176, 97),
    ("4096x2176 DCP 4K", 4096, 2176, 97),
    ("4096x2176 DCP 4K, 9 frames", 4096, 2176, 9),
    ("tile 1024x576", 1024, 576, 97),
]


class _Cfg:
    sampling_settings: dict = {}


class _Inner:
    model_config = _Cfg()


class _FakeModel:
    """Just enough of a ModelPatcher for ModelSamplingLTXV.execute (clone + add_object_patch)."""
    def __init__(self):
        self.model = _Inner()
        self.patches = {}

    def clone(self):
        m = _FakeModel()
        m.patches = dict(self.patches)
        return m

    def add_object_patch(self, name, obj):
        self.patches[name] = obj


def _r(t):
    return [None if math.isnan(float(x)) else round(float(x), 6) for x in t]


ver = (CORE / "comfyui_version.py").read_text(encoding="utf-8").strip().splitlines()[-1]
res = {"core_version": ver, "scheduler": {"steps": 20, "max_shift": 2.05, "base_shift": 0.95,
       "stretch": True, "terminal": 0.1}, "model_sampling": {"max_shift": 2.05, "base_shift": 0.95,
       "scheduler": "simple", "steps": 20}, "cases": []}
for name, w, h, f in CASES:
    lat = {"samples": torch.zeros(1, 128, (f - 1) // 8 + 1, h // 32, w // 32)}
    tokens = math.prod(lat["samples"].shape[2:])
    sig = nodes_lt.LTXVScheduler.execute(20, 2.05, 0.95, True, 0.1, latent=lat)[0]
    m = nodes_lt.ModelSamplingLTXV.execute(_FakeModel(), 2.05, 0.95, latent=lat)[0]
    ms_sig = comfy.samplers.calculate_sigmas(m.patches["model_sampling"], "simple", 20)
    row = {"name": name, "latent_thw": list(lat["samples"].shape[2:]), "tokens": tokens,
           "shift": round(tokens * (2.05 - 0.95) / 3072 + 0.95 - 1024 * (2.05 - 0.95) / 3072, 3),
           "ltxv_scheduler_sigmas": _r(sig), "ltxv_scheduler_nan": int(torch.isnan(sig).sum()),
           "model_sampling_simple_sigmas": _r(ms_sig), "model_sampling_nan": int(torch.isnan(ms_sig).sum())}
    res["cases"].append(row)
    print(f"{name:28} tok={tokens:7d} shift={row['shift']:6.2f} | LTXVScheduler {_r(sig)[1:4]} nan={row['ltxv_scheduler_nan']}"
          f" | ModelSamplingLTXV+simple {_r(ms_sig)[1:4]} nan={row['model_sampling_nan']}")
OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
print("core", ver)
