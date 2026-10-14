"""Integration probe (run by tests/test_core_integration.py in a clean subprocess; not a test module itself).

Builds tiny, randomly initialised core LTX models (LTX-Video LTXV and LTX-2 LTXAV) on CPU, samples them through
ComfyUI core's own sampler untiled, with a one-tile LTXE Tiled Model and with a multi-tile one, and prints JSON.
A random tiny model says nothing about image quality; this checks that the wrapper registers through core's
APPLY_MODEL chain, that the tiled path runs for both latent layouts, and that one tile is exactly a plain call.

    python -I tests/core_probe.py <ComfyUI core dir>
"""
import json
import sys
from pathlib import Path

CORE = Path(sys.argv[1])
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(TESTS))
sys.argv = [sys.argv[0], "--cpu", "--use-pytorch-cross-attention"]
import torch  # noqa: E402
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
import comfy.model_base  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.patcher_extension  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.supported_models  # noqa: E402
from _pack import load_pack  # noqa: E402

pack = load_pack()
nodes = sys.modules[pack.__name__ + ".experimental.nodes"]
CPU = torch.device("cpu")
STEPS = 3


def _patcher(kind):
    torch.manual_seed(0)
    cfg = {"in_channels": 128, "cross_attention_dim": 64, "attention_head_dim": 16, "num_attention_heads": 4,
           "caption_channels": 32, "num_layers": 2}
    if kind == "ltxav":
        cfg.update(image_model="ltxav", audio_in_channels=128, audio_cross_attention_dim=32,
                   audio_attention_head_dim=8, audio_num_attention_heads=4)
        mc = comfy.supported_models.LTXAV(cfg)
    else:
        cfg.update(image_model="ltxv")
        mc = comfy.supported_models.LTXV(cfg)
    mc.set_inference_dtype(torch.float32, None)
    model = (comfy.model_base.LTXAV if kind == "ltxav" else comfy.model_base.LTXV)(mc, device=CPU)
    for p in model.diffusion_model.parameters():
        torch.nn.init.normal_(p, std=0.02)
    return comfy.model_patcher.ModelPatcher(model, load_device=CPU, offload_device=CPU)


def _inputs(kind):
    g = lambda s: torch.Generator().manual_seed(s)  # noqa: E731
    video = torch.zeros(1, 128, 2, 8, 12)
    width = 64 if kind == "ltxav" else 32    # LTX-2 takes an already-processed context of 2 x caption_channels
    ctx = torch.randn(1, 7, width, generator=g(3))
    pos = [[ctx, {"attention_mask": torch.ones(1, 7)}]]
    neg = [[torch.zeros_like(ctx), {"attention_mask": torch.ones(1, 7)}]]
    if kind == "ltxav":
        import comfy.nested_tensor as nt
        audio = torch.zeros(1, 8, 9, 16)
        lat = nt.NestedTensor((video, audio))
        noise = nt.NestedTensor((torch.randn(video.shape, generator=g(1)), torch.randn(audio.shape, generator=g(2))))
    else:
        lat, noise = video, torch.randn(video.shape, generator=g(1))
    return lat, noise, pos, neg


def _parts(t):
    return list(t.unbind()) if hasattr(t, "unbind") and not torch.is_tensor(t) else [t]


def _wrapper_state(m):
    w = m.wrappers[comfy.patcher_extension.WrappersMP.APPLY_MODEL][nodes.WRAPPER_KEY][0]
    return w.ltxe_call


def run(kind):
    patcher = _patcher(kind)
    lat, noise, pos, neg = _inputs(kind)
    sample = lambda m: comfy.sample.sample(m, noise, STEPS, 1.0, "euler", "simple", pos, neg, lat, seed=1)  # noqa: E731
    base = _parts(sample(patcher))
    one_m = nodes.LTXETiledModel().run(patcher, 384, 256, 0, "trapezoid", True)[0]
    one = _parts(sample(one_m))
    tiled_m = nodes.LTXETiledModel().run(patcher, 192, 128, 0, "trapezoid", True)[0]
    tiled = _parts(sample(tiled_m))
    state = _wrapper_state(tiled_m)
    return {
        "kind": kind, "parts": [list(p.shape) for p in base],
        "one_tile_max_diff": max(float((a - b).abs().max()) for a, b in zip(base, one)),
        "tiled_max_diff": [float((a - b).abs().max()) for a, b in zip(base, tiled)],
        "base_mean_abs": [float(a.abs().mean()) for a in base],
        "tiled_finite": all(bool(torch.isfinite(p).all()) for p in tiled),
        "tiles": state.last_plan.count if state.last_plan else 0, "tile_calls": state.calls,
        "one_tile_calls": _wrapper_state(one_m).calls,
    }


if __name__ == "__main__":
    ver = (CORE / "comfyui_version.py").read_text(encoding="utf-8").strip().splitlines()[-1]
    print("RESULT " + json.dumps({"core_version": ver, "steps": STEPS, "runs": [run("ltxv"), run("ltxav")]}))
