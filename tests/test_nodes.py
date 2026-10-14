"""Node layer: input schemas, outputs, round trips through the node classes, plain-English refusals."""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pack import load_pack  # noqa: E402

pack = load_pack()
nodes = sys.modules[pack.__name__ + ".experimental.nodes"]
M = pack.NODE_CLASS_MAPPINGS
SOCKETS = {"IMAGE", "MASK", "MODEL", "LATENT", "SIGMAS", "CONDITIONING", "VAE", nodes.PLAN_TYPE}


def test_registered_nodes_and_menu_root():
    assert set(M) == {"LTXE_ResolutionPlan", "LTXE_PadFrames", "LTXE_CropBack", "LTXE_TiledModel", "LTXE_SigmaCheck"}
    for nid, cls in M.items():
        assert cls.CATEGORY.startswith("\U0001F43A C2C/\U0001F3A5 LTX Enhanced/"), (nid, cls.CATEGORY)
        assert "Experimental" in cls.DESCRIPTION
        schema = cls.INPUT_TYPES()
        for section in schema.values():
            for name, spec in section.items():
                opts = spec[1] if len(spec) > 1 else {}
                assert opts.get("tooltip") or spec[0] in ("SIGMAS",), f"{nid}.{name} has no tooltip"
        assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES)
    assert set(pack.NODE_DISPLAY_NAME_MAPPINGS) == set(M)


def _defaults(cls, **over):
    req = cls.INPUT_TYPES()["required"]
    vals = {k: (v[1].get("default") if len(v) > 1 and isinstance(v[1], dict) else None) for k, v in req.items()
            if not (isinstance(v[0], str) and v[0] in SOCKETS)}       # sockets are passed by the test itself
    vals.update(over)
    return vals


def test_resolution_plan_dcp4k():
    out = M["LTXE_ResolutionPlan"]().run(**_defaults(M["LTXE_ResolutionPlan"], width=4096, height=2160, frames=100,
                                                      target="DCP 4K (4096x2160)"))
    plan, w, h, length, text = out["result"]
    assert (w, h, length) == (4096, 2176, 105)
    assert plan["pad"]["pad_top"] == 8 and plan["pad"]["frames_after"] == 5
    assert "49 tiles" in text and out["ui"]["text"] == [text]
    json.dumps(plan)        # the plan must travel as JSON


def test_pad_frames_then_crop_back_restores_the_source():
    src = torch.rand(10, 45, 70, 3)
    mask = (src[..., 0] > 0.5).float()
    res = M["LTXE_PadFrames"]().run(src, **_defaults(M["LTXE_PadFrames"], target="source size"), mask=mask)["result"]
    padded, pmask, plan, w, h, length, _ = res
    assert padded.shape == (17, 64, 96, 3) and pmask.shape == (17, 64, 96) and (w, h, length) == (96, 64, 17)
    back = M["LTXE_CropBack"]().run(padded, plan, "content")[0]
    assert torch.equal(back, src)


def test_pad_frames_without_mask_returns_an_empty_mask():
    res = M["LTXE_PadFrames"]().run(torch.rand(1, 32, 32, 3), **_defaults(M["LTXE_PadFrames"], target="source size"))
    assert res["result"][1].shape == (1, 32, 32) and float(res["result"][1].abs().sum()) == 0.0


def test_pad_frames_refuses_too_large_in_plain_english():
    with pytest.raises(ValueError, match="larger than the target frame"):
        M["LTXE_PadFrames"]().run(torch.rand(1, 1100, 1920, 3), **_defaults(M["LTXE_PadFrames"]))


def test_sigma_check_stops_on_nan_and_passes_good_schedules():
    good = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0])
    out = M["LTXE_SigmaCheck"]().run(good, True)
    assert torch.equal(out["result"][0], good) and "looks usable" in out["result"][1]
    with pytest.raises(ValueError, match="tile size"):
        M["LTXE_SigmaCheck"]().run(torch.tensor([float("nan")] * 4 + [0.0]), True)
    report = M["LTXE_SigmaCheck"]().run(torch.tensor([float("nan")] * 4 + [0.0]), False)["result"][1]
    assert "NaN 4" in report


def test_tiled_model_disabled_returns_the_same_model(monkeypatch):
    sentinel = object()
    assert M["LTXE_TiledModel"]().run(sentinel, 1024, 576, 0, "trapezoid", False)[0] is sentinel
    monkeypatch.setenv("LTXE_DISABLE_TILING", "1")
    assert M["LTXE_TiledModel"]().run(sentinel, 1024, 576, 0, "trapezoid", True)[0] is sentinel


def test_tiled_model_refuses_a_non_ltx_model():
    pytest.importorskip("comfy.model_base")

    class NotLTX:
        model = object()

    with pytest.raises(ValueError, match="needs an LTX model"):
        M["LTXE_TiledModel"]().run(NotLTX(), 1024, 576, 0, "trapezoid", True)


def test_kill_switch_registers_no_nodes(monkeypatch):
    monkeypatch.setenv("LTXE_DISABLE", "1")
    fresh = load_pack("ltxe_pack_disabled_probe")
    assert fresh.NODE_CLASS_MAPPINGS == {}


class _FakeVAE:
    def __init__(self, formula):
        self.downscale_index_formula = formula


def test_vae_scale_factors_drive_the_plan():
    cls = M["LTXE_ResolutionPlan"]
    plan, w, h, length, _ = cls().run(**_defaults(cls, width=1000, height=600, frames=10, target="source size"),
                                     vae=_FakeVAE((4, 16, 16)))["result"]
    assert (w, h, length) == (1008, 608, 13)            # 16-px cells, 4k+1 frames
    assert plan["geometry"]["spatial"] == 16 and plan["geometry"]["temporal"] == 4
    plan32 = cls().run(**_defaults(cls, width=1000, height=600, frames=10, target="source size"))["result"]
    assert plan32[1:4] == (1024, 608, 17)               # the LTX geometry when no VAE is connected


def test_crop_back_uses_the_plan_geometry():
    src = torch.rand(5, 30, 50, 3)
    res = M["LTXE_PadFrames"]().run(src, **_defaults(M["LTXE_PadFrames"], target="source size"),
                                    vae=_FakeVAE((4, 16, 16)))["result"]
    padded, plan = res[0], res[2]
    assert padded.shape == (5, 32, 64, 3)
    assert torch.equal(M["LTXE_CropBack"]().run(padded, plan, "content")[0], src)


def test_vae_without_video_factors_is_refused():
    cls = M["LTXE_ResolutionPlan"]
    with pytest.raises(ValueError, match="does not report its video scale factors"):
        cls().run(**_defaults(cls, width=64, height=64, frames=1, target="source size"), vae=_FakeVAE(None))
    with pytest.raises(ValueError, match="square cells"):
        cls().run(**_defaults(cls, width=64, height=64, frames=1, target="source size"), vae=_FakeVAE((8, 32, 16)))
