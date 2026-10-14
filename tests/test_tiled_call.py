"""TiledCall (the APPLY_MODEL wrapper body) with fake model executors: crops, blends, packing, refusals."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pack import load_pack  # noqa: E402

pack = load_pack()
tc = sys.modules[pack.__name__ + ".ltx_enhanced.tiled_call"] if (pack.__name__ + ".ltx_enhanced.tiled_call") in sys.modules \
    else __import__(pack.__name__ + ".ltx_enhanced.tiled_call", fromlist=["x"])

S = tc.TileSettings(tile_w=8, tile_h=6)           # cells; canvas below is 14 x 20 -> several tiles


def _x(b=2, c=4, f=3, h=14, w=20, seed=0):
    return torch.randn(b, c, f, h, w, generator=torch.Generator().manual_seed(seed))


def cellwise(x, t, c_concat, c_crossattn, control, transformer_options, **kw):
    """A 'model' whose x0 at a cell depends only on that cell (and the shared sigma / mask there)."""
    out = torch.tanh(x) * (1 - t.view(-1, *[1] * (x.ndim - 1)))
    if "denoise_mask" in kw:
        out = out * kw["denoise_mask"]
    return out


def test_cellwise_model_tiled_equals_untiled():
    x, t = _x(), torch.tensor([0.7, 0.7])
    call = tc.TiledCall(S)
    out = call(cellwise, x, t)
    assert call.last_plan.count > 1 and call.calls == call.last_plan.count
    assert torch.allclose(out, cellwise(x, t, None, None, None, {}), atol=1e-6)


def test_denoise_mask_is_cropped_per_tile():
    x, t = _x(), torch.tensor([0.4, 0.4])
    mask = torch.rand(2, 1, 3, 14, 20, generator=torch.Generator().manual_seed(3))
    out = tc.TiledCall(S)(cellwise, x, t, denoise_mask=mask)
    assert torch.allclose(out, cellwise(x, t, None, None, None, {}, denoise_mask=mask), atol=1e-6)


def test_one_tile_is_a_plain_call():
    seen = []

    def ex(x, *a, **kw):
        seen.append(x)
        return x * 2

    x = _x(h=6, w=8)
    out = tc.TiledCall(S)(ex, x, torch.tensor([0.5, 0.5]))
    assert len(seen) == 1 and seen[0] is x and torch.equal(out, x * 2)


def test_each_tile_sees_only_its_crop_and_its_index():
    calls = []

    def ex(x, t, cc, ca, ctl, opts, **kw):
        calls.append((tuple(x.shape[-2:]), opts["ltxe_tile"]))
        return x

    x = _x()
    call = tc.TiledCall(S)
    out = call(ex, x, torch.tensor([0.5, 0.5]))
    plan = call.last_plan
    assert [c[1] for c in calls] == [(i, plan.count) for i in range(plan.count)]
    assert all(c[0] == (6, 8) for c in calls)
    assert torch.allclose(out, x, atol=1e-6)          # identity model -> blend reproduces the canvas


def test_packed_audio_video_shares_audio_and_tiles_video():
    video, audio = _x(b=1), torch.randn(1, 8, 5, 16, generator=torch.Generator().manual_seed(5))
    x, shapes = tc.pack([video, audio])

    def ex(xp, t, cc, ca, ctl, opts, **kw):
        v, a = tc.unpack(xp, kw["latent_shapes"])
        assert tuple(a.shape) == tuple(audio.shape)    # every tile gets the whole audio stream
        return tc.pack([torch.tanh(v), a + 1.0])[0]

    out = tc.TiledCall(S)(ex, x, torch.tensor([0.5]), latent_shapes=shapes)
    v_out, a_out = tc.unpack(out, shapes)
    assert torch.allclose(v_out, torch.tanh(video), atol=1e-6)
    assert torch.allclose(a_out, audio + 1.0, atol=1e-6)


def test_pack_matches_core_layout():
    core_utils = pytest.importorskip("comfy.utils")
    a, b = _x(b=2), torch.randn(2, 8, 5, 16)
    ours, shapes = tc.pack([a, b])
    theirs, their_shapes = core_utils.pack_latents([a, b])
    assert torch.equal(ours, theirs) and list(shapes) == list(their_shapes)
    for u, v in zip(tc.unpack(ours, shapes), core_utils.unpack_latents(theirs, their_shapes)):
        assert torch.equal(u, v)


def _coords(b, f, h, w, spatial=32, t0=0):
    """keyframe_idxs for f guide frames on an h x w grid: [batch, (t, y, x), tokens, (start, end)] in pixels."""
    t, y, x = torch.meshgrid(torch.arange(f), torch.arange(h), torch.arange(w), indexing="ij")
    start = torch.stack([t + t0, y * spatial, x * spatial]).reshape(3, -1).float()
    return torch.stack([start, start + torch.tensor([1.0, spatial, spatial]).view(3, 1)], dim=-1).expand(b, 3, -1, 2).clone()


def test_guide_tokens_are_cropped_and_shifted_per_tile():
    f_content, f_guide, h, w = 3, 2, 14, 20
    x = _x(b=1, f=f_content + f_guide)                      # guide frames are appended to the video latent
    kf = _coords(1, f_guide, h, w, t0=40)
    entries = [{"latent_shape": [1, h, w], "pre_filter_count": h * w, "strength": 0.8, "pixel_mask": None},
               {"latent_shape": [1, h, w], "pre_filter_count": h * w, "strength": 1.0,
                "pixel_mask": torch.rand(1, 1, 1, h * 32, w * 32)}]
    gk = {"tokens_per_frame": h * w, "first_latent_frame": 0, "num_keyframes": 1}
    seen = []

    def ex(xt, t, cc, ca, ctl, opts, **kw):
        th, tw = xt.shape[-2:]
        k = kw["keyframe_idxs"]
        assert k.shape == (1, 3, f_guide * th * tw, 2)
        assert [e["pre_filter_count"] for e in kw["guide_attention_entries"]] == [th * tw, th * tw]
        assert kw["guide_attention_entries"][1]["pixel_mask"].shape[-2:] == (th * 32, tw * 32)
        assert kw["generated_keyframes"]["tokens_per_frame"] == th * tw
        # coordinates are tile-local: the first token of every guide frame sits at pixel (0, 0)
        grid = k[0, :, :, 0].reshape(3, f_guide, th, tw)
        assert grid[1].min() == 0 and grid[2].min() == 0
        assert grid[1].max() == (th - 1) * 32 and grid[2].max() == (tw - 1) * 32
        assert torch.equal(grid[0][:, 0, 0], torch.tensor([40.0, 41.0]))   # time is untouched
        seen.append((th, tw, opts["ltxe_tile"]))
        return torch.tanh(xt)

    call = tc.TiledCall(S)
    out = call(ex, x, torch.tensor([0.5]), keyframe_idxs=kf, guide_attention_entries=entries, generated_keyframes=gk)
    assert len(seen) == call.last_plan.count > 1
    assert torch.allclose(out, torch.tanh(x), atol=1e-6)
    # the caller's conditioning is never modified
    assert torch.equal(kf, _coords(1, f_guide, h, w, t0=40)) and entries[0]["pre_filter_count"] == h * w


def test_guide_without_entries_is_split_by_whole_frames():
    x = _x(b=1, f=4)
    seen = []

    def ex(xt, *a, **kw):
        seen.append(kw["keyframe_idxs"].shape[2] == xt.shape[-2] * xt.shape[-1])
        return xt

    tc.TiledCall(S)(ex, x, torch.tensor([0.5]), keyframe_idxs=_coords(1, 1, 14, 20))
    assert seen and all(seen)


@pytest.mark.parametrize("kw,match", [
    ({"keyframe_idxs": _coords(1, 1, 7, 10), "guide_attention_entries": [{"latent_shape": [1, 7, 10], "pre_filter_count": 70}]},
     "Downscaled guides"),
    ({"keyframe_idxs": torch.zeros(1, 3, 13, 2)}, "not whole frames"),
    ({"keyframe_idxs": _coords(1, 1, 14, 20), "guide_attention_entries": [{"latent_shape": [2, 14, 20], "pre_filter_count": 560}]},
     "do not add up"),
    ({"guide_attention_entries": [{"latent_shape": [1, 14, 20], "pre_filter_count": 280, "pixel_mask": torch.ones(1, 1, 1, 10, 10)}]},
     "attention mask"),
    ({"generated_keyframes": {"tokens_per_frame": 99, "first_latent_frame": 0, "num_keyframes": 1}}, "different latent size"),
])
def test_inconsistent_guides_are_refused_in_plain_english(kw, match):
    with pytest.raises(tc.TilingUnsupported, match=match):
        tc.TiledCall(S)(cellwise, _x(b=1), torch.tensor([0.5]), **kw)


def test_broadcast_denoise_mask_is_shared():
    x = _x(b=1)
    mask = torch.full((1, 1, 3, 1, 1), 0.5)
    out = tc.TiledCall(S)(cellwise, x, torch.tensor([0.5]), denoise_mask=mask)
    assert torch.allclose(out, cellwise(x, torch.tensor([0.5]), None, None, None, {}, denoise_mask=mask), atol=1e-6)


def test_empty_keyframe_tensor_is_not_a_guide():
    x = _x(b=1)
    out = tc.TiledCall(S)(cellwise, x, torch.tensor([0.5]), keyframe_idxs=torch.zeros(1, 3, 0, 2))
    assert torch.allclose(out, cellwise(x, torch.tensor([0.5]), None, None, None, {}), atol=1e-6)


def test_unknown_inputs_and_controlnet_are_refused():
    with pytest.raises(tc.TilingUnsupported, match="does not know how to tile: mystery"):
        tc.TiledCall(S)(cellwise, _x(b=1), torch.tensor([0.5]), mystery=torch.zeros(3))
    with pytest.raises(tc.TilingUnsupported, match="ControlNet"):
        tc.TiledCall(S)(cellwise, _x(b=1), torch.tensor([0.5]), None, None, object())
    with pytest.raises(tc.TilingUnsupported, match="does not match the latent grid"):
        tc.TiledCall(S)(cellwise, _x(b=1), torch.tensor([0.5]), denoise_mask=torch.ones(1, 1, 3, 7, 10))


def test_kill_switch_passes_through(monkeypatch):
    monkeypatch.setenv("LTXE_DISABLE_TILING", "1")
    seen = []

    def ex(x, *a, **kw):
        seen.append(x.shape)
        return x

    tc.TiledCall(S)(ex, _x(), torch.tensor([0.5, 0.5]))
    assert seen == [torch.Size([2, 4, 3, 14, 20])]


def test_non_video_latent_is_refused():
    with pytest.raises(tc.TilingUnsupported, match="5-D video latent"):
        tc.TiledCall(S)(cellwise, torch.zeros(1, 4, 14, 20), torch.tensor([0.5]))
