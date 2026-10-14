"""Planning core (ltx_enhanced): geometry, padding + crop-back, tile plans, blending, sigma checks, run plan."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ltx_enhanced import geometry, padding, plan as runplan, schedule, tiles  # noqa: E402
from ltx_enhanced.blend import TileAccumulator  # noqa: E402

G = geometry.LTXV


# ---------------------------------------------------------------- geometry
@pytest.mark.parametrize("px,aligned", [(1, 32), (32, 32), (33, 64), (1080, 1088), (2160, 2176), (1920, 1920)])
def test_aligned_size(px, aligned):
    assert G.aligned_size(px) == aligned


@pytest.mark.parametrize("n,aligned", [(1, 1), (2, 9), (9, 9), (97, 97), (98, 105), (100, 105), (121, 121)])
def test_aligned_frames_is_8k_plus_1(n, aligned):
    assert G.aligned_frames(n) == aligned
    assert G.pixel_frames(G.latent_frames(aligned)) == aligned


def test_geometry_refuses_unaligned():
    with pytest.raises(ValueError):
        G.latent_size(1080)
    with pytest.raises(ValueError):
        G.latent_frames(100)


# ---------------------------------------------------------------- padding
def _clip(f, h, w, c=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(f, h, w, c, generator=g)


@pytest.mark.parametrize("w,h,top,bottom,sh", [(1920, 1080, 4, 4, 1088), (4096, 2160, 8, 8, 2176)])
def test_dcp_offsets(w, h, top, bottom, sh):
    p = padding.make_plan(w, h, 97, w, h)
    assert (p.sample_w, p.sample_h, p.sample_frames) == (w, sh, 97)
    assert (p.pad_left, p.pad_top, p.pad_right, p.pad_bottom) == (0, top, 0, bottom)


@pytest.mark.parametrize("mode", padding.PAD_MODES)
@pytest.mark.parametrize("placement", padding.PLACEMENTS)
def test_pad_then_crop_is_identity(mode, placement):
    src = _clip(10, 45, 70)
    p = padding.make_plan(70, 45, 10, mode=mode, placement=placement, value=0.25)
    canvas = padding.pad_images(src, p)
    assert canvas.shape == (17, 64, 96, 3)                      # 10 -> 8k+1 = 17 frames, 45 -> 64, 70 -> 96
    assert torch.equal(padding.crop_back(canvas, p), src)


def test_frames_repeat_last_and_are_dropped():
    src = _clip(100, 32, 32)
    p = padding.make_plan(32, 32, 100)
    canvas = padding.pad_images(src, p)
    assert canvas.shape[0] == 105 and p.frames_after == 5
    assert torch.equal(canvas[100:], src[-1:].expand(5, -1, -1, -1))
    assert torch.equal(padding.crop_back(canvas, p), src)


def test_reflect_works_for_pads_larger_than_the_image():
    src = _clip(1, 5, 7)
    p = padding.make_plan(7, 5, 1, 70, 50)                      # pad far beyond the source size
    canvas = padding.pad_images(src, p)
    assert canvas.shape == (1, 64, 96, 3)
    assert torch.equal(padding.crop_back(canvas, p), src)
    # every value in the padded canvas comes from the source (reflection, not zeros)
    assert set(canvas.flatten().tolist()) <= set(src.flatten().tolist())


def test_reflect_mirrors_without_repeating_the_edge():
    src = torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1).expand(1, 1, 4, 1).contiguous()
    p = padding.make_plan(4, 1, 1, 7, 1, placement="center")      # 7 wide -> 32 aligned; left pad 1 + 14
    canvas = padding.pad_images(src, p)
    row = canvas[0, p.pad_top, :, 0].tolist()
    i = p.pad_left
    assert row[i:i + 4] == [0, 1, 2, 3]
    assert row[i - 1] == 1 and row[i + 4] == 2                  # mirror around the edge sample


def test_constant_padding_value():
    src = torch.ones(1, 30, 30, 3)
    p = padding.make_plan(30, 30, 1, mode="constant", value=0.5)
    canvas = padding.pad_images(src, p)
    assert canvas[0, 0, 0, 0].item() == 0.5 and canvas[0, p.pad_top, p.pad_left, 0].item() == 1.0


def test_pad_refuses_a_larger_source_with_a_plain_message():
    with pytest.raises(ValueError, match="larger than the target frame"):
        padding.make_plan(2048, 1080, 9, 1920, 1080)


def test_fit_inside_keeps_aspect_and_fits():
    p = padding.make_plan(1998, 1080, 9, 1920, 1080, fit="fit-inside")
    assert p.scaled_w == 1920 and p.scaled_h == round(1080 * 1920 / 1998)
    assert p.content.w <= 1920 and p.content.h <= 1080
    out = padding.crop_back(padding.pad_images(_clip(9, 1080, 1998), p), p)
    assert out.shape == (9, p.scaled_h, 1920, 3)
    assert padding.crop_back(padding.pad_images(_clip(9, 1080, 1998), p), p, region="delivery").shape == (9, 1080, 1920, 3)


def test_crop_to_fill_records_the_crop():
    p = padding.make_plan(2048, 858, 9, 1920, 1080, fit="crop-to-fill")
    assert (p.crop.w, p.crop.h) == (1920, 1080) and p.scaled_h == 1080
    out = padding.crop_back(padding.pad_images(_clip(9, 858, 2048), p), p)
    assert out.shape == (9, 1080, 1920, 3)


def test_masks_follow_the_same_transform():
    src = _clip(9, 45, 70)
    mask = (src[..., 0] > 0.5).float()
    p = padding.make_plan(70, 45, 9)
    pm = padding.pad_masks(mask, p)
    assert pm.shape == (9, 64, 96)
    c = p.content
    assert torch.equal(pm[:9, c.y:c.y + c.h, c.x:c.x + c.w], mask)
    assert padding.pad_masks(mask[0], p).shape == (9, 64, 96)     # a single [H, W] mask is broadcast


def test_crop_back_after_2x_upscale():
    p = padding.make_plan(1920, 1080, 9, 1920, 1080)
    up = torch.zeros(9, 2176, 3840, 3)
    up[:, 8:2168, :, :] = 1.0                                     # content area at 2x
    out = padding.crop_back(up, p)
    assert out.shape == (9, 2160, 3840, 3) and bool((out == 1.0).all())


def test_crop_back_refuses_wrong_sizes():
    p = padding.make_plan(70, 45, 9)
    with pytest.raises(ValueError, match="uniform scale"):
        padding.crop_back(torch.zeros(9, 64, 100, 3), p)
    with pytest.raises(ValueError, match="dropped frames"):
        padding.crop_back(torch.zeros(5, 64, 96, 3), p)
    with pytest.raises(ValueError, match="plan was made for"):
        padding.pad_images(_clip(9, 46, 70), p)


def test_pad_plan_round_trips_through_json():
    p = padding.make_plan(1998, 1080, 50, 1920, 1080, fit="fit-inside")
    assert padding.PadPlan.from_dict(json.loads(json.dumps(p.to_dict()))) == p


# ---------------------------------------------------------------- tiles
@pytest.mark.parametrize("length,tile,ov", [(60, 32, 16), (34, 18, 9), (128, 32, 16), (68, 18, 9), (33, 32, 16),
                                             (50, 32, 16), (100, 40, 1), (17, 8, 4), (5, 5, 2)])
def test_split_axis_covers_and_respects_overlap(length, tile, ov):
    ts = tiles.split_axis(length, tile, ov)
    assert ts[0].start == 0 and ts[-1].end == length
    for a, b in zip(ts, ts[1:]):
        overlap = a.end - b.start
        assert overlap >= ov and a.right_ramp == overlap == b.left_ramp
    # fewest tiles: one fewer would break the overlap
    n = len(ts)
    if n > 2:
        assert ((n - 1) * tile - length) / (n - 2) < ov


def test_split_axis_refuses_overlap_above_half():
    with pytest.raises(ValueError, match="half the tile"):
        tiles.split_axis(100, 32, 17)


def test_trapezoid_one_cell_overlap_blends_half_and_half():
    w = tiles.trapezoid_1d(8, 1, 1)
    assert w[0].item() == 0.5 and w[-1].item() == 0.5 and w[1:-1].eq(1).all()


@pytest.mark.parametrize("h,w,th,tw,ov", [(34, 60, 18, 32, 9), (68, 128, 18, 32, 9), (40, 40, 16, 16, 1), (34, 60, 34, 60, 0)])
def test_trapezoid_weights_sum_to_one_without_triple_coverage(h, w, th, tw, ov):
    p = tiles.plan_tiles(h, w, th, tw, ov)
    s = p.weight_sum("trapezoid")
    if max(p.axis_coverage) <= 2:      # neighbour ramps are complementary, so they sum to exactly 1
        assert torch.allclose(s, torch.ones_like(s), atol=1e-12)
    assert bool((s > 0).all())


def test_tile_counts_for_dcp_canvases():
    # 1024x576 tiles (32x18 cells), half overlap: DCP 2K canvas 60x34 cells -> 3x3, DCP 4K 128x68 -> 7x7
    assert tiles.plan_tiles(34, 60, 18, 32).count == 9
    assert tiles.plan_tiles(68, 128, 18, 32).count == 49


def test_triple_coverage_is_reported_and_still_normalised():
    p = tiles.plan_tiles(50, 8, 32, 8, 16)              # starts 0, 9, 18 along rows
    assert [t.start for t in p.rows] == [0, 9, 18] and p.axis_coverage == (3, 1) and p.max_coverage == 3
    assert bool((p.weight_sum() > 0).all())


def test_gaussian_floor():
    g = tiles.gaussian_1d(32, 0.05, floor=0.2)
    assert g.min().item() >= 0.2 and abs(g.max().item() - 1.0) < 0.01


# ---------------------------------------------------------------- blend
@pytest.mark.parametrize("kind", tiles.WEIGHTS)
def test_blend_of_a_cellwise_function_equals_the_untiled_result(kind):
    canvas = torch.randn(1, 4, 3, 34, 60, generator=torch.Generator().manual_seed(1))
    f = lambda x: torch.tanh(x) * 2 - 0.3                     # noqa: E731 - acts per cell
    p = tiles.plan_tiles(34, 60, 18, 32, 9)
    acc = TileAccumulator(canvas.shape, p, kind=kind, floor=0.05 if kind == "gaussian" else 0.0)
    for t in p.tiles:
        ys, xs = t.slices()
        acc.add(t, f(canvas[..., ys, xs]))
    assert torch.allclose(acc.result(), f(canvas), atol=1e-5)


def test_blend_refuses_a_gap():
    p = tiles.plan_tiles(34, 60, 18, 32, 9)
    acc = TileAccumulator((1, 1, 1, 34, 60), p)
    acc.add(p.tiles[0], torch.zeros(1, 1, 1, 18, 32))
    with pytest.raises(RuntimeError, match="not covered"):
        acc.result()


# ---------------------------------------------------------------- schedule checks
def test_check_sigmas_flags_the_measured_failures():
    repro = ROOT / "crash_repros" / "001-ltxv-schedule-at-4k" / "result_core_0.39.0.json"
    cases = {c["name"]: c for c in json.loads(repro.read_text(encoding="utf-8"))["cases"]}
    nan = [float("nan") if v is None else v for v in cases["4096x2176 DCP 4K"]["ltxv_scheduler_sigmas"]]
    assert not schedule.check_sigmas(nan)["ok"] and schedule.check_sigmas(nan)["nan_count"] == 20
    assert not schedule.check_sigmas(cases["3840x2176 UHD"]["ltxv_scheduler_sigmas"])["ok"]          # flat 0.1
    assert not schedule.check_sigmas(cases["2560x1440 QHD"]["ltxv_scheduler_sigmas"])["ok"]          # staircase at 1.0
    assert not schedule.check_sigmas(cases["1920x1088 DCP 2K"]["model_sampling_simple_sigmas"])["ok"]  # saturated
    assert schedule.check_sigmas(cases["tile 1024x576"]["ltxv_scheduler_sigmas"])["ok"]
    assert schedule.check_sigmas(cases["1920x1088 DCP 2K"]["ltxv_scheduler_sigmas"])["ok"]
    assert "tile size" in schedule.check_sigmas(nan)["message"]


def test_check_sigmas_flags_rising():
    assert schedule.check_sigmas([1.0, 0.5, 0.7, 0.0])["rising"] == 1


# ---------------------------------------------------------------- run plan
def test_run_plan_dcp4k_summary_and_round_trip():
    p = runplan.make_run_plan(4096, 2160, 97, 4096, 2160, tile_w_px=1024, tile_h_px=576)
    assert (p.pad.sample_w, p.pad.sample_h, p.latent_frames) == (4096, 2176, 13)
    assert p.tiles.count == 49 and p.tile_tokens == 13 * 18 * 32
    assert p.canvas_tokens == 13 * 68 * 128
    text = p.summary()
    assert "49 tiles" in text and "warning" in text
    again = runplan.Plan.from_dict(json.loads(p.to_json()))
    assert again.to_dict() == p.to_dict()


def test_run_plan_untiled_small_has_no_warning():
    p = runplan.make_run_plan(768, 512, 97)
    assert p.tiles.count == 1 and p.warnings() == []
    assert math.isclose(p.canvas_tokens, 13 * 16 * 24)


def test_planning_core_imports_no_comfyui():
    for mod in ("geometry", "padding", "tiles", "blend", "schedule", "plan"):
        src = (ROOT / "ltx_enhanced" / f"{mod}.py").read_text(encoding="utf-8")
        assert "import comfy" not in src and "from comfy" not in src, mod


def test_package_has_no_side_effects_on_import():
    spec = importlib.util.spec_from_file_location("ltxe_core_probe", ROOT / "ltx_enhanced" / "__init__.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
