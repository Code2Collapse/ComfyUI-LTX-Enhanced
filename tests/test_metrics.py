"""Benchmark metrics on synthetic clips with known defects."""
import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT))
import metrics as M  # noqa: E402
from ltx_enhanced.plan import make_run_plan  # noqa: E402


def _smooth(f=6, h=96, w=160, seed=0):
    """Smooth, slowly moving, NON-periodic content: a sum of random low-frequency sinusoids. (An upsampled noise
    grid would itself repeat every few pixels and look like a tile grid to the metrics.)"""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    clip = torch.zeros(f, h, w)
    for _ in range(6):
        fx, fy = (torch.rand(2, generator=g) * 0.06 + 0.005).tolist()
        ph, sp = (torch.rand(2, generator=g) * 6.28).tolist()
        for t in range(f):
            clip[t] += torch.sin(xx * fx + yy * fy + ph + 0.2 * t * sp)
    clip = (clip - clip.min()) / (clip.max() - clip.min())
    return clip.unsqueeze(-1).expand(f, h, w, 3).contiguous()


def test_seam_score_finds_a_seam_and_ignores_clean_frames():
    clean = _smooth()
    seamed = clean.clone()
    seamed[:, :, 80:, :] = (seamed[:, :, 80:, :] + 0.08).clamp(0, 1)      # a brightness step at x = 80
    assert M.seam_score(clean, [80])["x"] < 1.5
    assert M.seam_score(seamed, [80])["x"] > 5.0
    assert M.seam_score(seamed, [40])["x"] < 1.5                          # nothing at a wrong position


def test_grid_score_detects_a_periodic_grid():
    clean = _smooth(w=256)
    grid = clean.clone()
    for x in range(32, 256, 32):
        grid[:, :, x:, :] = (grid[:, :, x:, :] + 0.03 * (-1) ** (x // 32)).clamp(0, 1)
    assert M.grid_score(grid, 32) > 2.5 and M.grid_score(clean, 32) < 1.3


def test_flicker_and_frozen_start():
    still = _smooth(f=1).expand(10, -1, -1, -1).contiguous()
    noisy = still + 0.05 * torch.randn(still.shape, generator=torch.Generator().manual_seed(1))
    assert M.flicker(still) == 0.0 and M.flicker(noisy) > 0.01
    moving = torch.stack([torch.roll(_smooth(f=1)[0], s * 3, dims=1) for s in range(40)])
    stalled = torch.cat([moving[:1].expand(24, -1, -1, -1), moving[1:]])
    assert M.frozen_start(stalled, fps=24) and not M.frozen_start(moving, fps=24)


def test_bad_frames_and_psnr():
    clip = _smooth()
    clip[2] = float("nan")
    clip[4] = 0.5
    r = M.bad_frames(clip)
    assert r["nonfinite"] == [2] and r["blank"] == [4] and not r["ok"]
    a = _smooth()
    assert M.psnr(a, a) == math.inf and 20 < M.psnr((a + 0.05).clamp(0, 1), a) < 30


def test_plan_boundaries_match_the_tile_edges_in_the_cropped_frame():
    plan = make_run_plan(1920, 1080, 9, 1920, 1080, tile_w_px=1024, tile_h_px=576).to_dict()
    xb, yb = M.boundaries_from_plan(plan)
    # columns: tiles at 0, 448, 896 px (cells 0, 14, 28 of 32 px), 1024 wide -> overlap edges
    assert xb == [448, 896, 1024, 1472]
    # rows: tiles at 0, 256, 512 px on the 1088 canvas, minus the 4 px top pad
    assert yb == [252, 508, 572, 828]
    xb2, _ = M.boundaries_from_plan(plan, scale=2.0)
    assert xb2 == [896, 1792, 2048, 2944]


def test_all_metrics_bundle():
    plan = make_run_plan(160, 96, 6, tile_w_px=96, tile_h_px=64).to_dict()
    out = M.all_metrics(_smooth(), plan=plan, reference=_smooth())
    assert out["bad_frames"]["ok"] and out["psnr"] == math.inf and "seam_score" in out
    with pytest.raises(ValueError):
        M.psnr(_smooth(), _smooth(w=80))


def test_runner_patches_and_summary():
    import run_benchmark as R
    prompt = {"5": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20}}}
    out = R.apply_patches(prompt, [{"node": "5", "input": "steps", "value": 8}])
    assert out["5"]["inputs"]["steps"] == 8 and prompt["5"]["inputs"]["steps"] == 20     # the input is not mutated
    with pytest.raises(KeyError, match="node 9"):
        R.apply_patches(prompt, [{"node": "9", "input": "x", "value": 1}])
    runs = [{"status": "success", "seconds": 30.0, "peak_vram_bytes": 2 ** 31, "error": None,
             "metrics": {"9": {"flicker": 0.01, "seam_score": {"max": 1.2}}}},
            {"status": "success", "seconds": 10.0, "peak_vram_bytes": 2 ** 30, "error": None,
             "metrics": {"9": {"flicker": 0.03, "seam_score": {"max": 1.4}}}},
            {"status": "error", "seconds": None, "peak_vram_bytes": None, "error": {"message": "OOM"}, "metrics": {}}]
    s = R.summarise(runs)
    assert s["succeeded"] == 2 and s["first_run_seconds"] == 30.0 and s["median_seconds_warm"] == 10.0
    assert s["peak_vram_gib"] == 2.0 and s["median_seam_score"] == pytest.approx(1.3)
    assert s["errors"] == [{"message": "OOM"}]


def test_flat_frames_give_finite_ratios():
    flat = torch.full((2, 64, 128, 3), 0.5)
    flat[:, :, 64:, :] = 0.7                       # one real edge, nothing else
    assert math.isfinite(M.grid_score(flat, 32)) and M.grid_score(flat, 32) < 1e3
    assert math.isfinite(M.seam_score(flat, [64])["x"])


def test_runner_plan_auto_matches_the_node_plan(tmp_path, monkeypatch):
    import run_benchmark as R
    seen = {}
    monkeypatch.setattr(R, "_get", lambda server, path, timeout=60: {"system": {}, "devices": []})
    monkeypatch.setattr(R, "run_once", lambda server, prompt, timeout=7200: seen.setdefault("ran", True) and
                        {"status": "error", "seconds": None, "peak_vram_bytes": None, "error": {"message": "x"},
                         "outputs": {}})
    (tmp_path / "p.json").write_text('{"1": {"class_type": "X", "inputs": {}}}', encoding="utf-8")
    (tmp_path / "a.json").write_text('{"only": []}', encoding="utf-8")
    assert R.main(["--prompt", str(tmp_path / "p.json"), "--arms", str(tmp_path / "a.json"), "--repeats", "1",
                   "--plan-auto", "1920", "1080", "97", "1024", "576", "--out", str(tmp_path / "out")]) == 0
    res = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    assert res["config"]["plan"] == [1920, 1080, 97, 1024, 576] and seen["ran"]
