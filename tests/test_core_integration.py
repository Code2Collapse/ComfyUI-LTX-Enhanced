"""LTXE Tiled Model inside ComfyUI core's real sampler (tiny random LTXV and LTXAV models, CPU, clean subprocess).

Skipped when ComfyUI core is not found (see conftest.py). Not a quality test: see tests/core_probe.py.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent


def _core() -> Path | None:
    for c in [os.environ.get("LTXE_COMFY_CORE", ""), str(ROOT.parent.parent / "ComfyUI_windows_portable" / "ComfyUI")]:
        if c and (Path(c) / "comfy_api").is_dir():
            return Path(c)
    return None


@pytest.fixture(scope="module")
def probe():
    core = _core()
    if core is None:
        pytest.skip("ComfyUI core not found (set LTXE_COMFY_CORE)")
    out = subprocess.run([sys.executable, "-I", str(TESTS / "core_probe.py"), str(core)], capture_output=True,
                         text=True, timeout=900, encoding="utf-8", errors="replace")
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, f"probe failed (exit {out.returncode}):\n{out.stderr[-3000:]}"
    return json.loads(lines[-1][7:])


@pytest.mark.parametrize("i,kind", [(0, "ltxv"), (1, "ltxav")])
def test_tiled_model_runs_in_core_sampler(probe, i, kind):
    r = probe["runs"][i]
    assert r["kind"] == kind
    assert r["one_tile_max_diff"] == 0.0, "one tile must be exactly a plain model call"
    assert r["one_tile_calls"] > 0
    assert r["tiles"] > 1 and r["tile_calls"] % r["tiles"] == 0 and r["tile_calls"] >= r["tiles"] * probe["steps"]
    assert r["tiled_finite"]
    # different tiles see different context, so a tiled run differs from an untiled one - but only a little for
    # this near-linear random model; a wrong crop or blend would put whole-canvas-sized errors here
    for diff, mean in zip(r["tiled_max_diff"], r["base_mean_abs"]):
        assert 0.0 < diff < 0.05 * max(mean, 1e-6) or diff == 0.0
    if kind == "ltxav":
        assert len(r["parts"]) == 2, "the audio-video latent must come back as video + audio"


def test_ic_lora_guide_attached_by_core_is_tiled(probe):
    r = probe["guide_runs"][0]
    assert r["error"] is None, r["error"]
    assert r["guide_tokens"] == 8 * 12 and r["entries"] == [[1, 8, 12]]
    assert r["one_tile_max_diff"] == 0.0
    assert r["tiles"] > 1 and r["tile_calls"] >= r["tiles"] * probe["steps"] and r["tiled_finite"]
    assert r["tiled_max_diff"] < 0.05 * max(r["base_mean_abs"], 1e-6)


def test_downscaled_guide_is_refused_in_plain_english(probe):
    r = probe["guide_runs"][1]
    assert r["one_tile_max_diff"] == 0.0               # one tile never touches the guide bookkeeping
    assert r["error"] and "Downscaled guides" in r["error"]
