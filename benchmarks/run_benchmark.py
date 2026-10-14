"""Run A/B arms of one ComfyUI workflow and record time, peak VRAM and output metrics (standard library + torch).

    python benchmarks/run_benchmark.py --prompt workflow_api.json --arms arms.json --repeats 3 --out results/run1
        [--server http://127.0.0.1:8188] [--seed-node 5 --seed-input noise_seed --seeds 1,2,3]
        [--plan plan.json | --plan-auto 1920 1080 97 1024 576] [--plan-scale 2] [--grid-stride 1024] [--fps 24]

* ``--prompt``: the workflow exported with "Export (API)" (node id -> {class_type, inputs}).
* ``--arms``: {"arm name": [{"node": "12", "input": "enabled", "value": false}, ...], ...}. An empty list runs the
  prompt as exported.
* Each arm runs ``--repeats`` times; with ``--seed-node`` the seed input is set from ``--seeds`` (or 1..repeats), the
  same seeds for every arm, so arms are compared on equal noise.
* Time: from core's execution_start to execution_success timestamps; the first run of each arm is reported
  separately because it includes model loading.
* Memory: polled from /system_stats every 0.2 s for the server's first device. ``peak_used`` = total - lowest free
  (everything on that device, other programs included); ``peak_increase`` = free just before the run - lowest free
  during it (what the run added, the better A/B number). Both are sampled and can miss a short spike. On a server
  started with --cpu the "device" is system RAM, so the figures are RAM, and ``memory_device`` says so.
* Frames: every IMAGE an output node returns (SaveImage, PreviewImage) is downloaded and measured with
  benchmarks/metrics.py. ``--plan`` (an LTXE plan JSON) or ``--plan-auto W H FRAMES TILE_W TILE_H`` (the same plan
  LTXE Resolution Plan makes for that source size, delivered at the source size) adds the seam score on its tile
  boundaries.

Writes <out>/results.json (config, server system info, every run, per-arm summary) and the downloaded frames.
It only talks to the server you name; it never starts or stops one.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import statistics
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _get(server: str, path: str, timeout: float = 60):
    with urllib.request.urlopen(server + path, timeout=timeout) as r:
        return json.loads(r.read())


def _post(server: str, path: str, body: dict, timeout: float = 60):
    req = urllib.request.Request(server + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def apply_patches(prompt: dict, patches: list[dict]) -> dict:
    """A copy of ``prompt`` with each {node, input, value} applied; unknown nodes are an error, not a no-op."""
    out = copy.deepcopy(prompt)
    for p in patches:
        node = str(p["node"])
        if node not in out:
            raise KeyError(f"arm patch names node {node}, which is not in the workflow")
        out[node].setdefault("inputs", {})[p["input"]] = p["value"]
    return out


class VramPoller:
    """Polls /system_stats and keeps the lowest free VRAM of the first device (a sampled peak)."""

    def __init__(self, server: str, interval: float = 0.2):
        self.server, self.interval = server, interval
        self.min_free = None
        self.total = None
        self.baseline_free = None
        self.device = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                d = _get(self.server, "/system_stats", timeout=5)["devices"][0]
                self.total = d.get("vram_total")
                self.device = f"{d.get('type')}: {d.get('name')}"
                free = d.get("vram_free")
                if self.baseline_free is None:
                    self.baseline_free = free
                if free is not None:
                    self.min_free = free if self.min_free is None else min(self.min_free, free)
            except Exception:  # noqa: BLE001 - a missed sample is not a failed run
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t.start()
        for _ in range(50):              # one sample before the prompt is queued: the baseline
            if self.baseline_free is not None:
                break
            time.sleep(0.05)
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=5)

    @property
    def peak_used(self):
        return None if self.total is None or self.min_free is None else self.total - self.min_free

    @property
    def peak_increase(self):
        return None if self.baseline_free is None or self.min_free is None else max(0, self.baseline_free - self.min_free)


def run_once(server: str, prompt: dict, timeout: float = 7200) -> dict:
    client = str(uuid.uuid4())
    with VramPoller(server) as vram:
        pid = _post(server, "/prompt", {"prompt": prompt, "client_id": client})["prompt_id"]
        t0 = time.time()
        rec = {}
        while time.time() - t0 < timeout:
            h = _get(server, f"/history/{pid}")
            rec = h.get(pid) or {}
            if rec.get("status", {}).get("completed") is not None and (
                    rec["status"].get("completed") or rec["status"].get("status_str") == "error"):
                break
            time.sleep(0.5)
    msgs = {m[0]: m[1] for m in rec.get("status", {}).get("messages", [])}
    start = msgs.get("execution_start", {}).get("timestamp")
    end = (msgs.get("execution_success") or msgs.get("execution_error") or {}).get("timestamp")
    err = msgs.get("execution_error")
    return {"prompt_id": pid, "status": rec.get("status", {}).get("status_str"),
            "seconds": None if start is None or end is None else (end - start) / 1000.0,
            "peak_vram_bytes": vram.peak_used, "peak_increase_bytes": vram.peak_increase, "vram_total_bytes": vram.total,
            "memory_device": vram.device,
            "error": None if not err else {"node_type": err.get("node_type"), "message": str(err.get("exception_message"))[:500]},
            "outputs": rec.get("outputs", {})}


def download_frames(server: str, outputs: dict, dest: Path):
    """Every image an output node returned, as one [F, H, W, 3] tensor per node (ordered as returned)."""
    import numpy as np
    import torch
    from PIL import Image

    clips = {}
    for node_id, out in outputs.items():
        frames = []
        for img in out.get("images", []):
            q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                                        "type": img.get("type", "output")})
            with urllib.request.urlopen(f"{server}/view?{q}", timeout=120) as r:
                data = r.read()
            dest.mkdir(parents=True, exist_ok=True)
            (dest / f"{node_id}_{img['filename']}").write_bytes(data)
            frames.append(np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32) / 255.0)
        if frames:
            clips[node_id] = torch.from_numpy(np.stack(frames))
    return clips


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def summarise(runs: list[dict]) -> dict:
    ok = [r for r in runs if r["status"] == "success"]
    warm = ok[1:] if len(ok) > 1 else ok
    seams = [m["seam_score"]["max"] for r in ok for m in r.get("metrics", {}).values() if "seam_score" in m]
    flick = [m["flicker"] for r in ok for m in r.get("metrics", {}).values()]
    gib = lambda v: None if v is None else round(v / 2 ** 30, 2)  # noqa: E731
    return {"runs": len(runs), "succeeded": len(ok), "first_run_seconds": ok[0]["seconds"] if ok else None,
            "median_seconds_warm": _median([r["seconds"] for r in warm]),
            "memory_device": next((r.get("memory_device") for r in ok if r.get("memory_device")), None),
            "peak_vram_gib": gib(max((r["peak_vram_bytes"] for r in ok if r["peak_vram_bytes"] is not None), default=None)),
            "peak_increase_gib": gib(max((r.get("peak_increase_bytes") for r in ok
                                          if r.get("peak_increase_bytes") is not None), default=None)),
            "median_seam_score": _median(seams), "median_flicker": _median(flick),
            "errors": [r["error"] for r in runs if r["error"]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--prompt", required=True, type=Path)
    ap.add_argument("--arms", required=True, type=Path)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed-node")
    ap.add_argument("--seed-input", default="seed")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--plan", type=Path)
    ap.add_argument("--plan-auto", nargs=5, type=int, metavar=("W", "H", "FRAMES", "TILE_W", "TILE_H"))
    ap.add_argument("--plan-scale", type=float, default=1.0)
    ap.add_argument("--grid-stride", type=int, default=0)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)

    import metrics as M

    prompt = json.loads(a.prompt.read_text(encoding="utf-8"))
    arms = json.loads(a.arms.read_text(encoding="utf-8"))
    plan = json.loads(a.plan.read_text(encoding="utf-8")) if a.plan else None
    if plan is None and a.plan_auto:
        sys.path.insert(0, str(HERE.parent))
        from ltx_enhanced.plan import make_run_plan
        w, h, f, tw, th = a.plan_auto
        plan = make_run_plan(w, h, f, w, h, tile_w_px=tw, tile_h_px=th).to_dict()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()] or list(range(1, a.repeats + 1))
    system = _get(a.server, "/system_stats")
    res = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "server_system": system,
           "config": {"prompt": a.prompt.name, "arms": arms, "repeats": a.repeats, "seeds": seeds[:a.repeats],
                      "seed_node": a.seed_node, "seed_input": a.seed_input, "plan": a.plan.name if a.plan else (a.plan_auto or None),
                      "plan_scale": a.plan_scale, "grid_stride": a.grid_stride, "fps": a.fps},
           "arms": {}, "summary": {}}
    for name, patches in arms.items():
        runs = []
        for i in range(a.repeats):
            p = list(patches)
            if a.seed_node:
                p.append({"node": a.seed_node, "input": a.seed_input, "value": seeds[i % len(seeds)]})
            r = run_once(a.server, apply_patches(prompt, p))
            clips = download_frames(a.server, r.pop("outputs"), a.out / "frames" / name / f"run{i + 1}") \
                if r["status"] == "success" else {}
            r["metrics"] = {}
            for node_id, frames in clips.items():
                m = M.all_metrics(frames, plan=plan, scale=a.plan_scale, fps=a.fps)
                if a.grid_stride:
                    m["grid_score"] = M.grid_score(frames, a.grid_stride)
                r["metrics"][node_id] = m
            runs.append(r)
            print(f"{name} run {i + 1}: {r['status']} {r['seconds']} s, peak VRAM {r['peak_vram_bytes']}", flush=True)
        res["arms"][name] = runs
        res["summary"][name] = summarise(runs)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
