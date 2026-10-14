# Benchmarks

Each result here records:

- the hardware and the driver;
- the ComfyUI and front-end versions;
- the model version;
- the seed, prompt, sampler, steps, resolution and frame count.

A result that could not be reproduced is marked as such, never dropped.

| File | What it does |
|---|---|
| `metrics.py` | Output metrics: seam score on known tile boundaries, grid score without a plan, flicker, motion profile and frozen start, bad frames (NaN, infinite, blank), PSNR. Tested on synthetic clips with known defects (`tests/test_metrics.py`). |
| `run_benchmark.py` | Runs A/B arms of one exported workflow on a ComfyUI server. Records time, peak memory and the metrics of every output. Writes `results.json` and the frames. |

## GPU protocol

This needs LTX weights and a GPU, so it is run on the owner's machine. The aim is to move the LTXE nodes from *experimental* to *benchmarked* ([stability labels](../docs/stability.md)), or to show that they should not move.

### 1. Build the workflow

Start from a working LTX-2.5 workflow, one per test. The first is a **text-to-video** at DCP 2K, then:

1. Put **LTXE Resolution Plan** first, target *DCP 2K (1920x1080)*. Wire its width, height and length into `Empty LTXV Latent Video`.
2. Put **LTXE Tiled Model** between the model loader (and LoRA loaders) and the sampler. Use 1024 × 576 tiles, overlap 0 (= half a tile), trapezoid.
3. Give the scheduler a **tile-sized** latent (a second `Empty LTXV Latent Video` of 1024 × 576 with the same length), or manual sigmas. Put **LTXE Sigma Check** after it.
4. Decode, then **LTXE Crop Back** (plan from step 1, region *content*), then `Save Image`.

The runner rebuilds the same plan from the size, with `--plan-auto 1920 1080 97 1024 576`. It needs that plan to know where the tile boundaries are.

### 2. Export it

Use **Export (API)** to save `workflow_api.json`. Note two node ids: LTXE Tiled Model, and the node that holds the seed (`RandomNoise` or `KSampler`).

### 3. Write the arms

`arms_2k.json`, with the node ids from step 2:

```json
{
  "untiled":          [{"node": "12", "input": "enabled", "value": false}],
  "tiled_trapezoid":  [],
  "tiled_gaussian":   [{"node": "12", "input": "weight", "value": "gaussian"}],
  "tiled_overlap_256":[{"node": "12", "input": "min_overlap", "value": 256}]
}
```

The `untiled` arm at DCP 2K is the quality reference. On a 24 GB card it may run out of memory: that result counts too, so keep it.

### 4. Run

```
<ComfyUI python> benchmarks/run_benchmark.py --server http://127.0.0.1:8188 --prompt workflow_api.json --arms arms_2k.json \
    --repeats 3 --seed-node 25 --seed-input noise_seed --plan-auto 1920 1080 97 1024 576 --fps 24 \
    --out benchmarks/results/2026-10-xx_<gpu>_t2v_2k
```

`<ComfyUI python>` is the interpreter ComfyUI runs with; the metrics need its torch. The seeds are the same for every arm, 1..3 by default. The first run of each arm includes model loading and is reported separately.

### 5. Repeat for the other tests

| Test | What changes | Compare with |
|---|---|---|
| T2V DCP 2K | as above | untiled at 2K |
| Refine to DCP 4K | IC-LoRA Refine Details on an upscaled latent, tiles 1024 × 576, target DCP 4K | Lightricks' `LTXV Tiled Fusion Sampler` with the same LoRA, tile size and seed. That is a separate workflow; run it with `arms = {"fusion": []}`. |
| Long clip | `LTXV Context Windows` (core) before LTXE Tiled Model, 241 frames | the same without context windows, if it fits |

### 6. Send back

Send the whole results folder: `results.json` plus the frames. `results.json` already holds `/system_stats` (GPU, driver, ComfyUI and PyTorch versions). Also add the model file names and the prompt.

### What the numbers mean

- **seconds**: core's own execution time (execution start to success).
- **peak_vram_gib**: total minus the lowest free memory on the GPU, other programs included.
- **peak_increase_gib**: what the run added over the memory free just before it. This is the better number for A/B comparisons.
- Both memory figures are sampled every 0.2 s, so a short spike can be missed.
- On a `--cpu` server the "device" is system RAM, and `memory_device` says so.
- **seam score**: edge strength on the tile boundaries over the rest of the frame. About 1.0 means no visible seam. Compare arms on the same seed.
- **flicker**: frame-to-frame change of fine detail. Lower is steadier. Compare arms only on the same content.
