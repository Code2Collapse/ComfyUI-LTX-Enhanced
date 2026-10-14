# LTXVTiledFusionSampler and LTXVGetTilingSizes

**Source:** [Lightricks/ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) at `7f22eb16eb` (2026-10-13). Files: `tiled_fusion_sampler.py`, `tiled_fusion_plan.py`, `tiled_fusion_sizes.py`, `tiled_fusion_sampler.md`.

**Licence:** LTX-2 Community License. This note describes behaviour only. It contains no upstream code.

**Method:** static reading. Measurements of core behaviour that the sampler relies on are cited where used.

**Markers:**

- **[O]** observed: read in the code at that revision, or measured;
- **[I]** inferred: follows from what was observed, but was not run;
- **[U]** unknown.

## What it is

Per-step latent fusion in the MultiDiffusion family (arXiv 2302.08113), with Gaussian blending weights (Mixture of Diffusers, arXiv 2302.02412):

- one latent canvas and one noise field for the whole frame;
- at every denoising step, the model runs on overlapping crops of the canvas;
- the stepped crops are blended back into the canvas.

Because every tile follows the same trajectory, seams do not form the way they do with tiles sampled independently. [O, docstring and module doc]

It is built for IC-LoRA passes (refine, upscale, restore), not for text-to-video. [O]

## Algorithm

**Inputs** [O]:

- a 5-D video latent `[1, C, F, H, W]` with a `noise_mask`;
- positive and negative conditioning from the IC-LoRA guide node;
- a SAMPLER and descending SIGMAS ending at 0;
- the tile size in pixels, overlap fraction `f`, blend variance `v`, grid cycle `k`, and temporal window `tile_frames`.

**Refusals** [O]:

- batch size above 1;
- a latent that is not 5-D (so the packed audio-video LTX-2 latent is out);
- a latent without a `noise_mask`.

So plain T2V or I2V at high resolution is not served by this node.

**Scale factors** [O]:

- temporal, height and width pixels per latent cell come from the VAE's `downscale_index_formula` when a VAE is connected;
- otherwise they come from the model's latent format;
- the last fallback is (8, 32, 32).

**Spatial grid (one axis)** [O]:

- Notation: canvas `C` latents, tile `T = min(tile_px / scale, C)`, minimum overlap `m = max(2, floor(T·f))`, clamped to `T − 1`.
- The tile count `n` is the smallest `n ≥ 2` such that `(n·T − C)/(n − 1) ≥ m`, or 1 when `C ≤ T`.
- The origins are `round(i·(C − T)/(n − 1))` for `i = 0..n−1`. They are evenly spread, and the first and last tiles touch the canvas edges.

**Shifted grids (grid_cycle k > 1)** [O]:

- Interior origins move by `+off`. The first and last stay pinned. The result is clamped to `[1, C − T − 1]` and deduplicated.
- `off = min(4, ov − m)` per axis (the smaller of the two axes), where `ov` is the realised overlap of the first pair. Below 1 the cycle is disabled with a warning.
- The k grids are offsets (0,0), (off,off), (off,0) and (0,off). Step `s` uses grid `s mod k`.
- [I] Coverage holds: the first gap grows by at most `ov − m`, so overlap stays ≥ m ≥ 2.
- [I] The shift is at most 4 latents (128 px). So cycling moves seams by at most 128 px, whatever the tile size.

**Blend weight** [O]:

- A separable Gaussian per tile axis of length `n`: `w(x) = exp(−(x − (n−1)/2)² / (2·v·n²))`, with no floor.
- With the default `v = 0.05`, the weight at a tile edge is `exp(−1/(8v)) = exp(−2.5) ≈ 0.082` [I, arithmetic].
- The weight sums are precomputed per grid and clamped to at least 1e-12.

**Temporal windows** [O]:

- `tile_frames` is snapped to the nearest `8n+1`; ties keep the shorter length.
- Windows are planned on latent frames with the same grid rule and a minimum overlap of 50%. An environment variable overrides that fraction.
- The temporal weight is a Gaussian of the same form.
- For a window that starts inside the previous one, its first latent is not merged. It is a fresh-clip causal start frame (encodes 1 pixel frame), not a mid-clip frame (encodes 8 frames). This is on by default, with an environment kill switch.
- Windowing needs a per-window guide encode written onto the conditioning by the guide node (the "temporal plan"). Without it, windowing is disabled with a warning. A whole-clip guide is never sliced.

**Noise and the canvas** [O]:

- One CPU generator with the given seed draws noise of the full latent shape.
- The canvas starts as `noise_scaling(σ0, noise, latent_in)`, in the model's processed latent space.
- Canvas tensors live on the GPU when `16 × numel` bytes is under 25% of free VRAM, otherwise on the CPU. This is the `auto` setting; `gpu` and `cpu` force it.

**Per tile, once before the loop** [O]:

- crop the latent, the noise and the mask;
- crop the guide metadata: `keyframe_idxs` reshaped per guide entry to (3, f, gh, gw, 2), cropped to the tile, with the pixel origin subtracted so every tile looks like a clip at origin (0, 0);
- run `process_conds` for that tile.

All processed conds are cached for the whole run, on the GPU. [O] A guide latent whose grid differs from the canvas (factor > 1) is refused. [O]

**Each step** `σ_s → σ_{s+1}`, for every window and every tile of the step's grid [O]:

1. Slice the canvas, plus that window's own guide block.
2. Call `sampler.sampler_function` once, with the two-sigma slice and an inpaint wrapper around a CFG pass.
3. Accumulate `x_next · w_tile · w_time` over the content frames.

The canvas becomes `acc / wsum`, with the guide frames carried along unchanged. At the end the appended guide frames are dropped (frames whose mask max is ≥ 0.999, counted from the end).

**Cost** [I, arithmetic from the grid rule]:

- Model calls per step = tiles × windows.
- DCP 2K canvas (60 × 34 latents), 1024 × 576 tile (32 × 18), f = 0.5: a 3 × 3 = 9-tile grid. Its area ratio is 3.5, so the overlap overhead is about 2.5×.
- DCP 4K canvas (128 × 68): 7 × 7 = 49 tiles per step against an area ratio of 15.1, about 3.2× overhead.

## Size planner (LTXVGetTilingSizes)

[O]:

- A ladder by long edge: qHD 960, HD 1280, FullHD 1920. "4K" is 2 × FullHD, and "8K" is 2 × 4K.
- A source within 200 px of a step is kept as it is.
- Every size is snapped to the nearest multiple of 32, ties up. **The snap resizes, it does not pad.**
- `tile_frames` is 97, stretched up to 121 when that avoids a second window.
- The planner reports tile counts and window counts with the same grid rule as the sampler.

[I] Consequences:

- A 1920 × 1080 source becomes 1920 × 1088: stretched by 0.74% vertically, not padded and cropped back.
- "4K" from a 16:9 source is 3840 × 2176, not DCI 4096 × 2160 (DCP 4K).
- There is no aspect-preserving padding, no crop-back step, and no record of offsets.

## Failure modes

1. **[I] Ancestral samplers inject the same noise in every tile and at every step.**
   - Core's default ancestral noise sampler builds a new generator seeded with `extra_args["seed"]` on each call (`comfy/k_diffusion/sampling.py` `default_noise_sampler`, observed at core `e0a32bce84`).
   - The fusion loop calls the sampler once per tile per step with the same seed and the same tile shape.
   - So the "fresh" noise is one fixed pattern, repeated across tiles and across steps.
   - Upstream's documentation only says ancestral samplers "can seam".
2. **[O] History samplers** (DPM++ 2M, LMS) lose their multi-step memory. Upstream documents this.
3. **[O] T2V / I2V at 4K is not served.** A noise mask is required.
4. **[O] The audio-video latent (LTX-2 joint generation) is not served.** The node needs a 5-D latent.
5. **[I] Conds cached per (window, tile) on the GPU.** At DCP 4K with several windows that is 49 × windows sets of processed conditioning. [U] Their size has not been measured.
6. **[I] The schedule must not be computed from the full canvas.** Core's token-count shift saturates above about 2K and returns NaN at DCP 4K (see [core geometry and schedule](core-ltx-geometry-and-schedule.md)). Upstream's examples use Manual Sigmas, which avoids it. A user who wires LTXVScheduler to the full-canvas latent does not.
7. **[I] The blend has no floor.** A tile's edge still carries 8% weight, so attention edge effects near tile borders leak at about 8% into the blend.

## What we keep, and what we do differently

| Keep (backed by papers or upstream measurements) | Improve (our hypothesis, to be benchmarked) |
|---|---|
| Per-step fusion with one canvas and one noise field | Coherent stochastic noise: slice one canvas-level noise draw per step, so ancestral and SDE samplers stay seamless |
| Overlap ≥ 0.5 by default (upstream's statement about grid artefacts; to reproduce) | Plan the overlap from measured seam metrics, not one fixed fraction; compare against SpotDiffusion-style shifted grids with low overlap and more cycles (arXiv 2407.15507) |
| Fresh causal encode per temporal window; skip the window's first latent | Same, plus carry the audio latent for LTX-2 joint generation (a research item) |
| Tile coordinates normalised to each tile | Schedule from the tile token count, plus a guard that refuses NaN or saturated sigmas |
| | Aspect-preserving padding with exact crop-back offsets (DCP 1920 × 1088 → 1920 × 1080, DCP 4K 4096 × 2176 → 4096 × 2160), instead of a resize snap |
| | T2V / I2V / keyframe latents without IC-LoRA |
