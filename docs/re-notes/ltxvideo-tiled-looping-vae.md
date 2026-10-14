# LTXVTiledSampler, LTXVLoopingSampler, LTXVTiledVAEDecode

**Source:** [Lightricks/ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) at `7f22eb16eb` (2026-10-13). Files: `tiled_sampler.py`, `looping_sampler.py`, `looping_sampler.md`, `easy_samplers.py` (extend sampler, linear overlap transition), `tiled_vae_decode.py`.

**Licence:** LTX-2 Community License. Behaviour notes only.

**Markers:** [O] observed, [I] inferred, [U] unknown.

## LTXVTiledSampler: independent spatial tiles

**What it does** [O]:

- Splits the latent into `h × v` tiles (1–6 each), with an `overlap` in latent cells (1–8, default 1 = 32 px).
- Every tile but the last has length `(C + (n−1)·overlap) // n`. The last tile takes the remainder.
- Runs a complete `SamplerCustomAdvanced` trajectory per tile, with the same NOISE object and the same guider.
- Optional image conditioning is cropped per tile and added as a guide.
- Blends the tiles afterwards with linear ramps over the overlap.
- Both of its outputs are the same blended result. There is no separate denoised (x0) estimate.

**Blend arithmetic** [I, from `linspace` semantics]:

- The left/top ramp of a tile runs `0 → 1` across `overlap` cells, endpoints included. The right/bottom ramp of its neighbour runs `1 → 0` over the same cells. The two always sum to 1.
- With `overlap = 1` the ramps are `[0]` and `[1]`. With `overlap = 2` they are `[0, 1]` and `[1, 0]`.
- So at the default overlap, and at 2, every latent cell comes from exactly one tile: **a hard seam, no blending**.
- Blending only starts at `overlap ≥ 3`, and its effective width is `overlap − 2` cells.

**Noise** [I]:

- Core's random-noise source draws from the seed with the latent's shape (`comfy.sample.prepare_noise`).
- The same seed and the same tile shape give the same noise, so **every same-sized tile starts from identical noise**.
- That is a direct path to repeated structure across tiles.

**Conditioning** [O]:

- Image conditioning is cropped per tile.
- The text prompt is the same for every tile, so each tile is told to render the whole scene. That is the classic cause of repeated objects in independent tiling. [I]

## LTXVLoopingSampler: temporal chain inside spatial tiles

**Structure** [O]:

- The outer loop is over spatial tiles, with the same tile rule and linear blend as above (the same hard seam at `spatial_overlap` 1–2).
- The inner loop is over temporal chunks of `temporal_tile_size` with `temporal_overlap` (pixel frames, steps of 8).
- The first chunk is a base or in-context sample.
- Each later chunk extends the previous one. It conditions on the previous chunk's last `overlap` latents at `temporal_overlap_cond_strength` (default 0.5). It can renormalise with AdaIN against the first chunk, or against per-frame reference latents.

**Seeds** [O]:

- Per chunk and tile: `first_seed + start_index·(V·H) + v·H + h + offset`.
- They are distinct per (chunk, tile), unlike LTXVTiledSampler.

**Temporal join** [O]:

- The extend step blends the overlap with a linear transition whose endpoints are excluded, so the join is a real crossfade (unlike the spatial ramps).

**Coupling** [I]:

- Spatial tiles are never coupled during sampling. Each tile runs its own temporal chain, so seams can drift over time as well as in space.
- Per-step fusion (the fusion sampler) avoids exactly that.

## LTXVTiledVAEDecode / LTXVSpatioTemporalTiledVAEDecode

**Spatial** [O]:

- Tiles are decoded independently, using the same tile rule.
- They are blended in pixel space with linear ramps over `overlap × 32` pixels (32 px at the default), endpoints included.
- [I] At 32 px or more, the endpoint effect is negligible.
- [U] Whether 32 px covers the decoder's edge effects (causal 3-D convolutions with upsampling) has not been measured. Edge artefacts get little weight at the ramp start.

**Temporal** [O]:

- Chunks of `temporal_tile_length` latents (default 16) with `temporal_overlap` (default 1).
- Each later chunk starts at `max(1, start − overlap − 1)`, so it never re-decodes latent 0.
- The first decoded frame of each later chunk is dropped. The overlapping frames are then blended.

**Precision** [O]: there are `working_device` (cpu / auto) and `working_dtype` (float16 / float32 / auto) options.

## What this means for our design

1. **Independent tiles are not a quality baseline.** They are a speed and memory baseline, with known seam and repetition causes: identical noise, a hard seam at the default overlap, and a global prompt per tile. We implement them only as a reference arm in the benchmarks.
2. **Blend ramps must exclude their endpoints**, or use a smooth window with a floor, so that `overlap = 1–2` still blends. A unit test can show the hard seam without any model.
3. **Noise is drawn once per canvas and sliced.** No tile ever re-seeds.
4. **Temporal joins:** crossfade with excluded endpoints, plus the causal first-latent rule from the fusion sampler.
5. **VAE decode tiling:** measure the decoder's edge-error profile against the distance to the tile border on real weights (owner GPU, ledger L7.107), then pick the overlap from that curve instead of a fixed 32 px.
