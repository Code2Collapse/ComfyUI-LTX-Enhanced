# ComfyUI core: LTX latent geometry and the token-count schedule

**Source:** [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI). Read at `e0a32bce84` (2026-10-13, version 0.40.0) and measured on the owner's stack (0.39.0).

Files:

- `comfy/latent_formats.py` (LTXV, LTXAV);
- `comfy_extras/nodes_lt.py` (EmptyLTXVLatentVideo, ModelSamplingLTXV, LTXVScheduler);
- `comfy/sd.py` (VAE scale formulas, `vae_encode_crop_pixels`);
- `comfy/k_diffusion/sampling.py` (`default_noise_sampler`).

**Licence:** GPL-3.0. This note describes behaviour only. The pack calls core at run time.

**Markers:** [O] observed, [I] inferred, [U] unknown.

## Latent geometry

| Fact | Value | Marker |
|---|---|---|
| Latent channels (LTXV, LTXAV video stream) | 128 | O |
| Spatial compression | 32 px per latent cell | O |
| Temporal compression | 8 frames per latent frame. The first latent frame encodes 1 pixel frame (causal) | O (formula `(8, 32, 32)`, downscale `floor((t + 7)/8)`) |
| Empty latent size | `[B, 128, (L−1)//8 + 1, H//32, W//32]` | O |
| Widget steps | width / height step 32, length step 8 | O |

**Consequences** [I, arithmetic]:

- The empty latent **floors**. A 1080-row request gives 33 latent rows, which decode to 1056 rows. A length of 100 gives 13 latent frames, which decode to 97 frames. The extra rows and frames are dropped silently.
- **VAE encode center-crops** each spatial dimension to a multiple of 32 [O, `vae_encode_crop_pixels`]. A 1920 × 1080 frame loses 12 rows top and bottom (1920 × 1056).
- So a DCP frame survives a round trip only if it is padded to 1088 (2K) or 2176 (4K) before encode, and cropped back after decode. **Core does neither.**

## The token-count schedule

[O] `ModelSamplingLTXV` and `LTXVScheduler` derive the shift from the latent's token count `N = F·H·W` (latent frames × rows × columns):

- the shift is linear in `N`: `shift(N) = N·(max_shift − base_shift)/3072 + base_shift − 1024·(max_shift − base_shift)/3072`;
- the defaults are 2.05 / 0.95, so `shift(1024) = 0.95` and `shift(4096) = 2.05`;
- LTXVScheduler maps a linear grid `t` through `e^s / (e^s + (1/t − 1))`, then stretches the result so the last non-zero sigma equals `terminal` (0.1).

Measured with core's own nodes, on CPU and without weights, for 20 steps and 97 frames. The ModelSamplingLTXV rows use the `simple` scheduler.

| Latent size | Tokens N | Shift | LTXVScheduler (20 steps) | ModelSamplingLTXV + `simple` (20 steps) |
|---|---|---|---|---|
| 768 x 512 | 4 992 | 2.37 | normal: 1.0, 0.993 ... 0.36, 0.1 | normal: 1.0, 0.995 ... 0.36 |
| tile 1024 x 576 | 7 488 | 3.27 | normal | normal: 1.0 ... 0.58 |
| 1280 x 704 | 11 440 | 4.68 | normal | slow: 2 steps >= 0.999, last 0.85 |
| 1920 x 1088 (DCP 2K) | 26 520 | 10.08 | normal: 1.0 ... 0.57, 0.1 | **saturated**: all 20 steps >= 0.999 |
| 4096 x 2176, 9 frames | 17 408 | 6.82 | normal | 10 steps >= 0.999, last 0.98 |
| 2560 x 1440 | 46 800 | 17.34 | **staircase**: 1.0 x 14, 0.82 x 4, 0.64, 0.1 | saturated |
| 3840 x 2176 | 106 080 | 38.57 | **flat 0.1** | saturated (1.0) |
| 4096 x 2176 (DCP 4K) | 113 152 | 41.10 | **NaN x 20** | saturated (1.0) |

Raw results:

- workspace evidence `docs/evidence/L7.106/ltxv_sigmas_core0.39.0.json`;
- `…_core_head_e0a32bce84.json` (core 0.40.0, identical numbers);
- the probe script is in [crash_repros/001-ltxv-schedule-at-4k](../../crash_repros/001-ltxv-schedule-at-4k/).

[I] Reading:

- **ModelSamplingLTXV saturates from DCP 2K up:** with a generic scheduler, every step but the last stays at σ ≥ 0.999, then jumps to 0.
- **LTXVScheduler survives 2K.** It degrades to a staircase at 2560 x 1440 and collapses to a constant 0.1 at 3840 x 2176.
- At DCP 4K the stretch divides 0 by 0 and every sigma is NaN. A NaN schedule turns the whole video to NaN, which would show up as the blank or NaN outputs reported upstream (ComfyUI-LTXVideo #512, #523). [U] That link is not proven.
- The shift depends on the sequence length the model sees. In tiled sampling the model sees a **tile**, not the canvas. So the schedule that matches the model's training regime is the one computed from the tile's token count. A 1024 × 576 tile gives the normal 3.27.

## Ancestral noise

[O] `default_noise_sampler(x, seed)` builds a new generator seeded with `seed` on every call, drawing `x.size()` noise.

[I] A per-step, per-tile tiling loop that calls the sampler function with the same seed therefore injects one fixed noise pattern per tile shape. The pattern repeats across tiles and across steps. See [the fusion sampler note](ltxvideo-tiled-fusion-sampler.md), failure mode 1.

## Design rules taken from this

1. **Pad, never crop or stretch:**
   - pad to the aligned size before encode;
   - record the offsets;
   - crop back exactly after decode;
   - frames: extend to `8n+1` and drop the added frames after decode.
2. **Schedule from the tile.** The planner reports the token count the model will see and the resulting shift.
3. **Refuse a schedule that is NaN or saturated.** The planner reports it as a plain-English error before sampling.
4. **Noise:** one draw per canvas per step, sliced per tile.
