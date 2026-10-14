# LTX-2 official tiled diffusion (ltx-pipelines / ltx-core)

**Source:** [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) at `be6d6d0a69` (2026-10-13). Files:

- `packages/ltx-pipelines/src/ltx_pipelines/utils/tiled_diffusion.py` (TiledDiffusionModel);
- `packages/ltx-core/src/ltx_core/modality_tiling.py` (VideoModalityTilingHelper);
- `packages/ltx-core/src/ltx_core/tiling.py` (masks and split rules);
- `ltx_core/multigpu/transformer/tiled_data_parallel.py` (the multi-GPU form).

**Licence:** LTX-2 Community License. Behaviour notes only.

**Markers:** [O] observed, [I] inferred, [U] unknown.

**Also seen:** Arturogpj/wan2gp-ltx-tiled-refine `eac68e6964` (Apache-2.0 label). Its module docstring says it ports `TiledDiffusionModel` / `VideoModalityTilingHelper` onto WanGP, so it is treated as LTX-licensed for reuse. Its documented settings are cited below as behaviour. [O]

## Architecture: blend predictions, keep one sampler

[O, class docstring and `forward`]:

- **The sampler owns one full-canvas latent and steps it once per diffusion step.**
- The tiling sits inside the model call. Each call:
  1. evaluates the transformer on each video tile in turn;
  2. blends the tiles' **x0 predictions** with separable trapezoid masks;
  3. returns one full-canvas prediction.
- Blending x0 equals blending velocities when the weights sum to 1, because `to_denoised` is affine in the velocity and every tile sees the same per-token timesteps (the docstring's argument). [O]

**Audio (LTX-2 joint audio-video)** [O]:

- Every tile gets the **full audio stream**, with a different video context.
- The audio predictions are summed and divided by the number of tiles.

**Conditioning tokens (IC-LoRA guides)** [O]:

- A tile keeps the guide tokens that overlap it.
- Each kept guide token gets the weight `1 / (number of tiles that keep it)`.

**Positions** [O]: `normalize_positions` (default on) shifts each tile's positions to start at zero.

**Multi-GPU** [O]: the same helper spreads tiles round-robin across ranks.

**Consequence** [I]:

- Any sampler works unchanged: history methods (multi-step) and stochastic methods (ancestral, SDE). They see one canvas and one noise draw per step, so there is no per-tile re-seeding.
- This removes failure modes 1 and 2 of the ComfyUI fusion sampler ([note](ltxvideo-tiled-fusion-sampler.md)).

## Split rules and masks

| Rule | Behaviour | Marker |
|---|---|---|
| Trapezoid mask | Linear ramps over the overlap, flat elsewhere. **Endpoints excluded**: a ramp of length r takes r interior values of a 0 → 1 line with r + 2 points, so a 1-cell overlap still blends 50/50. A causal option starts the first temporal tile's ramp at 0. | O |
| `split_by_size(size, overlap)` | Fixed stride `size − overlap`. The last tile ends at the edge. An optional minimum size grows a short last tile leftwards and widens the previous ramp to match. Validation checks coverage and that ramps equal overlaps. | O |
| `split_by_size_pinned(size, min_overlap)` | First and last tiles pinned to the edges, the rest evenly spread, with the fewest tiles that give at least `min_overlap`. **`min_overlap ≤ size/2` is enforced**: a tile never overlaps the one two steps away. | O |
| Temporal split | In pixel frames, with causal handling of the first latent frame. Optional **seams**: on known handover cells, the overlap is denoised for context and then dropped, not blended (used by DFR stages). | O |

**Comparison with the ComfyUI fusion sampler** [I]:

- The ComfyUI node allows `overlap_frac` up to 0.75, so a cell can be covered three times. The official split forbids that.
- The ComfyUI node uses a Gaussian weight with no floor. The official code uses trapezoids, which give weight 1 away from the overlaps.

## Settings found in the field

[O, Arturogpj README and docstring; claims not reproduced]:

- IC-LoRAs run on their trained windows: Refine Details 1024 × 576, Restore 960 × 544.
- Tile presets at 1920 × 1088:

  | Preset | Overlap | Windows |
  |---|---|---|
  | quality (Lightricks default) | half a window | 9 |
  | balanced | ≥ 128 px | 6 |
  | fast | ≥ 32 px | 4 |

- A two-phase pipeline: phase 1 is a half-size draft. When the draft is at most 1.6 × the trained window area, it runs as one window.
- Each window gets its own runtime cache.

## What we take from this

1. **Architecture: tile inside the model call, blend predictions, leave the sampler alone.**
   - In ComfyUI this is a model wrapper that crops input, timestep, per-token denoise mask and guide tokens per tile.
   - It replaces "call the sampler per tile".
   - It is our primary design. Per-step latent fusion, as the ComfyUI node does it, stays as a comparison arm.
2. **Trapezoid ramps with excluded endpoints** as the default weight. The Gaussian and a "Gaussian with floor" are experimental options, chosen by seam metrics.
3. **Overlap at most half a tile**, by construction.
4. **Audio is shared by all tiles and averaged.** We do the same. Whether that averaging hurts lip sync at 4K is a GPU benchmark question (owner, L7.107).
5. **Shared guide tokens** weighted `1/k`.
6. **Seam-dropping temporal handovers** as an alternative to blending, to compare against crossfades for flicker.
