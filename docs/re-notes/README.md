# Reverse-engineering notes

One note per implementation studied: what it does, at which revision, with every statement marked *observed*, *inferred* or *unknown*. Notes are prose, maths and pseudocode in our own words. See [the clean-room procedure](../licensing/clean_room.md).

On method: these sources are complete repositories, so they were read directly. REA's binary and package analysers apply to shipped artefacts, such as compiled kernels and the desktop app builds, and will be used when one of those becomes relevant.

| Note | Covers |
|---|---|
| [ltxvideo-tiled-fusion-sampler.md](ltxvideo-tiled-fusion-sampler.md) | ComfyUI-LTXVideo `LTXVTiledFusionSampler`, its tile plan, and `LTXVGetTilingSizes` |
| [ltxvideo-tiled-looping-vae.md](ltxvideo-tiled-looping-vae.md) | `LTXVTiledSampler`, `LTXVLoopingSampler`, the extend sampler, tiled VAE decode |
| [ltx2-official-tiled-diffusion.md](ltx2-official-tiled-diffusion.md) | LTX-2 `TiledDiffusionModel`, `VideoModalityTilingHelper`, trapezoid masks and split rules; WanGP port settings |
| [core-ltx-geometry-and-schedule.md](core-ltx-geometry-and-schedule.md) | Core LTX latent geometry, encode crop, the token-count schedule (measured), ancestral noise seeding |

## How the implementations compare

| | LTXVTiledSampler | LTXVLoopingSampler | LTXVTiledFusionSampler | LTX-2 TiledDiffusionModel |
|---|---|---|---|---|
| Coupling | none: each tile is a full run | none spatially; autoregressive in time | per step, blending stepped latents | per step, blending x0 predictions |
| Who owns the sampler loop | core, per tile | core, per tile and chunk | the node (one call per tile per step) | core, once for the whole canvas |
| Samplers that work | any | any | single-step methods only; ancestral noise repeats per tile and step | any |
| Spatial weight | linear ramp with endpoints: **hard seam at overlap 1–2** | same | Gaussian, no floor | trapezoid, endpoints excluded |
| Noise | **identical per same-sized tile** | distinct seed per tile and chunk | one canvas draw (deterministic samplers) | one canvas draw |
| Overlap limit | 1–8 latent cells | 1–8 | fraction 0.25–0.75 | ≤ half a tile |
| Temporal | none | extend chunks, overlap conditioning, AdaIN | per-window fresh guide encode, skip the causal first latent | causal split, optional seams |
| Audio (LTX-2) | not handled | not handled | refused (needs a 5-D latent) | shared by all tiles, averaged |
| Use | any latent | T2V / I2V / long clips | IC-LoRA guide latents only | IC-LoRA stages in Lightricks pipelines |

## Integration point in ComfyUI (verified at core e0a32bce84)

Core runs a chain of **APPLY_MODEL wrappers** (`comfy.patcher_extension.WrappersMP.APPLY_MODEL`, added with `ModelPatcher.add_wrapper_with_key`):

- Each wrapper receives the full-canvas input, sigma and conditioning keywords.
- It returns the **denoised (x0)** prediction.
- Wrappers chain, so another pack's wrapper keeps working. The older single `model_function_wrapper` slot does not chain.

That is where per-tile evaluation with x0 blending fits: the official architecture, inside an unchanged ComfyUI sampler.

The LTX keywords a tile wrapper must handle (`comfy/model_base.py` `LTXV` / `LTXAV`):

| Keyword | Per-tile handling |
|---|---|
| `denoise_mask` | crop per tile (per-token timesteps) |
| `keyframe_idxs` and `guide_attention_entries` | crop and shift per tile |
| `latent_shapes` | AV latents arrive packed. Unpack, tile the video part, and repack with the tile's shapes |
| `audio_denoise_mask`, `ref_audio` | shared, not cropped |
| `c_crossattn`, `attention_mask`, `frame_rate` | shared |
| `generated_keyframes` | [U] meaning per tile still to be read |

## Design decisions these notes support

Each decision is checked by a benchmark before it reaches production.

1. **Tile inside the model call** (APPLY_MODEL wrapper), blend x0 predictions, leave the sampler alone. This is our primary path. Per-step latent fusion and independent tiles stay as benchmark arms.
2. **Trapezoid weights with excluded endpoints**, overlap at most half a tile. Gaussian weights are an experimental option.
3. **Pad and crop back**, never stretch or crop the source:
   - DCP 2K 1920 × 1080 → 1920 × 1088;
   - DCP 4K 4096 × 2160 → 4096 × 2176;
   - frames → 8n+1.
4. **Schedule from the tile token count.** Refuse NaN or saturated sigmas, which core gives above about 2K ([crash repro 001](../../crash_repros/001-ltxv-schedule-at-4k/)).
5. **One noise field per canvas.** Never re-seed per tile.
6. **Audio is shared across tiles and averaged.** Guide tokens are shared with weight 1/k.
7. **Temporal windows:** causal first-latent handling. Crossfade or seam-drop, chosen by a flicker metric.
