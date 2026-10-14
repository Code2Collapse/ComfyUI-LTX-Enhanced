# Architecture: resolution planning, padding and tiled denoising

Status: design for ledger L7.102 (integration targets) and L7.103 (planners). Written before the code, as the owner's engineering bar requires (A7). The decisions come from the [reverse-engineering notes](../re-notes/README.md).

## Goal

LTX video at DCP 2K (1920 × 1080 delivered, 1920 × 1088 sampled) and DCP 4K (4096 × 2160 delivered, 4096 × 2176 sampled), and at any custom size, with:

- no stretching or silent cropping of the source;
- tiled denoising whose seams and memory use are measured;
- a plan the user can read before anything runs.

## Constraints

| Constraint | Source |
|---|---|
| Clean room: no code from LTX-licensed, GPL or no-licence sources | [licence register](../licensing/register.md) |
| Integrate at run time: core through its Python API, upstream ComfyUI-LTXVideo through the graph only | register, D-1 |
| Latent geometry: 32 px per cell; 8 frames per latent frame plus one causal first frame; 128 channels | [core note](../re-notes/core-ltx-geometry-and-schedule.md) |
| Core's empty latent floors sizes, and its VAE encode centre-crops | core note |
| Core's token-count schedule saturates above ~2K and is NaN at DCP 4K | crash repro 001 |
| Planning code is testable without ComfyUI or a GPU | repository layout (`ltx_enhanced/` imports no ComfyUI) |
| Works inside subgraphs; never breaks saved workflows | standing orders, owner rule 2026-10-14 |

## Components

```
ltx_enhanced/            pure Python + torch, no ComfyUI import
  geometry.py            LatentGeometry presets; aligned sizes; 8n+1 frame rule
  padding.py             PadPlan: offsets, mode, frame padding; apply to images and masks; crop back
  tiles.py               axis split (pinned, overlap <= half a tile); trapezoid weights; coverage check; 3-D tile plan
  blend.py               accumulate x0 tiles and normalise
  schedule.py            sigma checks (NaN, non-monotonic, saturated); token counts per tile
  plan.py                one JSON-serialisable plan that joins the above, plus the human summary
nodes/ or experimental/  ComfyUI nodes, thin wrappers over ltx_enhanced
```

### Data flow

```
source frames ─► LTXE Resolution Plan ─► padded frames ─► (user's encode / guide nodes) ─► sampler
                    │ PLAN                                                                  ▲
                    │                                     MODEL ─► LTXE Tiled Model ────────┘
                    ▼                                                 (APPLY_MODEL wrapper)
decoded frames ─► LTXE Crop Back ─► delivered frames (exact source size and length)
```

- **LTXE Resolution Plan**:
  - Takes the source size and length, or an IMAGE batch.
  - Takes a target: DCP 2K, DCP 4K, or custom, plus a fit rule and a pad mode.
  - Returns the padded frames, a PLAN, the aligned width, height and length for `EmptyLTXVLatentVideo`, and a summary.
  - The summary covers tile counts, the tokens per tile, and a warning when the model would see a saturating token count.
- **LTXE Crop Back**: removes exactly the padding the PLAN recorded, in space and in time.
- **LTXE Tiled Model**:
  - Clones the MODEL and adds an APPLY_MODEL wrapper (core `WrappersMP.APPLY_MODEL`, chained, keyed `ltx_enhanced.tiles`).
  - Each model call: split the canvas into tiles; crop the input and the per-token keywords; call the next executor per tile; blend the x0 outputs with trapezoid weights; return the full canvas.
  - The sampler, its noise, its history and CFG are untouched.

## Padding rules

1. **Pad, never stretch.** Target-aligned sizes are multiples of 32. A source smaller than the target is padded. A source larger is refused unless the user chose a fit rule:
   - `pad only`: the default;
   - `fit inside then pad`: resize keeping the aspect ratio, then pad;
   - `crop to fill`: explicit, recorded.
2. **Offsets are recorded in the plan**: left, top, right, bottom in pixels, and frames added at the end.
   - Placement is `center` by default: the odd pixel goes to the bottom or right.
   - `top-left` is available for workflows that need exact pixel registration.
3. **Pad modes:**
   - `reflect`: the default for generation and refinement, so edges have plausible content;
   - `replicate`;
   - `constant` with a value.
   - Black is not a safe constant in latent space. Upstream issue #534 is about the VAE zero point, so padding is done in **pixel space before encode**. Its effect on the edge rows is a benchmark item.
4. **Frames:** pad to `8n + 1` by repeating the last frame. The plan records how many frames to drop after decode.
5. **One transform for everything.** Frames, masks, guide images and control videos all go through the same PadPlan, so they stay registered.

## Tile rules

- Per axis, the fewest tiles with overlap ≥ `min_overlap`, first and last pinned to the edges, the rest evenly spread. `min_overlap ≤ tile/2` is enforced.
- Weights:
  - default: a trapezoid with linear ramps over the overlap, endpoints excluded, so even a 1-cell overlap blends;
  - experimental: Gaussian, and Gaussian with a floor.
- Normalisation divides by the summed weights. Coverage is verified: every cell has a weight sum > 0, and with trapezoids exactly 1.
- Temporal tiles respect the causal first latent frame. A temporal tile that does not start at 0 is not a fresh clip. This is handled in L7.108; the first slice tiles **space only** and keeps all frames in each tile.
- Shared tokens:
  - audio (LTXAV) goes to every tile and is averaged;
  - text conditioning is shared;
  - per-token video keywords are cropped: `denoise_mask`, and guide tokens via `keyframe_idxs` / `guide_attention_entries`.

## Failure modes and responses

| Failure | Response |
|---|---|
| Sigmas NaN, non-decreasing, or saturated (more than half the steps ≥ 0.999) | The plan summary warns. `LTXE Tiled Model` refuses NaN sigmas with a plain-English message naming the cause and the fix (schedule from the tile size, or manual sigmas). |
| Source larger than the target with `pad only` | Refuse, and say which fit rule to choose |
| A cond keyword the wrapper does not know how to crop | Refuse with its name. Never silently pass a full-canvas tensor into a tile. |
| An IC-LoRA guide whose grid does not match the canvas (downscaled guide) | Refuse (first slice). Supporting it is a later item. |
| Batch > 1 | Tile each batch item the same way. Tiles are cut along H and W only. |
| One tile covers the canvas | The wrapper calls the model once, unchanged: exact equality with untiled sampling |
| Out of memory in a tile | Not caught in the first slice. The memory planner (L7.108) sizes tiles from measured VRAM. |

## Budgets

The first slice adds no VRAM beyond one tile's model call, plus the full-canvas accumulators:

- accumulators: two float32 canvases, about 120 MB at DCP 4K and 97 frames;
- CPU time: crop and blend are linear in the canvas size.

The planner's tile counts must match what the wrapper runs; tests compare them.

## Compatibility

- **Core:** tested on the owner's stack (0.39.0, frontend 1.53.10) and on core HEAD. Requires `WrappersMP.APPLY_MODEL`, present in core 0.33.0, 0.39.0 and 0.40.0. Checked at import, with a plain-English error if it is missing.
- **Upstream ComfyUI-LTXVideo:** graph-level only. Its guide nodes' outputs (latents with `noise_mask`, conditioning with guide entries) are inputs we must handle. Its own samplers are untouched.
- **Saved workflows:** new node ids `LTXE_*`, never renamed without a migration.
- **Subgraphs:** the nodes have no front-end code in the first slice. Backend behaviour does not depend on where the node sits; a live test runs one inside a subgraph.

## Kill switch

- Environment `LTXE_DISABLE_TILING=1`: `LTXE Tiled Model` returns the model unpatched, and logs once.
- `LTXE_DISABLE=1`: the pack registers no nodes. Saved workflows then show missing nodes instead of running something different.

## Testing

All of these run on CPU in `tests/`:

- planner: alignment, offsets, round trip pad → crop = identity, frame rule, refusals;
- tiles: coverage, weight sums, overlap ≤ half, count matches the summary;
- wrapper, with fake models:
  - a cell-wise model (tiled = untiled exactly);
  - a model that reads its tile's coordinates (proves the crop offsets);
  - a 1-tile canvas (identity);
  - cond keyword cropping, including `denoise_mask`;
  - a refusal for unknown per-token keywords;
- core integration (when core is importable): the wrapper registers through `add_wrapper_with_key` on a real ModelPatcher, and a tiny randomly initialised core LTXV transformer runs tiled and untiled on CPU.

GPU work for the owner (L7.107): seam and flicker metrics, time and peak VRAM at DCP 2K and 4K, against LTXVTiledFusionSampler and untiled sampling.
