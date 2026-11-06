# ComfyUI-LTX-Enhanced

High-resolution LTX video generation for ComfyUI, aimed at DCP 2K (1920×1088) and DCP 4K (4096×2176) frames on real hardware. The pack will provide:

- a hardware-aware resolution and memory planner;
- spatial and temporal tile planners;
- padding, alignment and crop-back;
- overlap blending and seam reduction.

Each method is measured against the existing ones before it is recommended.

> **Status: first experimental nodes.** All five nodes are *experimental*: implemented and tested on CPU, including inside ComfyUI's real sampler with tiny random LTX models, but **not yet benchmarked on real LTX weights**. Image-quality claims wait for those GPU runs ([stability labels](docs/stability.md)).

## Nodes

The planning and tiling nodes are under **🐺 C2C / 🎥 LTX Enhanced** in the Add Node menu.

| Node | What it does | Label |
|---|---|---|
| **LTXE Resolution Plan** | Plans an LTX canvas for a size and length: the aligned width, height and 8k+1 frame count to give `Empty LTXV Latent Video`, the padding to remove afterwards, the tile grid, and the token counts | experimental |
| **LTXE Pad Frames** | Pads source frames, and an optional mask, into the aligned canvas: no stretching, no silent cropping. It returns the plan. | experimental |
| **LTXE Crop Back** | Uses the plan to remove exactly the padding: the extra frames at the end and the border. Works after a 2× upscale too. | experimental |
| **LTXE Tiled Model** | Makes every model call run tile by tile, and blends the tiles' predictions with trapezoid weights. Your sampler, noise and schedule are unchanged, so any sampler works, ancestral and multi-step ones included. In an LTX-2 audio-video latent, the audio is shared by all tiles. Full-size IC-LoRA guides are cut with the tiles. | experimental |
| **LTXE Sigma Check** | Passes a schedule through, and stops the run with a plain-English message when it is NaN, rising, flat or saturated | experimental |

### SDR → HDR helpers (new 2026-11-04)

These twelve are under **🐺 C2C / 🎥 LTX Enhanced / HDR**. They prepare, plan and deliver an SDR clip around the
official LTX-2.5 SDR-to-HDR IC-LoRA graph (ComfyUI-LTXVideo + core nodes). They do not run the model themselves. The design is in
[docs/HDR_ARCHITECTURE.md](docs/HDR_ARCHITECTURE.md).

| Node | What it does | Label |
|---|---|---|
| **LTXE HDR Source Audit** | Checks the clip before any model loads: size, frame rate, bit depth (estimated from the values), clipped highlights, near-white areas, crushed shadows, duplicate frames, likely cuts, softness and an upscale hint. Material that is already HDR is sent to the Native HDR path. | experimental |
| **LTXE HDR Input Color** | Converts the source into ACEScct before resizing, as the official graph does. Ordinary video and linear Rec.709 go through core's Convert Image Color Space; ACEScg uses this pack's ACEScct curve, because core has no ACEScg input. The untouched source is passed on. | experimental |
| **LTXE HDR Chunk Planner** | Plans overlapping 8k+1-frame windows that deliver every frame exactly once. It pads the last window, conditions at min(source, 30) fps while keeping the source rate and frame numbers, and estimates tokens. | experimental |
| **LTXE HDR Take Segment** | Cuts one planned window, padded to 8k+1, and reports which of its frames are delivered. | experimental |
| **LTXE HDR 8-bit Cleanup** | Removes 8-bit banding before the model: fills in the values between code steps while every pixel stays within half a code of the source, so re-quantising gives back exactly the source (checked on each run). Only 8-bit sources are changed. Measured on synthetic ramps: the error to the true gradient drops from 0.29 to 0.02 codes. | experimental |
| **LTXE HDR Highlight Analyzer** | Finds clipped highlights in the SDR source (a channel at the code ceiling; a bright but unclipped area is not counted) and a soft near-white map, with grow, feather, temporal smoothing and a user mask. | experimental |
| **LTXE HDR Synth Mask** | Decides where the generated HDR is used: the clip mask (optionally the near-white map), grown, feathered, smoothed over time, plus your own mask, with a red preview and coverage / flicker numbers. The model still sees the whole frame. | experimental |
| **LTXE HDR Shadow Recovery** | Optional, off by default: where the source is crushed to black, uses the generated shadow detail at the source's local exposure (measured from the just-above-black pixels around it). Outside those areas the result is exactly the source; inside it never goes below the source. | experimental |
| **LTXE HDR Source-Preservation Composite** | Keeps the source outside the synth mask and takes the generated HDR inside it, in linear ACEScg: where the mask is 0 the result is exactly the source in that space. An optional per-channel gain matches the generation to the source exposure; the source is never changed. | experimental |
| **LTXE HDR Seam Exposure Match** | Matches a window's exposure to the window before it: the frames it shares with the previous window (already written as EXR) measure the step, and the correction fades out over `seam_width` frames, so the rest of the window is untouched. A difference above `max_stops` is reported, not hidden. Run the windows in order. | experimental |
| **LTXE HDR QC** | Checks a delivered shot from the output folder: every frame present once, readable and free of NaN / Inf, EXR header (precision, compression, ACEScg primaries, fps), exposure jumps between frames (seams) and flicker. It recommends accept, review or hold, and never re-renders by itself. | experimental |
| **LTXE HDR Delivery** | Writes the delivered frames of one decoded window as scene-linear ACEScg EXRs (half float, ZIP, no clamping), named after the source frames (`shot_hdr_1001.exr` ...), so the windows of a shot fill one sequence (one number width for the whole shot). Adds a JSON sidecar per window. Files stay inside ComfyUI's output folder. | experimental |

**Status:**
- Tested on CPU, and run in ComfyUI's real executor (core 0.39.0).
- The HDR model run itself (LTX-2.5 22B on a GPU) has **not** been benchmarked by this pack.
- `LTXE_DISABLE_HDR=1` removes just these twelve nodes.

**Workflow** `workflows/2.5/hdr/LTXE_SDR_to_HDR_Pro.json`: SDR video in, ACEScg EXR frames out, one window per
run, plus an HLG clip and a tonemapped preview per window. Only the clipped highlights come from the generation;
the rest of the frame is the source. It needs ComfyUI-LTXVideo and the official LTX-2.5
SDR-to-HDR files. It was built and checked in ComfyUI 0.39 / frontend 1.53.10 (it loads in both renderers; the
server reports only the model and video files that were absent there); the model itself was not run. Seams between
windows are matched in exposure but not cross-faded, and audio is not muxed yet.

`workflows/2.5/hdr/LTXE_SDR_to_HDR_Pro_Subgraphs.json` is the same graph with its stages packaged as editable
native subgraphs (guide and latents, sampling, decode and EXR, source preservation, seam match, HLG and preview):
source, plan and models stay outside where you set them. Its run is identical (the API prompt was checked equal to
the flat graph's), and it loads in both renderers on ComfyUI 0.39 / frontend 1.53.10.

**Workflow** `workflows/2.5/hdr/LTXE_HDR_Review_and_QC.json`: run it on the delivered folder when every window
is done. It was run for real in ComfyUI 0.39: a complete two-window shot is accepted; with one frame deleted it
holds and names the frame.

**Workflow** `workflows/2.5/hdr/LTXE_HDR_Native_EXR.json` (experimental): an EXR plate that is already HDR, refined
by the official LTX-2.5 general-refine IC-LoRA (tiled fusion, float32 VAE) and written as ACEScg EXR frames by
source frame, one window per run. It is not the SDR-to-HDR conversion, and the prompt is encoded locally (the
official graph's paid Gemma API node is not used). Built and checked in ComfyUI 0.39 / frontend 1.53.10: 60 of 60
links, loads clean in both renderers, the server reports only the absent model files. The model was not run.

### Why padding matters

ComfyUI core floors the empty-latent size and centre-crops on VAE encode. So a 1920 × 1080 clip is sampled at 1920 × 1056, and its top and bottom 12 rows are lost.

DCP needs 1920 × 1088 (2K) and 4096 × 2176 (4K) canvases, cropped back to 1920 × 1080 and 4096 × 2160. **LTXE Pad Frames → … → LTXE Crop Back** does exactly that:

- the padding is reflected by default;
- the offsets are recorded in the plan;
- the frame count is padded to 8k+1 and trimmed again.

`custom_width` / `custom_height` appear only for target *custom*, and `pad_value` only for pad mode *constant*. This works in both renderers and inside subgraphs.

With a VAE connected (optional), the plan reads the VAE's own scale factors instead of assuming the LTX geometry of 32 px per cell and 8 frames per latent frame.

### Why the schedule matters

Core's `LTXVScheduler` and `ModelSamplingLTXV` derive their shift from the latent's token count. Measured on core 0.39.0 and 0.40.0 ([crash repro 001](crash_repros/001-ltxv-schedule-at-4k/)):

- `ModelSamplingLTXV` with a generic scheduler keeps every step at σ ≥ 0.999 from DCP 2K;
- `LTXVScheduler` becomes a staircase at 2560 × 1440, flat 0.1 at 3840 × 2176, and **NaN** at DCP 4K.

When tiling, give the scheduler a **tile-sized** latent, since the model sees tiles. Put **LTXE Sigma Check** after the scheduler.

### Wiring for a tiled run

```
Load model ─► LTXE Tiled Model ─► (your sampler: SamplerCustomAdvanced / KSampler)
LTXE Resolution Plan ─► width / height / length ─► Empty LTXV Latent Video ─► sampler
Empty LTXV Latent Video (TILE size) ─► LTXVScheduler ─► LTXE Sigma Check ─► sigmas
decoded frames ─► LTXE Crop Back (plan) ─► Save
```

**IC-LoRA guides** (`LTXV Add Guide`, IC-LoRA guide nodes) work when attached at full size:

- the guide frames are cut with each tile;
- their positions are moved to the tile's origin;
- each guide's strength and attention mask follow along.

**Not supported by LTXE Tiled Model:** downscaled guides (downscale factor above 1) and ControlNet. The node stops with a message naming the problem, instead of passing whole-frame data into a tile.

**Long clips:** temporal windows come from ComfyUI core's own **LTXV Context Windows** node, which handles
windows, overlap, the causal first frame and guides per window. LTXE Tiled Model then tiles each window spatially.
The two were tested together in either order.

**Out of memory:** with `on_out_of_memory` = *shrink tiles* (the default), a tile that runs out of GPU memory:

- frees memory;
- halves the larger tile side and retries, down to 128 px;
- keeps the smaller size for the rest of the run.

A tile as large as the frame therefore means "tile only if needed". Tested on CPU with a model that fails above a
memory budget. Behaviour on a real GPU is still to be measured.

**Kill switches:**

- `LTXE_DISABLE_TILING=1` turns LTXE Tiled Model into a pass-through;
- `LTXE_DISABLE=1` registers no nodes.

### Evidence for this release

- 115 tests on CPU, run on the developer's machine (the suite is not published: owner rule 2026-10-23), covering:
  - pad → crop is the identity for every pad mode and placement;
  - exact DCP offsets;
  - tile coverage and weight sums;
  - a per-cell model tiled = untiled;
  - packed audio-video;
  - guide-token cropping and shifting;
  - refusals.
- The same suite runs LTXE Tiled Model inside core's own sampler, with tiny random LTX-Video and LTX-2 audio-video models:
  - one tile is bit-identical to a plain call;
  - several tiles run and blend;
  - a guide attached by core's own `LTXVAddGuide` path is tiled (9 tiles × 3 steps);
  - a downscaled guide is refused;
  - core's LTXV Context Windows plus spatial tiles works in both wrapping orders (9 tiles × 3 steps × 3 windows).
  - These are integration checks, not quality results.
- Live on the owner's stack (core 0.39.0, frontend 1.53.10): all nodes render in the classic and Nodes 2.0 renderers. A real run works at the root and **inside a subgraph**:
  - pad → crop-back returns the exact source size;
  - Sigma Check reports the NaN schedule from a DCP-4K latent, or stops the run with the message.

## What it works with

The pack runs inside ComfyUI and uses the LTX support that you install yourself: ComfyUI core's own LTX nodes, or [Lightricks/ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo). It connects to them only through the node graph, by passing models, latents, conditioning and images between nodes. It contains no Lightricks code. It never downloads or bundles model weights.

**Model licence.** The LTX-2, 2.3 and 2.5 weights are under the LTX-2 / LTX-2.x Community License Agreements, and those terms apply to you when you use them:

- use restrictions in Attachment A;
- entities with annual revenue of at least $10M need a paid commercial licence from Lightricks.

See the [model registry](docs/registry/models.md) for the licence of each model repository.

## Clean room, Apache-2.0

This pack is Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)); the few files ported from the author's other Apache packs are listed in [NOTICE.md](NOTICE.md). It learns from many implementations whose licences do not allow their code into an Apache project: the LTX Community Licenses, GPL tiling packs and ComfyUI core. From those it takes behaviour notes only, written in our own words. The planners and blending are written from scratch, from published papers and those notes.

- [Licence register](docs/licensing/register.md): every source, dependency and integration point, its licence, and what may be reused. Open points are listed there too.
- [Clean-room procedure](docs/licensing/clean_room.md): how reading and writing are kept apart, and how a release is checked.

## Research record

| Record | What it holds |
|---|---|
| [Ecosystem registry](docs/registry/ecosystem.md) | 40 repositories: official, core, tiling, community, acceleration, quantisation. Each has its revision, licence and reuse rule. |
| [Model registry](docs/registry/models.md) | 68 Lightricks model repositories with their model-card licences |
| [Papers](docs/papers/README.md) | 17 papers with verified arXiv ids: LTX-2, LTX-Video, MultiDiffusion and successors, long-video and video super-resolution work |
| [Known upstream issues](docs/registry/known_issues.md) | 17 upstream failures to reproduce: OOM, size rules, VAE zero-point, NaN, portability |
| [Reverse-engineering notes](docs/re-notes/) | How the existing implementations actually work: upstream's fusion, tiled and looping samplers, tiled VAE decode, LTX-2's official tiled diffusion, core LTX geometry. Includes a comparison table and the design decisions they support. |
| [Crash reproductions](crash_repros/) | 001: core's LTX schedule saturates above 2K and is NaN at DCP 4K (core 0.39.0 and 0.40.0) |

The JSON files next to each table are the record. Run `python tools/registry.py render` after editing one. `python tools/registry.py drift` shows which recorded repositories have moved since.

## Repository layout

| Folder | Contents |
|---|---|
| `ltx_enhanced/` | Model-agnostic core: planners, padding, blending. Imports no ComfyUI. |
| `nodes/` | Production ComfyUI nodes (none yet: a node moves here once benchmarked) |
| `experimental/` | Experimental nodes: the five LTXE nodes above |
| `benchmarks/` | Benchmark scripts and raw results |
| `crash_repros/` | Minimal reproductions of every significant defect, with sanitized logs |
| `.github/` | CI: boots a fresh ComfyUI with only this pack and fails on an import failure, no registered node or a traceback (the test suite stays on the developer's machine) |
| `workflows/` | Example workflows, built from scratch |
| `docs/` | Registry, papers, reverse-engineering notes, licensing, stability labels |
| `release_notes/` | One file per release |

## Roadmap

1. Inventory: ecosystem, models, papers, known issues, licences. **Done.**
2. Reproduce the official and community workflows, including the known failures. (Needs GPU runs with LTX weights, done on the owner's machine.)
3. Reverse-engineer the most relevant implementations and compare their algorithms. **First pass done:** [notes](docs/re-notes/README.md).
4. Reproducible baselines on the available hardware. (GPU: owner.)
5. Original resolution planning, tiling, padding and blending. **First slice done:** planner, padding, crop-back, tiled model call, sigma check (experimental).
6. Benchmark existing techniques. Add the inference and quality optimisations that measure well.
7. Test combined methods for regressions and interactions.
8. Publish verified code, reproducible experiments and transparent results.

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
