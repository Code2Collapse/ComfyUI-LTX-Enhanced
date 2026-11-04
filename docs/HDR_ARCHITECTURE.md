# LTX-Enhanced HDR — architecture (design before code, owner A7)

Owner directive A23 (2026-11-04; verbatim in the workspace at `docs/evidence/A23/`): a native ComfyUI SDR -> HDR
reconstruction and compositing system on the official LTX-2.5 SDR-to-HDR IC-LoRA. Stability label for everything
below: **experimental** until a real LTX-2.5 GPU run is benchmarked (`docs/stability.md`).

## 1. What the official pipeline really is (verified 2026-11-04)

Read for behaviour only (clean-room, `docs/licensing/clean_room.md`) from Lightricks/ComfyUI-LTXVideo `220e1d4`
(2026-10-20), `example_workflows/2.5/HDR_workflows/LTX-2.5_ICLoRA_SDR_to_HDR_Distilled.json` and
`LTX-2.5_V2V_TiledFusion_SDR_to_HDR.json`, plus ComfyUI core 0.39.0:

| Stage | Official node(s) | Settings in the official graph |
|---|---|---|
| Input colour | core `ImageColorSpace` | `sRGB` -> `HDR ACEScct`, BEFORE resizing |
| Resize | core `ResizeImageMaskNode` | scale to a multiple of 32, lanczos |
| Scene embeddings | `LTXVLoadConditioning` | `ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors`, to positive AND negative; no prompt |
| IC-LoRA | `LTXICLoRALoaderModelOnly` | `ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors`, strength 1 |
| Guide | `LTXAddVideoICLoRAGuide` | strength 1, the full ACEScct source |
| Conditioning fps | `LTXVConditioning` + `ComfyMathExpression` | `a if a <= 30 else 30`; output keeps the source rate |
| Sampling | `SamplerCustomAdvanced`, `CFGGuider` 1, euler | 8 steps, `ManualSigmas` 1.0 0.99375 0.9875 0.98125 0.975 0.909375 0.725 0.421875 0.0 |
| Decode | `LTXVVAEForceFloat32` + `VAEDecodeTiled` + `LTXVHDRDecodePostprocess` | ACEScct decode -> ACEScg EXR (`save_exr`, `exr_color_space` acescg) |
| HLG master | core `ImageColorSpace` ACEScct -> `HDR` + core `CreateVideo` + `SaveVideo` | Rec.2020 HLG |
| Large canvas | `LTXVGetTilingSizes` + `LTXVTiledFusionSampler` (+ x2 latent upscaler stage) | official tiled path |

Names in the directive that do **not** exist at this revision: `LTXVSDRToHDRWorkingSpace`, `LTXVSaveHLG`, "LTXV Load
Fixed Scene Embeddings". We wire the real nodes above. No seam-keyframe node appears in these graphs; the directive's
seam values (0.95 / 1.0) are unverified in ComfyUI and are not exposed until an official source is found.

Core `ImageColorSpace` offers sRGB, HDR (HLG), HDR PQ, linear (Rec.709), HDR LogC3 and HDR ACEScct, but **no ACEScg
input**. LTX-E adds only that missing curve (ACEScg -> ACEScct, same AP1 primaries) and is tested against core.

## 2. Division of work

- **Official / core nodes do the model work**: loading, IC-LoRA, guides, sampling, decode, EXR from the decoder, HLG.
  LTX-E never re-implements or copies them (LTX-2 Community License); it wires them in example workflows.
- **LTX-E adds what the official graph lacks**, as model-independent code in `ltx_enhanced/hdr/` with thin ComfyUI
  wrappers in `experimental/hdr_nodes.py`:
  - **source audit** (before any model load);
  - **input colour dispatch** (core for sRGB / linear, ours for ACEScg, passthrough for ACEScct);
  - **chunk planner** (8k+1 segments, overlap, exact frame accounting, conditioning-fps vs output-fps);
  - later: masks, preservation composite, seam exposure match, QC, delivery manifest, cache.
- Reuse: `LTXE Resolution Plan` / `Pad Frames` / `Crop Back` for geometry. `LTXE Tiled Model` is NOT offered for HDR
  until benchmarked against the official Tiled Fusion path (P4).

## 3. Constraints

- Frame counts: the video VAE encodes 8k+1 frames (`ltx_enhanced/geometry.py`); a segment is always 8k+1 frames.
- Conditioning fps <= 30; delivered fps = the source's exact rate (kept as a fraction where known).
- The IC-LoRA sees the full frame; masks are used only AFTER generation (preservation composite), never as model input.
- Colour: transform before resize; every transform is named in the manifest; no double transform.

## 4. Failure modes and how they surface

| Failure | Behaviour |
|---|---|
| ACEScct source sent through an sRGB transform (double transform) | input dispatch refuses an already-HDR choice it cannot verify; manifest records the transform |
| HDR / EXR source routed to SDR-to-HDR | source audit flags float / > 1.0 input as "already HDR: use the Native HDR workflow" |
| Segment not 8k+1, or frames lost / duplicated at seams | planner refuses in words; exact-cover invariant tested |
| Conditioning at > 30 fps | planner emits `conditioning_fps` = min(src, 30); output fps never changed |
| Token count in the measured-unsafe range | the existing plan warnings (`plan.py`) are carried into the HDR plan |
| Missing weights | workflows use the official loaders; LTX-E never downloads |

## 5. Budgets (estimates, labelled; no measurements yet)

The planner reports per-segment token counts and a pixel-buffer estimate (frames x H x W x 3 x 4 bytes) as
ESTIMATE. The HDR Converter author's published timings are not LTX-E numbers and are not quoted as such. Real
budgets come from P5 GPU benchmarks.

## 6. Compatibility and kill switch

- New nodes only (no change to the five existing LTXE nodes); `LTXE_DISABLE=1` already removes every LTX-E node;
  `LTXE_DISABLE_HDR=1` removes only the HDR nodes.
- Version isolation: the planner and dispatch carry `model_family` = `ltx-2.5-sdr-to-hdr`; LTX-2.3 HDR (LogC3) is a
  different stack and is refused by name if mixed in.

## 6b. P2 slice 1 design: highlight masks and source-preservation composite (spec 4.5, 4.6, 4.9)

The IC-LoRA always sees the full frame (spec 4.6). Masks only decide, AFTER generation, which generated pixels reach
the final image.

- **Alignment.** Masks are computed on the SDR source resized with the same settings as the guide (scale to a
  multiple of 32, lanczos) and cut with the same plan and window index, so they line up pixel for pixel with the
  generated window. Every per-window node checks the frame count against the plan.
- **Highlight Analyzer (4.5).** `clip` = a channel at or above the hard-clip code (default 0.998). The channel policy
  is any / all / luma. A naturally bright pixel below the ceiling is never in the clip mask. `near_white` is a
  separate soft ramp on luma (default 0.92 up to the clip level). Expansion is a max-filter in pixels, feather is a
  box blur, and temporal smoothing is a centred mean over n frames. The user mask can add, subtract or replace. The
  report gives per-frame clipped and near-white fractions and the frame-to-frame mask change.
- **Synth Mask (4.6).** Built from the clip mask plus, optionally, the near-white ramp. It has signed grow
  (expand / contract), feather, temporal smoothing and user add / subtract. Output: SYNTH_MASK, a red-overlay preview
  and mask QC (coverage and flicker).
- **Source-Preservation Composite (4.9).** Works in linear ACEScg.
  - The source in working space is the decode of the guide window (the frames the model saw: core sRGB -> ACEScct,
    then this pack's exact ACEScct -> ACEScg). The generated image is the decode of the VAE output.
  - Optional colour match: a per-channel gain, estimated once per window as the median of generated / source over
    trusted pixels. Trusted means mask < 0.05, not crushed, not near-white. The gain is applied to the GENERATED
    image, so the approved source is never moved.
  - `final = src * (1 - m) + gen * m` with `m = synth_mix * mask`, combined with the optional shadow / custom masks.
    The result is re-encoded to the input encoding so LTXE HDR Delivery still trims and converts it.
  - Report: in the preserved region (m == 0), the maximum difference to the source in the working space (0 = exact
    there) and after the encode round trip (float error). Also the mean difference in the generated region, the
    gains, and coverage. Bit-exactness is claimed only in the working space, never across encodings.
- **Not in this slice:** 8-bit cleanup (4.3), shadow recovery (4.4), seam keyframes (4.8: no official ComfyUI node
  found at 220e1d4), seam exposure match (4.10), QC (4.12).

## 7. Status

| Piece | Status |
|---|---|
| Upstream facts (section 1) | verified by reading source + workflows (no GPU run) |
| Source audit, input colour, chunk planner | P1, this slice |
| Delivery (ACEScg EXR, source frame numbers, sidecar) + `LTXE_SDR_to_HDR_Pro.json` | P1 (L9.131): CPU tests, executor run, workflow load-validated |
| Highlight Analyzer, Synth Mask, Source-Preservation Composite (section 6b) | P2 slice 1 (L9.132): CPU tests, executor run, wired into the Pro workflow |
| Model run on real weights | NOT RUN (needs LTX-2.5 22B on a GPU) |
