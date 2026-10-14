# 001: LTX schedule saturates above 2K and is NaN at DCP 4K

**Component:** ComfyUI core, `LTXVScheduler` and `ModelSamplingLTXV` (`comfy_extras/nodes_lt.py`).

**Seen on:** core 0.39.0 and core 0.40.0 (`e0a32bce84`, upstream HEAD on 2026-10-14). CPU, no model weights needed.

**Status:** reproduced. Not reported upstream yet.

## Symptom

Both nodes derive their shift from the latent's token count, frames × rows × columns, and the shift grows linearly with it. At 97 frames:

| Latent size | Shift | LTXVScheduler, 20 steps | ModelSamplingLTXV + `simple`, 20 steps |
|---|---|---|---|
| 1024 x 576 (a typical tile) | 3.3 | normal (1.0 ... 0.45, 0.1) | normal (1.0 ... 0.58) |
| 1920 x 1088 (DCP 2K) | 10.1 | normal (1.0 ... 0.57, 0.1) | **all 20 steps >= 0.999** (last 0.9992 -> 0) |
| 2560 x 1440 | 17.3 | **staircase**: 1.0 x 14, 0.82 x 4, 0.64, 0.1 | all 20 steps >= 0.999 |
| 3840 x 2176 | 38.6 | **flat 0.1** for all 20 steps | all 20 steps = 1.0 |
| 4096 x 2176 (DCP 4K) | 41.1 | **all 20 sigmas NaN** | all 20 steps = 1.0 |

A sampler fed a saturated schedule does no denoising until the final step (σ ≈ 1 → 0). The staircase repeats steps at one sigma, and NaN sigmas turn the latent to NaN. ModelSamplingLTXV matters when the sigmas come from a generic scheduler (BasicScheduler `simple` / `normal`). LTXVScheduler builds its own sigmas.

## Reproduce

```
python -I probe.py <path to ComfyUI core> result.json
```

The probe imports core's own nodes and builds empty latents. It uses a minimal stand-in model object, because ModelSamplingLTXV only clones the model and attaches an object patch. It prints and saves every sigma. The results from the two core versions are in this folder.

## Cause

From reading the code; not verified with a debugger.

- `shift = N·(max_shift − base_shift)/3072 + base_shift − 1024·(…)/3072`, with no cap.
- From about shift 10 up, `e^shift / (e^shift + 1/t − 1)` is within float32 rounding of 1.0 for most steps.
- LTXVScheduler then stretches the non-zero sigmas: `σ' = 1 − (1 − σ)·(1 − terminal)/(1 − σ_last)`.
  - When the values `1 − σ` are a few float32 ulps, they take only a few distinct values, so the schedule becomes a staircase (2560 × 1440).
  - When they all equal the same ulp, every `σ'` is exactly `terminal`: a flat 0.1 (3840 × 2176).
  - When `1 − σ_last` is exactly 0, the stretch is 0/0 and every sigma is NaN (4096 × 2176).

## Consequence for this pack

- The resolution planner computes and reports the schedule for the token count the model actually sees: the **tile**.
- Before sampling, it refuses a NaN or saturated schedule with a plain-English message.
- That planner check is ledger L7.103. Its regression test will use the sizes in this table.
