# ComfyUI-LTX-Enhanced

High-resolution LTX video generation for ComfyUI, aimed at DCP 2K (1920×1088) and DCP 4K (4096×2176) frames on real hardware. The pack will provide:

- a hardware-aware resolution and memory planner;
- spatial and temporal tile planners;
- padding, alignment and crop-back;
- overlap blending and seam reduction.

Each method is measured against the existing ones before it is recommended.

> **Status: research phase. No nodes yet.** So far the repository holds the ecosystem inventory, the licence register, the reverse-engineering notes and the first crash reproduction. Nodes arrive in the order of the [roadmap](#roadmap). Each one is labelled *experimental* until a benchmark backs it ([stability labels](docs/stability.md)).

## What it works with

The pack runs inside ComfyUI and uses the LTX support that you install yourself: ComfyUI core's own LTX nodes, or [Lightricks/ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo). It connects to them only through the node graph, by passing models, latents, conditioning and images between nodes. It contains no Lightricks code. It never downloads or bundles model weights.

**Model licence.** The LTX-2, 2.3 and 2.5 weights are under the LTX-2 / LTX-2.x Community License Agreements, and those terms apply to you when you use them:

- use restrictions in Attachment A;
- entities with annual revenue of at least $10M need a paid commercial licence from Lightricks.

See the [model registry](docs/registry/models.md) for the licence of each model repository.

## Clean room, Apache-2.0

This pack is Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). It learns from many implementations whose licences do not allow their code into an Apache project: the LTX Community Licenses, GPL tiling packs and ComfyUI core. From those it takes behaviour notes only, written in our own words. The planners and blending are written from scratch, from published papers and those notes.

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
| `nodes/` | Production ComfyUI nodes |
| `experimental/` | Experimental algorithms and nodes, behind explicit options |
| `benchmarks/` | Benchmark scripts and raw results |
| `crash_repros/` | Minimal reproductions of every significant defect, with sanitized logs |
| `tests/` | Automated tests (`python -m pytest tests`) |
| `workflows/` | Example workflows, built from scratch |
| `docs/` | Registry, papers, reverse-engineering notes, licensing, stability labels |
| `release_notes/` | One file per release |

## Roadmap

1. Inventory: ecosystem, models, papers, known issues, licences. **Done.**
2. Reproduce the official and community workflows, including the known failures. (Needs GPU runs with LTX weights, done on the owner's machine.)
3. Reverse-engineer the most relevant implementations and compare their algorithms. **First pass done:** [notes](docs/re-notes/README.md).
4. Reproducible baselines on the available hardware.
5. Original resolution planning, tiling, padding and blending.
6. Benchmark existing techniques. Add the inference and quality optimisations that measure well.
7. Test combined methods for regressions and interactions.
8. Publish verified code, reproducible experiments and transparent results.

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
