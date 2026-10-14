# Licence register

Checked 2026-10-14. This page records why ComfyUI-LTX-Enhanced can be Apache-2.0, which outside material it may use, and the points that are still open. It is an engineering record, not legal advice. Every doubtful point below is marked for the owner.

## This pack

- **Licence:** Apache-2.0, full text in [LICENSE](../../LICENSE) (copied from https://www.apache.org/licenses/LICENSE-2.0.txt, sha256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`).
- **Contains:** original code, tests, benchmarks, workflows built from scratch, research notes written in our own words.
- **Never contains:**
  - code copied from Lightricks/ComfyUI-LTXVideo or Lightricks/LTX-2;
  - model weights or text encoders;
  - Lightricks example workflows (those are "Complementary Materials" under the LTX licences);
  - code from GPL, AGPL, no-licence, non-commercial or other restricted sources.

## Reuse rules by licence

Each repository's rule is in the "Reuse" column of [the ecosystem registry](../registry/ecosystem.md).

| Source licence | What this pack may take | Why |
|---|---|---|
| Apache-2.0, MIT, BSD | Code, with attribution: a `# PORTED FROM: <repo>@<revision> <path> (<licence>)` header and an entry in NOTICE.md | Permissive into permissive is allowed. Apache-2.0 section 4 needs the NOTICE and the change notice. |
| LTX-2 Community License (2026-01-05), LTX-2.x Community License (2026-08-11) | Behaviour notes only | Section 3: a Derivative "must be distributed exclusively under the terms of this Agreement". Section 1 counts "any modified or extended versions of the Complementary Materials" as a Derivative. |
| GPL-3.0, AGPL-3.0 | Behaviour notes only | Copyleft cannot flow into a permissive licence (standing order 9). |
| No licence file | Behaviour notes only | No licence means all rights reserved. |
| RES4LYF (AGPL-3.0 plus a no-commercial-service clause), WanGP Community License 2.0, PolyForm Noncommercial | Nothing, beyond the idea level | The terms restrict commercial use or resale. An Apache pack must not carry that code. |
| An MIT or Apache label on code ported from LTX-2 (dgrauet/ltx-2-mlx; Arturogpj/wan2gp-ltx-tiled-refine, whose docstring says it ports LTX-2's TiledDiffusionModel) | Behaviour notes only | A port of LTX-2 code is a Derivative: the LTX licence governs the ported parts, whatever the label says. |
| Apache or MIT repositories that run inside another product (LTX-Desktop) | Only files that are the author's own, checked file by file | The host product's own code keeps its own licence. |

## Integration points

| Point | How the pack uses it | Licence | Verdict |
|---|---|---|---|
| ComfyUI core: `comfy.*`, `comfy_api` V3 types, `comfy.ldm.lightricks` | Imported at run time from the user's own install. Code never copied. | GPL-3.0 | Allowed. Apache-2.0 code may be combined with GPL-3.0 (one-way compatible). The user installs both, and this pack distributes no core code. Other Apache-2.0 packs on core follow the same model: city96/ComfyUI-GGUF, kijai/ComfyUI-WanVideoWrapper, welltop-cn/ComfyUI-TeaCache, Zehong-Ma/ComfyUI-MagCache. See D-2. |
| ComfyUI front end | Our web files run beside it in the browser. | GPL-3.0 | Allowed on the same basis. |
| Lightricks/ComfyUI-LTXVideo, the user's separate install | **Graph level only.** The user wires MODEL, LATENT, CONDITIONING, VAE and IMAGE values between upstream nodes and ours. Our Python never imports upstream modules, never instantiates upstream node classes and never reads upstream files. | LTX-2 Community License | Allowed with this boundary. A value passed through the graph is not a modified or extended version of upstream code. See D-1. |
| LTX model weights (LTX-2, 2.3, 2.5, IC-LoRAs) | Loaded by core or upstream loaders that the user runs. This pack never downloads or bundles weights. | LTX-2 / LTX-2.x Community License; LTX-Video 0.9.x open-weights licences (see [models](../registry/models.md)) | The user's own obligation. The README states the key terms: Attachment A use restrictions, and a paid commercial licence for entities with annual revenue of at least $10M. |
| torch 2.9.0 | Run-time dependency, already in ComfyUI | BSD-3-Clause (package metadata) | Allowed |
| numpy 2.4.6 | Run-time dependency, already in ComfyUI | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 (package metadata) | Allowed |
| pytest 9.1.1 | Tests only | MIT | Allowed |
| SageAttention (optional, later) | Optional run-time backend if installed. Never bundled. | Apache-2.0 | Allowed |

The dependency licences were read from the package metadata in the owner-stack Python (`importlib.metadata`) on 2026-10-14.

## Doubtful points (NEEDS_OWNER for a legal read; engineering mitigations are in place)

**D-1. "Works based on LTX-2."**

- What the licence says: it defines "LTX-2" to include the models' "software and algorithms". It counts "works based on LTX-2" as Derivatives.
- Why it matters: a tiling planner that knows LTX latent geometry could be argued to be "based on" LTX-2.
- Mitigations:
  - The planning core is model-agnostic. Geometry is a parameter set, and the LTX presets are plain numbers published in the LTX-2 paper (arXiv 2601.03233), such as the spatial and temporal compression.
  - The tiling and blending methods come from published papers (MultiDiffusion, Mixture of Diffusers, SpotDiffusion and the others in [papers](../papers/README.md)), not from Lightricks code.
  - Integration stays at graph level.
- Status: the owner chose this model (decision D0.20, A13 Q1, "subject to verifying ..."). A legal read is the owner's call.

**D-2. Plugins inside a GPL host.**

- The FSF's view is that plugins sharing an address space with a GPL program form one combined work.
- This affects whoever distributes the combination. It does not change the licence of our own source. Apache-2.0 is compatible with GPL-3.0 in that direction.
- This is ordinary ComfyUI ecosystem practice, and the owner's other packs follow the same model.

**D-3. One-team clean room.**

- The same engineer reads LTX-licensed code for behaviour and then writes ours.
- Mitigations:
  - Notes are prose, maths and pseudocode, with no code text. See [the clean-room procedure](clean_room.md).
  - Every note names the revision it describes.
  - A provenance check runs before each release. It compares token n-grams against the inspected upstream revisions, and anything above the threshold blocks the release. The check is ledger work, L7.104 / L7.109.

## Change log

- 2026-10-14: first version (ledger L7.105).
