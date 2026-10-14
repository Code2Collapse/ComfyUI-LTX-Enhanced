# Stability labels

Every node, algorithm and option in this pack carries exactly one label. The label sits in the node's description and in the node table of the README. A label only moves up when its evidence exists.

| Label | Meaning | Evidence it needs |
|---|---|---|
| **experimental** | Implemented and unit-tested on CPU. Not yet compared with anything. | Passing tests in `tests/` |
| **benchmarked** | Measured against a baseline on named hardware, with a fixed seed, prompt, model version, sampler, steps, resolution and frame count. | A raw result file and the script in `benchmarks/` |
| **validated** | Benchmarked, and the result holds: repeated trials, ablations, no regressions when combined with the other methods, and the claimed quality or memory win reproduced. | Benchmark results plus an interaction-test report |
| **production-ready** | Validated, and in a release: compatibility matrix filled in (oldest and newest supported core and front end), crash repros closed, plain-English errors. | Release notes entry + compatibility matrix |

Image-quality claims need A/B evidence on the real model (a GPU run). Until then a feature stays **experimental**, whatever its CPU tests say.
