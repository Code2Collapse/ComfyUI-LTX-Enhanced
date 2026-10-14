# Clean-room procedure

This pack learns from implementations whose licences do not allow their code into an Apache-2.0 project:

- Lightricks/ComfyUI-LTXVideo and Lightricks/LTX-2 (LTX Community Licenses);
- the GPL tiling packs;
- ComfyUI core.

We study how they behave. We never copy what they wrote. This page is the procedure that keeps those two things apart and lets anyone audit the result.

## Reading

- Read only at a recorded revision: the `revision` in [sources.json](../registry/sources.json).
- Keep reference checkouts in a temporary directory outside this repository. Never vendor them, never commit them, never import them from our code.
- Run nothing from a checkout. Static reading and REA analysis only, unless a reproduction needs the real program. In that case it runs in the isolated testbed, never in the owner's production ComfyUI.

## Writing notes (docs/re-notes/)

Each note:

- names the repository, the revision, and the files and functions it describes;
- describes behaviour as prose, maths, tensor shapes, numbers and pseudocode in our own structure;
- contains no code text from a restricted source: no pasted lines, and no renamed-variable paraphrase of a function;
- marks every statement as one of:
  - **observed**: read in the code at that revision, or measured;
  - **inferred**: follows from what was observed, but was not run;
  - **unknown**.
- compares the behaviour with the documentation and the papers, and says where they disagree.

## Writing code

- Write from the notes, the papers and our own design documents, not with an upstream file open beside the editor.
- Code from a permissive source (Apache-2.0, MIT, BSD) may be ported. It must carry a `# PORTED FROM: <repo>@<revision> <path> (<licence>)` header and an entry in NOTICE.md.
- Each module states, in its docstring, which papers or notes it implements.

## Checking

- Before a release, a provenance check compares token n-grams of our source against the inspected restricted revisions. Any match above the threshold blocks the release until it is rewritten or explained. (To be built: ledger L7.104 / L7.109.)
- [The licence register](register.md) is updated whenever a source, dependency or integration point is added.
