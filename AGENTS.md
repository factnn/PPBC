# PPBC CFD fork

- This is the development repository for the automotive CFD adaptation. Read `CFD_ADAPTATION.md` and `repro/README.md` before changing CFD code.
- Use branch `cfd-adaptation`; `origin` is `factnn/PPBC`, and `upstream` is `aletovvladimir/PPBC`. Preserve upstream history and attribution.
- Develop CFD changes under `repro/`. The parent workspace's `repro` path is a symlink to the same directory, not a second source copy.
- Keep datasets, experiment outputs, checkpoints, compiled extensions, credentials, and paper PDFs out of commits. Do not overwrite existing experiment outputs.
- The CFD runtime currently depends on local compiled modules described in `CFD_ADAPTATION.md`; do not claim that a fresh clone can train without them.
- Verify full-point Cd evaluation, normalization statistics, and physical conventions. Prior mini-pilot results are pipeline checks, not publication-grade performance evidence.
- Validate changes proportionately; do not launch full training for routine checks. Preserve existing code style and avoid unrelated edits to upstream algorithms.
