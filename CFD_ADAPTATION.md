# CFD adaptation of PPBC

This fork tracks the automotive CFD adaptation originally developed by Leiyao and the subsequent reproduction fixes maintained by Peiyu.

- Upstream: https://github.com/aletovvladimir/PPBC
- Fork: https://github.com/factnn/PPBC
- Development branch: `cfd-adaptation`
- Imported upstream commit: `288f4f2c01769b18ab81c0d23eeb527dac00426f`
- `main` retains upstream code; CFD work lives under `repro/` on the development branch.

## Code and provenance

`repro/CFD/` is the existing CFD reproduction workspace, imported without changing its numerical implementation. The primary entry point is `repro/CFD/experiments/ppbc_drivaernetpp_transformer_easy.py`. Its data module and experiment script include the existing downsampling correction and data-root normalization patches. The older `repro/CFD/ppbc_drivaernetpp_transformer.py` is retained for provenance; use the experiments entry point for current runs.

`repro/make_drivaerstar_npy.py` converts DrivAerStar data, and `repro/plot_cd_relative_error_comparison.py` plots experiment results. The patch scripts document earlier fixes and should not be reapplied without checking the code. Historical README results describe prior runs, not validation performed during this import.

The CFD loop is a separate adaptation, not a claim that all upstream selection strategies have been integrated. In particular, the current adaptation does not include the upstream trust/BANT client selection machinery.

## Runtime prerequisites

The current local CFD implementation requires two externally supplied compiled modules:

- `repro/CFD/model/transformer_impl.cpython-310-x86_64-linux-gnu.so`
- `repro/CFD/util/cfd_utils_impl.cpython-310-x86_64-linux-gnu.so`

These binaries remain available in the original local workspace but are not committed. A fresh clone is therefore **not yet a self-contained CFD training environment**. Obtain compatible modules from the project maintainers or restore and validate their source before running. The recovered Python CFD utility source is retained separately in the project archive and has not been silently substituted for the compiled implementation.

Python dependencies include PyTorch, Lightning, Hydra/OmegaConf, NumPy, PyVista, tqdm, and Matplotlib for plotting. The root `requirements.txt` belongs to upstream image-classification experiments, not a validated CFD environment lockfile. Existing commands and environment details are in `repro/README.md`.

Dataset paths and statistics must be supplied locally. Data, splits, results, checkpoints, PDFs, and compiled extensions are excluded from CFD source tracking. Full-point Cd evaluation uses `--test-downsample-size -1 --eval-batch-size 1`; verify normalization and physical conventions before comparing results.

## Development

```bash
git switch cfd-adaptation
git fetch upstream
# When intentionally updating upstream:
git merge upstream/main
git push origin cfd-adaptation
```

In Peiyu's workspace, `FederatedLearning/repro` is a compatibility symlink to `FederatedLearning/PPBC/repro`. Edit either path; both refer to the same files. Frozen historical archives remain outside this repository.

The upstream README, citation, and MIT license are retained. The linked paper is *A Robust Training Method for Federated Learning with Partial Participation*: https://openreview.net/forum?id=QP0mGwfv0G (also identified by upstream `CITATION.cff`). Local paper downloads are kept in the parent project's `papers/` directory.
