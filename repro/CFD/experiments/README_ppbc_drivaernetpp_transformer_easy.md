# PPBC on DrivAerNet++ Easy Split

This experiment follows the Section 3 heterogeneity setting from `FL for Zhu_v2.pdf`:
vehicle body types are treated as classes and client data are split with a Dirichlet
distribution over the three classes.

Class mapping:

- `E`: `estateback`
- `N`: `notchback`
- `F`: `fastback`

Implementation notes:

- Data root defaults to `/mnt/nfs_project_a/shared/datasets/DrivAerNetPP`.
- Split defaults to `split/easy/{train_split.json,test_split.json}`.
- The model is always instantiated from `config/model/transformer.yaml`.
- The local CFD loss is instantiated from `config/loss/cfd_loss_wrapper.yaml`.
- Optimizer defaults are loaded from `config/train.yaml` and can be overridden by
  `--lr` and `--weight-decay`.

Full easy-split run template:

```bash
CUDA_VISIBLE_DEVICES=5 python experiments/ppbc_drivaernetpp_transformer_easy.py \
  --methods fedavg ppbc \
  --alphas 0.1 \
  --rounds 100 \
  --local-epochs 1 \
  --num-clients 6 \
  --client-split-mode single_class \
  --single-class-allocation proportional \
  --epoch-k 6 \
  --iter-k 2 \
  --ppbc-iterations 5 \
  --downsample-size 50000 \
  --batch-size 8 \
  --eval-batch-size 1 \
  --num-workers 16 \
  --lr 1e-4 \
  --weight-decay 0.01 \
  --precision bf16 \
  --output-dir experiments/results/ppbc_drivaernetpp_transformer_easy
```

Outputs:

- `history.csv`: per-round metrics for all bodies and each body class.
- `summary.csv`: final-round `body=all` rows with the same aggregate metric names as
  the centralized baseline comparison (`c_*_mae`, `c_*_rmse`, `mse_*`, `rae_*`).
- `distribution_summary.csv`: client heterogeneity summary for each alpha.
- `client_class_counts_alpha_*.csv`: per-client class counts.
- `config.json`: resolved experiment settings and repository model/loss config paths.
- `run.log`: local copy of the PPBC/FedAvg runtime log.

For a strict one-body-class-per-client split, use `--client-split-mode single_class`.
With `--num-clients 6 --single-class-allocation proportional`, the easy train split
creates one E client, one N client, and four F clients, with per-client sizes close to
the global target while every client remains pure-class. In this mode `--alphas` is
retained as an experiment label/output grouping value; it no longer controls the
client class mix.

Pilot result summary:

| alpha | method | c_d_rmse | pressure_rmse | wss_rmse |
| ----- | ------ | -------: | ------------: | -------: |
| 0.1   | fedavg | 0.000253 |       156.970 |    0.708 |
| 0.1   | ppbc   | 0.000348 |       147.472 |    0.686 |
| 0.5   | fedavg | 0.000374 |       196.395 |    0.694 |
| 0.5   | ppbc   | 0.000308 |       139.977 |    0.692 |
| 10    | fedavg | 0.000347 |       159.029 |    0.683 |
| 10    | ppbc   | 0.000350 |       144.128 |    0.737 |

The pilot is intentionally small and should be treated as a pipeline validation rather
than a final benchmark. It shows PPBC improves pressure RMSE in all three alpha
settings and improves final `c_d_rmse` for alpha `0.5`; it does not consistently beat
FedAvg on final `c_d_rmse` in this short run.

## Centralized CFD body-subset baselines

Use `experiments/cfd_body_subset_baselines.py` to compare PPBC with original CFD
centralized training on controlled body-subset train splits. The test split is always
the same DrivAerNet++ easy test split used by PPBC.

Default body-subset train groups:

- `E`
- `F`
- `N`
- `E_F`
- `E_N`
- `F_N`
- `E_F_N`

Generate split files and runnable commands without starting training:

```bash
python experiments/cfd_body_subset_baselines.py
```

Run all centralized baselines sequentially on GPU 0:

```bash
python experiments/cfd_body_subset_baselines.py --execute
```

Run centralized baselines on multiple GPUs, with one body group per GPU at a time:

```bash
python experiments/cfd_body_subset_baselines.py \
  --execute \
  --cuda-visible-devices 0,1,2,3
```

Each worker sets a single `CUDA_VISIBLE_DEVICES=<gpu_id>` and still passes
`devices=[0]` to the CFD training script, so each experiment sees exactly one GPU.

The generated commands call the repository's original `train.py` and `test.py` with
CFD defaults: `num_epochs=100`, `batch_size=8`, `downsample_size=50000`,
`precision=bf16-mixed`, and default `CUDA_VISIBLE_DEVICES=0`.

Useful outputs:

- `manifest.csv`: train case counts and log paths for each body subset.
- `run_body_subset_baselines.sh`: explicit train/test commands.
- `comparison_summary.csv`: centralized baseline metrics, plus PPBC rows if
  `--ppbc-summary` points to an existing PPBC `summary.csv`.

## Cyclic body baseline

Use `experiments/cfd_body_cyclic_baseline.py` to test a non-federated sequential
training baseline: one repository Transformer checkpoint is trained repeatedly on
body-specific subsets in the order `E -> F -> N`. One cycle means training once on
each body subset, so `--cycles` should usually match the PPBC `--rounds`.

Full easy-split run template on GPU 0:

```bash
python experiments/cfd_body_cyclic_baseline.py \
  --cycles 10 \
  --order E F N \
  --local-epochs 100 \
  --batch-size 8 \
  --eval-batch-size 1 \
  --downsample-size 50000 \
  --test-downsample-size -1 \
  --num-workers 16 \
  --lr 1e-4 \
  --weight-decay 0.01 \
  --precision bf16 \
  --output-dir experiments/results/cfd_body_cyclic_baseline_easy
```

Useful options:

- `--order E F N`: training order for each cycle. The default is `E F N`.
- `--eval-after-stage`: additionally evaluates after each body stage, useful for
  checking forgetting/recovery across `E`, `F`, and `N`.

Outputs:

- `run.log`: runtime log.
- `config.json`: resolved cyclic baseline settings and repository config paths.
- `body_counts.csv`: train/test case counts for each body type.
- `history.csv`: per-cycle metrics for all bodies and each body class.
- `summary.csv`: final `cycle_end` aggregate row with PPBC-compatible metric columns.
