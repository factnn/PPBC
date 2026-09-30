# CFD Transformer

## Environment

```bash
conda create -n cfd_transformer python=3.10 -y
conda activate cfd_transformer
pip install -r requirements.txt
```

`requirements.txt` uses the CUDA 12.8 PyTorch wheel index.

## Dataset

Default layout:

```text
dataset/DrivAerNetPP/
|-- npy/
|   |-- *.npy
|   `-- *.npz
|-- split/
|   |-- train_split.json
|   `-- test_split.json
`-- mean_std.json
```

- `train_split.json` and `test_split.json`: JSON arrays of `.npy` file paths.
- Each `xxx.npy` file must have a matching `xxx.npz` file in the same directory.
- `.npy` fields: `centroid`, `area`, `normal`, `pressure`, `wss`.
- `.npz` fields: `num_points`, `center`, `flow_dir`, `flow_speed`, `air_density`, `frontal_area`.
- `mean_std.json`: normalization statistics used during evaluation.

## Train

Run with defaults from `config/train.yaml`:

```bash
python train.py
```

Resume training:

```bash
python train.py resume=true log_dir=<log_dir>
```

Checkpoints are saved to `<log_dir>/ckpt/epoch_*.pth`.

## Test

`ckpt_file` is required. Test batch size must be 1.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python test.py ckpt_file=<ckpt_file>
```

Save per-point predictions:

```bash
python test.py ckpt_file=<ckpt_file> save_results=true
```

Summary results are written to `<log_dir>/results.csv`. Per-point outputs are written to `<log_dir>/results/`.

## PPBC / FedAvg

Entry point:

```bash
python ppbc_drivaernetpp_transformer.py
```

PPBC groups body types by filename prefixes `E`, `N`, and `F`. Outputs are written to `--output-dir`, including `run.log`, `config.json`, `history.csv`, `summary.csv`, and client distribution CSV files.
