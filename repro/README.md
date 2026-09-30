# repro/ —— leiyao 那套联邦学习，我把它跑起来并补上缺失产物

> 单卡启动与论文核对见 [SINGLE_GPU.md](SINGLE_GPU.md)，入口为 `run_single_gpu.sh`。已完成单卡验证，见 [本次结果](SINGLE_GPU_RESULTS.md)。现已支持 checkpoint/resume 和逐轮预算记录；完整数据集 PPBC/FedAvg 对照使用 [FULL_SINGLE_GPU.md](FULL_SINGLE_GPU.md) 中的新入口。`q_m=1` 仍可通过客户端选择实现部分参与。

**为什么有这个目录**：leiyao 已离职，他做的 FL 适配只剩代码，pilot 的原始产物
（`history.csv` / `summary.csv` / `central_checkpoint_diagnosis.csv` / 绘图脚本）都不在机器上
（清查过程见 `../leiyao_FL_review.md` §6.2）。本目录把那套流程**原样跑通**，并补出等价产物。

---

## 1. 目录内容

| 路径 | 说明 |
| --- | --- |
| `CFD/` | leiyao 的 CFD 仓库 + FL 脚本的可运行副本（来自 `leiyao_archive`，**不含** datasets/logs） |
| `CFD/experiments/ppbc_drivaernetpp_transformer_easy.py` | 主脚本（FedAvg / PPBC，车身类型切分客户端） |
| `data/` | 数据根：软链到 leiyao 的 `split/{train,test}_split.json` 与 `mean_std.json`；点云直接走 split 里的绝对路径 |
| `run_minipilot.sh` | mini-pilot 启动脚本（CPU 小规模，评估用全量点） |
| `results/` | 产物：`history.csv` / `summary.csv` / `distribution_summary.csv` / `config.json` / `run.log` |
| `plot_cd_relative_error_comparison.py` | **重写** leiyao 被删的绘图脚本（接口按 `.pyc` 字节码复原） |
| `patch_downsample_guard.py` | 给"降采样缩小 Cd"这个坑打的防护补丁（见 §3） |

## 2. 怎么跑

```bash
cd /mnt/cfd/peiyu/CFD/FederatedLearning/repro/CFD
export MPLCONFIGDIR=/tmp/mpl-cfd OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/mnt/cfd/yichen/miniconda3/envs/zpy_cfd/bin/python

# mini-pilot（本机 CPU，约 20 分钟/档）
$PY experiments/ppbc_drivaernetpp_transformer_easy.py \
  --data-root ../data --split easy --methods fedavg ppbc --alphas 0.1 \
  --rounds 6 --local-epochs 1 --downsample-size 3000 --test-downsample-size -1 \
  --max-train-samples 200 --max-test-samples 4 --batch-size 1 --eval-batch-size 1 \
  --num-workers 2 --device cpu --precision fp32 --eval-every 0 \
  --ppbc-iterations 2 --epoch-k 3 --iter-k 1 \
  --num-clients 3 --client-split-mode single_class --single-class-allocation proportional \
  --output-dir ../results/minipilot_single_class
```

**有 GPU 时**（`--device cuda`，参数按 README 里的 "full easy-split run template" 原样）：

```bash
CUDA_VISIBLE_DEVICES=0 $PY experiments/ppbc_drivaernetpp_transformer_easy.py \
  --data-root ../data --split easy --methods fedavg ppbc \
  --alphas 0.1 0.5 10 --rounds 100 --local-epochs 1 \
  --num-clients 6 --client-split-mode single_class --single-class-allocation proportional \
  --epoch-k 6 --iter-k 2 --ppbc-iterations 5 \
  --downsample-size 50000 --test-downsample-size -1 \
  --batch-size 8 --eval-batch-size 1 --num-workers 16 \
  --lr 1e-4 --weight-decay 0.01 --precision bf16 --device cuda \
  --output-dir ../results/ppbc_easy_gpu
```

> ⚠️ 仓库自带的 torch 不支持 RTX 6000D（sm_120）；用共享 env（torch 2.8 / cu128）或换卡。
> ⚠️ `--eval-batch-size` 必须为 1（脚本自己会检查），`--test-downsample-size -1` = 评估用全量点。

出图：

```bash
$PY ../plot_cd_relative_error_comparison.py \
  --ppbc-history ../results/minipilot_single_class/history.csv \
  --central-value 0.0280 \
  --output-dir ../results/analysis_plots
```

（`--central-value` 是本机 `logs_locked/drivaernetpp/easy_train/train.log` 里集中式基线的最优 `rae_c_d`：
epoch 59 = 2.80%；该 csv 原件不在，故直接给数值。`--cyclic-summary` 不存在时会自动跳过。）

## 3. ⚠️ 最关键的坑：降采样会让 Cd 等比缩小

`calc_force_coeff` = `Σ_cells area·p·n·flow_dir / (0.5·ρ·U²·A_ref)`，**只在当前这批点上求和**。
`PointCloudDataset` 按 `downsample_size` 随机抽点后，被抽掉的单元的 `area` 不再进入求和，于是：

> `c_d(降采样) ≈ c_d(真值) × N_sampled / N_total`

实测（案例 `E_S_WWC_WM_008`，N=474686，官方 `c_d=0.229442`）：

| downsample | 采样比 | c_d | ×(N/n) 修正 |
| --- | --- | --- | --- |
| 1 000 | 0.21 % | 0.000586 | 0.278 |
| 5 000 | 1.05 % | 0.002264 | 0.215 |
| 50 000 | 10.5 % | 0.021550 | 0.205 |
| **全量** | 100 % | **0.208694** | （与 npz 官方 `c_p` 逐位一致） |

- **leiyao 的 pilot 表 `c_d_rmse ≈ 2.5e-4` 就是这么来的**：不是"0.09% 的 Cd 误差"，
  而是被缩小 100–1000 倍的 Cd 上的 RMSE，**不能与集中式的 `rae_c_d ≈ 3%` 直接比**。
- 自检信号：评估若用全量点，`c_d` 应在 **0.2–0.3**；出现 `1e-4`/`1e-3` 量级就是踩了这个坑。
- 训练本身不受影响（loss 在归一化场上算），受影响的只有 Cd 这类面积积分量。
- `patch_downsample_guard.py` 给复现副本加了 `num_points_total` 透传 + 自动回缩放 + 警告；
  但**正式结论请直接用 `--test-downsample-size -1`**（回缩放仍残留 6–21% 采样噪声）。

## 4. 与 leiyao 原版的差异（改动都在我这边，原版未动）

1. **没有改上游 PPBC 仓库**（`thirdparty/PPBC` 的 `git status` 是干净的）——leiyao 是另写脚本调用它，我保持同样做法。
2. 我这份副本**只加了降采样防护补丁**（`patch_downsample_guard.py`），训练逻辑与 leiyao 完全一致。
3. 原脚本的 PPBC 只实现了**随机部分参与**那一档（`select_clients` = `rng.choice`），
   没有上游的 trust 集 / BANT / loss / gradient_norm 选择，也没有 SCAFFOLD —— 我未做改动。
4. `plot_cd_relative_error_comparison.py` 是我**重写**的（原文件只剩 `.pyc`），
   参数名/默认文件名/配色/纵轴文字按字节码复原，另加了 `--central-value` 以便在没有
   `central_checkpoint_diagnosis.csv` 时也能出图。

## 5. CPU 成本标定（8 线程，用于估算放大到 GPU 的规模）

| 点数 | 训练一步 | 纯推理 |
| --- | --- | --- |
| 5 000 | 0.27 s | 0.12 s |
| 50 000 | 2.62 s | 1.52 s |
| 474 686（全量） | 30.4 s | 18.4 s |

## 6. 我跑出来的 mini-pilot 结果（2026-09-30，CPU）

设置：easy split，训练 200 例（分层抽样）、测试 4 例，`--downsample-size 3000`（训练），
**`--test-downsample-size -1`（评估用全量点 → Cd 口径正确）**，6 轮通信 × 1 local epoch，
FedAvg 与 PPBC 各跑一遍。两个切分模式各一次。

| 切分模式 | 客户端构成 | 方法 | `c_d_rmse` | `rae_c_d` | `mse_p` |
| --- | --- | --- | --- | --- | --- |
| 起始（未训练） | — | — | 0.2523 | ~101 % | 1.91 |
| `single_class` | 3 个纯类 client：E 34 / F 128 / N 38（熵 = 0） | FedAvg | 0.1178 | **39.4 %** | 0.576 |
| `single_class` | 同上 | PPBC | 0.1267 | **40.6 %** | 0.588 |
| `dirichlet` α=0.1 | 3 个 client，最小 2 / 最大 128（熵 = 0.209） | FedAvg | 0.1735 | **54.2 %** | 0.593 |
| `dirichlet` α=0.1 | 同上 | PPBC | 0.1010 | **34.5 %** | 0.557 |
| **集中式基线**（参照） | 5703 例、100 epoch | — | — | **2.80 %**（epoch 59 最优） | 0.037 |

**怎么读这张表**：

1. **这是管线验证，不是性能结论** —— 训练量只有集中式基线的 ~3.5 %（200/5703 例）和 6/100 epoch，
   所以 30–50 % 的 Cd 误差是正常的；要拿性能数字必须在 GPU 上按 README 模板跑满。
2. **PPBC 在极端异构（dirichlet α=0.1）下明显优于 FedAvg（34.5 % vs 54.2 %）**，
   在纯类切分下两者持平（40.6 % vs 39.4 %）——方向与上游论文一致，但**只有 4 个测试案例，噪声极大**，
   不能当结论用，只能当"值得在 GPU 上放大验证的信号"。
3. 所有 `c_d` 都在 **0.1–0.25** 的正确量级；如果看到 1e-4/1e-3，就是踩了 §3 的降采样坑。
4. `patch_downsample_guard.py` 已验证生效：故意用 `--test-downsample-size 1000`（ratio=478.6）时，
   脚本打印警告并把 `c_d` 从 5e-4 量级救回 **0.245**。

**产物**：
`results/minipilot_single_class/`、`results/minipilot_dirichlet/`（各含 `history.csv` / `summary.csv` /
`distribution_summary.csv` / `client_class_counts_alpha_0.1.csv` / `config.json` / `run.log`）、
`results/analysis_plots/cd_relative_error_{single_class,dirichlet}.{pdf,svg}`。

## 7. DrivAerStar 适配器（`make_drivaerstar_npy.py`）—— 用你们自己的数据集跑

leiyao 的脚本是照 DrivAerNet++ 的 `.npy/.npz` 接口硬写的。适配器把 **DrivAerStar 的 VTK 转成逐字段同构的
`.npy/.npz`**，于是**训练/联邦循环一行都不用改**就能跑自家数据。

```bash
PY=/mnt/cfd/yichen/miniconda3/envs/zpy_cfd/bin/python
# 小规模试跑（每 style 5 例，约 10 秒）
$PY make_drivaerstar_npy.py --out data_drivaerstar --limit 5
# 全量（14178 个 VTK，实测 ~0.55 s/例 ⇒ 单进程约 2.2 小时）
$PY make_drivaerstar_npy.py --out data_drivaerstar
# 跑联邦循环（数据根换成转换结果即可，其余参数不变）
cd CFD && $PY experiments/ppbc_drivaernetpp_transformer_easy.py \
  --data-root ../data_drivaerstar --split easy --methods fedavg ppbc \
  --num-clients 3 --client-split-mode single_class --rounds 1 \
  --downsample-size 3000 --test-downsample-size -1 --device cpu --eval-batch-size 1
```

产出：`data_drivaerstar/{npy/, split/{train,test}_split.json, split/easy/..., mean_std.json}`

**自校验**：转换器会把算出的 `c_d` 和官方 `cd_drag_results.zip` 对比 ——
15 例实测**平均相对误差 0.231%、最大 0.343%**，说明物理约定（`F = Σ[p·n̂+τ]A`、`ρ=1.25`、`U=40`、
`A_ref` 取 `frontal_area_<STYLE>.csv`）与官方一致。

## 8. ⚠️ 第二处口径问题：归一化统计量有两套（仓库自带，非 leiyao 引入）

| 用途 | 统计量来源 |
| --- | --- |
| 训练 / 反归一化（`datamodule._denormalize`） | `config/datamodule/*.yaml` 里**硬编码**的 `0/500、0/1.5、0/4` |
| 指标 `mse_p`/`mse_wss`（`CFDMetricWrapper`） | `metric.mean_std_file` 指向的数据集文件（DrivAerNet++：`-109.07/140.53` + 逐分量 wss） |

⇒ `mse_p` ≈ **物理 MSE ÷ 140.53²**，是"混合口径"的量：方法之间相对比较仍可用，但**绝对值不能跨口径引用**。
换到 DrivAerStar 更明显（pressure std ≈ **285**、wss ≈ (2.33, 0.98, 2.77)、centroid ≈ (1.39, 0.57, 0.33)）。

`patch_mean_std_from_dataroot.py` 让脚本优先读 `<data_root>/mean_std.json`，**训练与指标从此共用同一套**；
文件不存在时退回原硬编码值（对 DrivAerNet++ 的行为不变）。已实测：换 DrivAerStar 后脚本打印
`[patch] 已从 ../data_drivaerstar/mean_std.json 载入归一化统计量`，并正常跑完一轮（`c_d_rmse=0.254`，
正确物理量级）。

> 提示：`results/drivaerstar_check/` 就是"用自家数据集跑通"的证据（12 train / 3 test，1 轮）。
