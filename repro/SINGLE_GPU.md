# PPBC CFD 单卡运行

## 可行性与当前状态

一张 GPU 可以串行模拟多个客户端：`train_client` 每次创建并训练一个客户端模型，返回 CPU 上的模型差分；聚合状态及每个客户端的校正状态放在 CPU。无需为三个或六个客户端分别准备 GPU。真实多机器网络通信并未实现，此处是算法仿真。

2026-09-30 实测当前 `fornax` 的 `nvidia-smi` 无法连接驱动；既有 Python 3.10 环境中 PyTorch 为 `2.8.0.dev20250414+cu128`，`torch.cuda.is_available()` 为 false，设备数为 0。因此本次尚未完成 GPU 实跑，不能给出实测显存或速度。

## 启动

在 `PPBC/` 下执行。`PYTHON` 必须指向具备 CFD 依赖及两个本地编译模块兼容性的环境，详见 `../CFD_ADAPTATION.md`。本机已有环境可用以下路径；另一台机器须自行指定实际环境及数据根。

```bash
# 只预览命令，不导入模型或启动训练。
DRY_RUN=1 GPU=0 PYTHON=/mnt/cfd/yichen/miniconda3/envs/zpy_cfd/bin/python \
  bash repro/run_single_gpu.sh smoke

# 在已有 GPU 分配的节点执行：使用调度器设置的 CUDA_VISIBLE_DEVICES。
PYTHON=/mnt/cfd/yichen/miniconda3/envs/zpy_cfd/bin/python \
  bash repro/run_single_gpu.sh smoke

# smoke 通过后，扩大为 200 训练例、12 测试例、6 个外层周期。
PYTHON=/mnt/cfd/yichen/miniconda3/envs/zpy_cfd/bin/python \
  bash repro/run_single_gpu.sh pilot
```

可通过 `DATA_ROOT=/path/to/data` 更换数据；默认本地 DrivAerNet++ 的 `repro/data`。在独立服务器上可用 `GPU=0` 指定卡；在调度器内优先保留其设备分配。`PRECISION=bf16` 可在确认支持后使用，默认先用 fp32 排除混合精度问题。脚本只暴露一张卡，不会自动占用多卡或退回 CPU，已有输出目录也不会被覆盖。

smoke 使用 3 个纯车身类别客户端、18 个训练例、3 个测试例、每客户端最多 1 个 batch、1000 个训练点、2 个外层周期 × 2 次内部迭代。两周期用于实际经过上一周期误差反馈的路径。每内部迭代选 1 个客户端上传，3 个客户端都参与计算。评估始终使用全量点，batch size 为 1。

结果写到独立的 `repro/results/single_gpu_*`，含 `config.json`、`run.log`、`history.csv`、`summary.csv` 及客户端划分。检查损失和指标有限、客户端非空、物理量及归一化正确，并在目标卡上记录峰值显存和时间。全点评估可能比小样本训练更占显存，不能仅根据训练点数判断容量。

## 论文与代码核对

依据本地 21 页论文 *A Robust Training Method for Federated Learning with Partial Participation*，重点核对第 5 页 Algorithm 1、第 7 页设备掉线扩展及第 8 页实验协议。论文入口：https://openreview.net/forum?id=QP0mGwfv0G 。

| 项目 | 论文 / 当前实现 | 运行含义 |
| --- | --- | --- |
| 偏差校正 | 未上传的更新积累为代理量，后续周期使用 | 不能简单跳过所有未选中的客户端计算 |
| 部分参与 | `epoch_k` / `iter_k` 控制上传者，`q_m` 控制可用性掩码 | `q_m=1` 不代表全量上传；本脚本 `iter_k=1<3` |
| 更新量 | 论文使用随机梯度；CFD 使用本地优化后的模型差分，并重建优化器 | 是工程适配，不能直接宣称继承论文收敛保证 |
| 周期长度 | 论文理论用几何随机长度；现有实现固定 `ppbc_iterations` | 启动参数沿用已有实践实现 |
| 历史误差系数 | 上游代码及 CFD 均有 `0.5` 缩放，论文 Algorithm 1 未直接写成此形式 | 保留当前实现，正式复现前进一步验证 |
| 客户端权重 | 论文校正以 `1/M` 为参考；CFD sample 模式改用样本权重 | 先固定 `aggregation=equal`，避免混淆不同目标 |
| 掉线扩展 | 论文 Algorithm 2 每内部迭代生成掩码；CFD 每外层周期生成一次，仍执行所有客户端计算 | 先 `q_m=1`，暂不声称复现掉线实验 |
| 采样策略 | 当前 CFD 只实现 random；论文还评估其他规则 | 先跑通现有适配，再单独扩展策略 |

## 正式比较前的缺口

当前 FedAvg 每外层轮训练所有客户端一次；PPBC 每外层轮会训练所有客户端 `ppbc_iterations` 次。因此相同 `rounds` 的两条曲线并非等计算预算。论文实验要求相同采样规则，而当前 FedAvg 是全客户端聚合。正式结论需要补齐匹配的部分参与基线，并分别统计本地优化步数、上传更新次数（包含周期结束的代理量上传）与时间。

主实验脚本目前不保存模型 checkpoint、不支持 resume，CSV 也主要在整个流程结束后写出。先做短 smoke/pilot；长时正式实验前应补齐 checkpoint、恢复与增量记录。现有脚本不能用来生成可恢复的百轮正式训练任务。
