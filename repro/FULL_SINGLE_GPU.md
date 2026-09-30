# 完整数据集单卡训练对照

启动入口：`bash repro/run_full_single_gpu.sh`（从 PPBC 仓库根）。先确认设备空闲。

| 配置 | PPBC | 全参与 FedAvg 参考组 |
| --- | --- | --- |
| 数据 | DrivAerNet++ easy 全部 5703 train / 1154 test | 相同 |
| 客户端 | 6 个纯车身类别客户端，equal_classes，等客户端权重 | 相同 |
| 种子 | 42 | 42 |
| 外层周期 / 轮数 | 30 | 60 |
| 每周期内部迭代 | 2 | 1 |
| 每客户端本地 epoch | 1 | 1 |
| 训练点数 / batch | 5000 / 1 | 相同 |
| 测试点数 / batch | 全量 / 1 | 相同 |
| 精度 | fp32 | fp32 |
| 全测试集评估间隔 | 5 周期 | 10 轮 |
| 预计本地优化步数 | 342180 | 342180 |
| 预计模型更新上传次数 | 120 | 360 |
| 预计校正量上传次数 | 180 | 0 |

预算数字按全部客户端非空、无掉线、无 batch cap 推导；实际计数逐轮记录。PPBC 为 epoch_k=6、iter_k=2、q_m=1。上述上传次数是算法模拟量，并非真实网络传输。`upload_bytes` 只计上传浮点更新/校正张量的载荷，不含下发、协议头和其他网络开销。

这是**等本地优化步数**的全参与 FedAvg 对照，两者每轮采样与总上传量不同，不能声称是等通信预算的偏差校正消融。这里只有一个种子和一种异构划分，仍不足以形成最终性能结论。固定轮数结束，不按测试集指标挑选最佳模型。

## 运行与监控

```bash
GPU=5 PYTHON=/mnt/cfd/yichen/miniconda3/envs/zpy_cfd/bin/python \
  bash repro/run_full_single_gpu.sh
```

脚本按 PPBC → FedAvg 顺序运行，只暴露一张 GPU；结果根为 `repro/results/full_single_gpu_<时间>/`。可用 `STUDY_DIR` 指定新目录。目录中 `status.txt` 记录任务状态，`completed.txt` 记录已完成组，`*_command.sh` 保存完整命令。`study.lock` 防止相同目录被同时运行。

每组目录含 `latest_<method>_alpha_0.1.pt`、同名 `.csv`（已评估指标）、同名 `.json`（最近完整周期的进度）、`*_budget.csv`（逐轮预算），结束后另写 history/summary/runtime。模型 checkpoint 使用临时文件后原子替换；指标导出意外中断时以 checkpoint 为准。保留最后一个完整周期，没有保存周期内部进度；中断时当前未完成周期需要重做。

## 恢复

使用原数据、配置、Python/CUDA 环境和源代码版本，并先确认旧进程已结束：

```bash
RESUME=1 STUDY_DIR=/absolute/path/to/existing/study \
  GPU=5 PYTHON=/mnt/cfd/yichen/miniconda3/envs/zpy_cfd/bin/python \
  bash repro/run_full_single_gpu.sh
```

或使用原命令添加 `--resume /path/to/latest_ppbc_alpha_0.1.pt`。一次恢复一个方法和一个 alpha，配置/数据列表/主脚本摘要不符时拒绝恢复。checkpoint 包含全局模型、PPBC 的每客户端 final_errors、Python/NumPy/PyTorch/CUDA 随机状态、选择客户端的独立随机状态、历史指标与预算计数。客户端优化器本来就在每次本地训练时重建，因此周期边界无需恢复一个不存在的持久优化器。

只加载本项目可信来源的 checkpoint；其中包含 Python 随机状态等对象。未完成初始评估便中断时还没有 checkpoint，需给该组重新指定输出目录，或先保留并移开该组未完成目录再重启。

## 验证

`python -m unittest discover -s tests -v` 覆盖归一化和中断恢复。恢复测试注入随机客户端更新，在保存第 1 周期后模拟中断，核对 PPBC/FedAvg 的最终模型、PPBC 校正量、历史指标、随机轨迹及预算计数与不中断运行一致；也检查不兼容配置会被拒绝。
