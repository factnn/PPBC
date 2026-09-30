#!/usr/bin/env python3
"""给复现副本打「降采样会让 Cd 等比缩小」的防护补丁。

背景（见 ../leiyao_FL_review.md §7）：
    `calc_force_coeff` 在**当前这批点**上求和 area·p·n·flow_dir，分母却用完整参考面积，
    所以只要 `--downsample-size` / `--test-downsample-size` 生效，c_d 就按采样比等比缩小
    （实测 5 000 / 474 686 点 → 缩小约 95 倍）。leiyao 的 pilot 表 `c_d_rmse ≈ 2.5e-4`
    就是这么来的。

本补丁做两件事：
    1) `PointCloudDataset.__getitem__` 把 `num_points_total`（降采样前的真实点数）放进 `others`；
    2) FL 脚本在评估阶段，如果启用了测试集降采样，就按 `N_total / N_sampled` 把
       `c_p / c_wss / c_d` 回缩放，并打印一次警告（回缩放仍有 6–21% 采样噪声，
       所以**推荐直接 `--test-downsample-size -1` 用全量点**）。

补丁是幂等的：已经打过就跳过。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "CFD"
DATASET = ROOT / "datamodule" / "point_cloud_datamodule.py"
SCRIPT = ROOT / "experiments" / "ppbc_drivaernetpp_transformer_easy.py"

DATASET_OLD = """        npy_data = np.load(data_file)
        if self.downsample_size > 0 and self.downsample_size < num_points:"""
DATASET_NEW = """        npy_data = np.load(data_file)
        num_points_total = num_points          # [patch] 降采样前的真实点数
        if self.downsample_size > 0 and self.downsample_size < num_points:"""

DATASET_OLD2 = """        others = {
            "file_path": data_file,"""
DATASET_NEW2 = """        others = {
            "file_path": data_file,
            "num_points_total": torch.tensor(num_points_total),   # [patch] 见 §7"""

SCRIPT_OLD = """            calc_force_coeff_wrapper(inputs, outputs, targets, others)
            metric_dict = metric_wrapper(outputs, targets)"""
SCRIPT_NEW = """            calc_force_coeff_wrapper(inputs, outputs, targets, others)
            # [patch] 降采样会让 area 积分等比缩小 → 按 N_total/N_sampled 回缩放（§7）
            if args.test_downsample_size > 0 and "num_points_total" in others:
                n_used = int(targets["pressure"].shape[-2])
                ratio = float(others["num_points_total"].reshape(-1)[0]) / max(n_used, 1)
                if not getattr(evaluate_state, "_warned", False):
                    print(
                        f"[WARN] 测试集降采样生效（{n_used} / {int(others['num_points_total'].reshape(-1)[0])} 点，"
                        f"ratio={ratio:.2f}）：Cd 类指标已按 N/n 回缩放，仍含 6-21% 采样噪声；"
                        f"正式结论请用 --test-downsample-size -1。",
                        flush=True,
                    )
                    evaluate_state._warned = True
                for key in ("c_p", "c_wss", "c_d"):
                    if key in outputs:
                        outputs[key] = outputs[key] * ratio
                    if key in targets:
                        targets[key] = targets[key] * ratio
            metric_dict = metric_wrapper(outputs, targets)"""


def patch(path: Path, pairs: list[tuple[str, str]], marker: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        print(f"[skip] 已打过补丁：{path}")
        return False
    for old, new in pairs:
        if old not in text:
            print(f"[FAIL] 在 {path} 里找不到锚点：\n{old[:120]}...")
            sys.exit(1)
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    print(f"[ok] 已打补丁：{path}")
    return True


def main() -> None:
    patch(
        DATASET,
        [(DATASET_OLD, DATASET_NEW), (DATASET_OLD2, DATASET_NEW2)],
        marker="num_points_total",
    )
    patch(SCRIPT, [(SCRIPT_OLD, SCRIPT_NEW)], marker="_warned")


if __name__ == "__main__":
    main()
