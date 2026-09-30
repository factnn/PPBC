#!/usr/bin/env python3
"""让 FL 脚本的归一化统计量以 `<data_root>/mean_std.json` 为准（训练与指标共用一套）。

为什么需要（见 ../leiyao_FL_review.md §9）：
    仓库自带的配置里，**训练/反归一化**用的是 `config/datamodule/*.yaml` 里硬编码的
    `0/500、0/1.5、0/4`，而**指标** `mse_p`/`mse_wss` 用的是 `metric.mean_std_file`
    指向的数据集文件（DrivAerNet++ 是 `-109.07/140.53` + 逐分量 wss）——两套不是一回事，
    于是 `mse_p` 变成了"物理 MSE ÷ 140.53²"这种混合口径。

    换成 DrivAerStar 后统计量差别更大（pressure std ≈ 285、wss ≈ 2.2），不统一就会
    训练用一套、评分用另一套。本补丁让脚本优先读 `<data_root>/mean_std.json`，
    训练与指标从此共用同一套；文件不存在时退回原来的硬编码值。

幂等：已打过就跳过。
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parent
    / "CFD"
    / "experiments"
    / "ppbc_drivaernetpp_transformer_easy.py"
)

OLD = """def main() -> None:
    args = parse_args()
    if not set(args.methods).issubset({"fedavg", "ppbc"}):"""

NEW = """def main() -> None:
    args = parse_args()
    # [patch] 归一化统计量以数据根目录的 mean_std.json 为准（训练与指标共用一套，§9）
    _mean_std_path = Path(args.data_root) / "mean_std.json"
    if _mean_std_path.exists():
        global MEAN_STD_DICT
        MEAN_STD_DICT = json.loads(_mean_std_path.read_text(encoding="utf-8"))
        print(f"[patch] 已从 {_mean_std_path} 载入归一化统计量", flush=True)
    if not set(args.methods).issubset({"fedavg", "ppbc"}):"""


def main() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    if "_mean_std_path" in text:
        print(f"[skip] 已打过补丁：{SCRIPT}")
        return
    if OLD not in text:
        print("[FAIL] 找不到锚点")
        sys.exit(1)
    SCRIPT.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"[ok] 已打补丁：{SCRIPT}")


if __name__ == "__main__":
    main()
