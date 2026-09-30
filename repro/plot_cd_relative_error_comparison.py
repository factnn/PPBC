#!/usr/bin/env python3
"""Plot Cd relative error for PPBC, full centralized, and cyclic training.

本脚本是对 leiyao 被删掉的 `plot_cd_relative_error_comparison.py` 的重写。
接口（参数名、默认文件名、配色、纵轴、输出名）是从原始 `.pyc` 字节码里恢复出来的：

    __pycache__/plot_cd_relative_error_comparison.cpython-310.pyc

原脚本要读三个文件：
    * PPBC history.csv                    （取 body=all 行里最优的 rae_c_d）
    * central_checkpoint_diagnosis.csv    （取 label 匹配的那一行）
    * cfd_body_cyclic_baseline_easy/summary.csv（取 body=all 行里最优的 rae_c_d）

那三个文件在本机都不存在（见 FederatedLearning/leiyao_FL_review.md §6.2），
所以本重写版额外支持：
    * `--central-value`：直接给集中式的 rae_c_d（数值），不需要 diagnosis csv；
    * `--ppbc-history` 指向任意 history.csv（例如 repro 跑出来的 mini-pilot）。

指标口径：rae_c_d = 阻力系数的平均相对绝对误差，画成百分数，越低越好。

⚠️ 重要：只有当 history.csv 是**在全量点上评估**得到的时候，rae_c_d 才与集中式可比。
   若评估时用了 `--test-downsample-size > 0`，`calc_force_coeff` 会因为
   "Σ area·p·n·flow_dir 只在采样点上求和" 而把 c_d 等比缩小
   （实测 5k/47.5万 点 → c_d 缩小约 95 倍），此时 rae_c_d 与绝对量都不可比。
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cfd-analysis")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_RESULTS_ROOT = Path("experiments/results")
DEFAULT_OUTPUT_DIR = Path("analysis_plots")

COLORS = {
    "Full centralized": "#009E73",
    "PPBC": "#0072B2",
    "Cyclic training": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="实验结果的根目录（默认 experiments/results）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="图和 csv 的输出目录",
    )
    parser.add_argument(
        "--ppbc-history",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "ppbc_drivaernetpp_transformer_easy" / "history.csv",
        help="PPBC 的 history.csv（取 body=all 的最优 rae_c_d）",
    )
    parser.add_argument(
        "--central-diagnosis",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "central_checkpoint_diagnosis.csv",
        help="集中式 checkpoint 诊断 csv（含 label 与 rae_c_d 两列）",
    )
    parser.add_argument(
        "--central-value",
        type=float,
        default=None,
        help="直接指定集中式的 rae_c_d（给了它就不再读 --central-diagnosis）",
    )
    parser.add_argument(
        "--central-label",
        default="Central E+F+N epoch 79",
        help="central_checkpoint_diagnosis.csv 里用作全量集中式结果的那一行的 label",
    )
    parser.add_argument(
        "--cyclic-summary",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "cfd_body_cyclic_baseline_easy" / "summary.csv",
        help="循环训练的 summary.csv（取 body=all 的最优 rae_c_d）",
    )
    parser.add_argument("--title", default="")
    parser.add_argument("--width", type=float, default=4.2)
    parser.add_argument("--height", type=float, default=3.6)
    parser.add_argument(
        "--output-stem", default="cd_relative_error_ppbc_central_cyclic"
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _best(rows: list[dict[str, str]], *, key: str, body: str = "all") -> float:
    values = [
        float(row[key])
        for row in rows
        if row.get("body") == body and row.get(key) not in (None, "", "nan")
    ]
    if not values:
        raise ValueError(f"no body={body!r} rows with a usable {key!r} column")
    return min(values)


def best_ppbc_rae(history_path: Path) -> float:
    rows = read_csv(history_path)
    rows = [r for r in rows if r.get("method") == "ppbc"] or rows
    try:
        return _best(rows, key="rae_c_d")
    except ValueError as exc:
        raise ValueError(f"{exc} in {history_path}") from exc


def central_rae(diagnosis_path: Path, label: str) -> float:
    rows = read_csv(diagnosis_path)
    for row in rows:
        if row.get("label") == label:
            return float(row["rae_c_d"])
    raise ValueError(f"no row with label {label!r} in {diagnosis_path}")


def cyclic_rae(summary_path: Path) -> float:
    try:
        return _best(read_csv(summary_path), key="rae_c_d")
    except ValueError as exc:
        raise ValueError(f"{exc} in {summary_path}") from exc


def write_summary(items: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "rae_c_d", "rae_c_d_percent"])
        writer.writeheader()
        for item in items:
            writer.writerow(item)


def plot(
    items: list[dict[str, object]],
    output_dir: Path,
    stem: str,
    title: str,
    width: float,
    height: float,
) -> tuple[Path, Path]:
    plt.rcParams["svg.fonttype"] = "none"
    fig, ax = plt.subplots(figsize=(width, height))

    labels = [str(item["label"]) for item in items]
    values = [float(item["rae_c_d_percent"]) for item in items]
    colors = [COLORS.get(label, "#999999") for label in labels]

    bars = ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$C_D$ relative error (%)")
    if title:
        ax.set_title(title)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    max_y = max(values) if values else 1.0
    ax.set_ylim(0, max_y * 1.18)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_y * 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )
    fig.text(
        0.01,
        0.01,
        "Metric: mean relative absolute error; lower is better.",
        fontsize=7,
        color="#555555",
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)
    return pdf_path, svg_path


def main() -> None:
    args = parse_args()
    items: list[dict[str, object]] = []

    if args.central_value is not None:
        central = float(args.central_value)
    else:
        central = central_rae(args.central_diagnosis, args.central_label)
    items.append(
        {
            "label": "Full centralized",
            "rae_c_d": central,
            "rae_c_d_percent": central * 100.0,
        }
    )

    ppbc = best_ppbc_rae(args.ppbc_history)
    items.append(
        {"label": "PPBC", "rae_c_d": ppbc, "rae_c_d_percent": ppbc * 100.0}
    )

    if args.cyclic_summary.exists():
        cyclic = cyclic_rae(args.cyclic_summary)
        items.append(
            {
                "label": "Cyclic training",
                "rae_c_d": cyclic,
                "rae_c_d_percent": cyclic * 100.0,
            }
        )
    else:
        print(f"[skip] cyclic summary 不存在：{args.cyclic_summary}")

    csv_path = args.output_dir / "cd_relative_error_comparison.csv"
    write_summary(items, csv_path)
    pdf_path, svg_path = plot(
        items,
        args.output_dir,
        args.output_stem,
        args.title,
        args.width,
        args.height,
    )
    for item in items:
        print(f"{item['label']:<18} rae_c_d = {item['rae_c_d_percent']:.2f}%")
    print(f"Wrote {csv_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
