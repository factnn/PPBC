#!/usr/bin/env python3
"""DrivAerStar(VTK) → DrivAerNet++ 风格(.npy/.npz) 适配器。

目的：让 leiyao 那套 FedAvg/PPBC 联邦学习循环**零改动**就能跑你们自己的 DrivAerStar 数据集。

产出的结构与 DrivAerNetPP **逐字段同构**（已核对）：

    npy/<STYLE>_<id>.npy   结构化数组（全 float32）
        centroid (N,3)  area (N,1)  normal (N,3)  pressure (N,1)  wss (N,3)
    npy/<STYLE>_<id>.npz
        num_points(int64) center(3,) flow_dir(1,3) flow_speed() air_density()
        frontal_area() c_p() c_wss() c_d()
    split/{train,test}_split.json   按 style 分层、确定性切分
    mean_std.json                   pressure/wss/centroid 的均值方差（给训练归一化用）

物理约定与 `DrivAerStar_Maker/6.vtk_to_force_coefficient/force_coefficient.py` 一致：
    F = Σ_cells [ p·n̂ + τ ]·A           （n̂ 为 cell normal，已归一化）
    q = 0.5·ρ·U²,  C = F·flow_dir / (q·A_ref),  ρ=1.25, U=40.0, flow_dir=(1,0,0)
    A_ref 取 frontal_area/frontal_area_<STYLE>.csv

自校验：`--cd-ref` 指向 `cd_drag_results.zip` 时，会把算出的 c_d 与官方 Cd 对比并打印误差分布。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pyvista as pv

STYLES = ("E", "F", "N")
NPY_DTYPE = np.dtype(
    [
        ("centroid", "<f4", (3,)),
        ("area", "<f4", (1,)),
        ("normal", "<f4", (3,)),
        ("pressure", "<f4", (1,)),
        ("wss", "<f4", (3,)),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/mnt/cfd/peiyu/CFD/datasets/DrivAerStar_full/drivaerstar"),
        help="DrivAerStar 根目录（含 vtk_E / vtk_F / vtk_N）",
    )
    parser.add_argument("--out", type=Path, required=True, help="输出根目录")
    parser.add_argument("--styles", nargs="+", default=list(STYLES), choices=list(STYLES))
    parser.add_argument("--limit", type=int, default=0, help="每个 style 最多转多少例（0=全部）")
    parser.add_argument("--rho", type=float, default=1.25)
    parser.add_argument("--u", type=float, default=40.0)
    parser.add_argument(
        "--frontal-dir",
        type=Path,
        default=Path(
            "/mnt/cfd/peiyu/CFD/DrivAerStar/DrivAerStar_Maker/"
            "6.vtk_to_force_coefficient/frontal_area"
        ),
    )
    parser.add_argument(
        "--cd-ref",
        type=Path,
        default=Path("/mnt/cfd/peiyu/CFD/DrivAerStar/cd_drag_results.zip"),
        help="官方 Cd 参考（zip，内含 stl_<STYLE>-cd_drag_results.csv）；不存在则跳过校验",
    )
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_frontal_areas(frontal_dir: Path) -> dict[str, dict[str, float]]:
    areas: dict[str, dict[str, float]] = {}
    for style in STYLES:
        path = frontal_dir / f"frontal_area_{style}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            key = [k for k in reader.fieldnames if k.lower() != "id"][0]
            areas[style] = {row["id"]: float(row[key]) for row in reader}
    return areas


def load_reference_cd(cd_ref: Path) -> dict[str, float]:
    """{'F/00002': 0.2569179, ...}"""
    ref: dict[str, float] = {}
    if not cd_ref.exists():
        return ref
    with zipfile.ZipFile(cd_ref) as z:
        for name in z.namelist():
            if not name.endswith(".csv"):
                continue
            # stl_F-cd_drag_results.csv -> F
            style = Path(name).stem.split("_")[1].split("-")[0]
            with z.open(name) as fh:
                text = fh.read().decode("utf-8", "ignore").splitlines()
            for row in csv.DictReader(text):
                cd_value = row.get("Cd") or row.get("\ufeffCd")
                case_id = row.get("ID") or row.get("\ufeffID")
                if cd_value and case_id:
                    ref[f"{style}/{case_id}"] = float(cd_value)
    return ref


def convert_case(
    vtk_path: Path, style: str, area_ref: float, rho: float, u: float
) -> tuple[np.ndarray, dict[str, float]]:
    mesh = pv.read(vtk_path)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()

    pressure = np.asarray(mesh.cell_data["Pressure"], dtype=np.float64).reshape(-1)
    normals = np.asarray(mesh.cell_data["Normals"], dtype=np.float64)
    areas = np.asarray(mesh.cell_data["Area"], dtype=np.float64).reshape(-1)
    wss = np.stack(
        [
            np.asarray(mesh.cell_data[f"WallShearStress{k}"], dtype=np.float64).reshape(-1)
            for k in ("i", "j", "k")
        ],
        axis=1,
    )
    centroids = np.asarray(mesh.cell_centers().points, dtype=np.float64)
    if not (len(pressure) == len(normals) == len(areas) == len(wss) == len(centroids)):
        raise RuntimeError(
            f"字段长度不一致: p={len(pressure)} n={len(normals)} a={len(areas)} "
            f"wss={len(wss)} c={len(centroids)}"
        )

    nrm = np.linalg.norm(normals, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    normals_u = normals / nrm

    # 与 force_coefficient.py 一致：F = Σ [p·n̂ + τ]·A，再投影到流向
    flow_dir = np.array([1.0, 0.0, 0.0])
    f_pressure = (pressure[:, None] * normals_u) * areas[:, None]
    f_shear = wss * areas[:, None]
    f_total = f_pressure + f_shear
    q = 0.5 * rho * u**2
    denom = q * area_ref
    c_p = float(f_pressure.sum(axis=0) @ flow_dir / denom)
    c_wss = float(f_shear.sum(axis=0) @ flow_dir / denom)
    c_d = c_p + c_wss

    center = np.array(
        [
            (mesh.bounds[0] + mesh.bounds[1]) / 2.0,
            (mesh.bounds[2] + mesh.bounds[3]) / 2.0,
            (mesh.bounds[4] + mesh.bounds[5]) / 2.0,
        ],
        dtype=np.float32,
    )

    structured = np.empty(len(areas), dtype=NPY_DTYPE)
    structured["centroid"] = centroids.astype(np.float32)
    structured["area"] = areas.astype(np.float32).reshape(-1, 1)
    structured["normal"] = normals_u.astype(np.float32)
    structured["pressure"] = pressure.astype(np.float32).reshape(-1, 1)
    structured["wss"] = wss.astype(np.float32)

    meta = {
        "num_points": len(areas),
        "center": center,
        "c_p": c_p,
        "c_wss": c_wss,
        "c_d": c_d,
        "frontal_area": area_ref,
    }
    return structured, meta


def main() -> None:
    args = parse_args()
    out_npy = args.out / "npy"
    out_npy.mkdir(parents=True, exist_ok=True)

    frontal = load_frontal_areas(args.frontal_dir)
    ref_cd = load_reference_cd(args.cd_ref)
    if ref_cd:
        print(f"载入官方 Cd 参考 {len(ref_cd)} 条，用于自校验")
    else:
        print("[warn] 没找到官方 Cd 参考，跳过校验")

    written: dict[str, list[str]] = {s: [] for s in args.styles}
    dims = {"pressure": 1, "wss": 3, "centroid": 3}
    stats = {
        key: {"sum": np.zeros(dim), "sumsq": np.zeros(dim), "n": 0}
        for key, dim in dims.items()
    }
    errors: list[tuple[str, float, float]] = []

    t0 = time.time()
    for style in args.styles:
        files = sorted((args.src / f"vtk_{style}").glob("*.vtk"))
        if args.limit:
            files = files[: args.limit]
        for i, vtk_path in enumerate(files, 1):
            case_id = vtk_path.stem
            if case_id not in frontal[style]:
                print(f"  [skip] {style}/{case_id}: frontal area 表里没有")
                continue
            structured, meta = convert_case(
                vtk_path, style, frontal[style][case_id], args.rho, args.u
            )
            stem = f"{style}_{case_id}"
            np.save(out_npy / f"{stem}.npy", structured)
            np.savez(
                out_npy / f"{stem}.npz",
                num_points=np.int64(meta["num_points"]),
                center=meta["center"],
                flow_dir=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                flow_speed=np.float64(args.u),
                air_density=np.float64(args.rho),
                frontal_area=np.float64(meta["frontal_area"]),
                c_p=np.float64(meta["c_p"]),
                c_wss=np.float64(meta["c_wss"]),
                c_d=np.float64(meta["c_d"]),
            )
            written[style].append(stem)

            for key in dims:
                arr = structured[key].astype(np.float64)
                stats[key]["sum"] += arr.sum(axis=0)
                stats[key]["sumsq"] += (arr**2).sum(axis=0)
                stats[key]["n"] += arr.shape[0]
            if f"{style}/{case_id}" in ref_cd:
                errors.append((stem, meta["c_d"], ref_cd[f"{style}/{case_id}"]))
            if i % 10 == 0 or i == len(files):
                print(f"  {style}: {i}/{len(files)}  ({time.time()-t0:.0f}s)", flush=True)

    # split：按 style 分层、确定性随机切分
    rng = random.Random(args.seed)
    train, test = [], []
    for style in args.styles:
        stems = written[style]
        rng.shuffle(stems)
        n_test = max(1, int(round(len(stems) * args.test_frac)))
        test.extend(stems[:n_test])
        train.extend(stems[n_test:])
    split_dir = args.out / "split"
    (split_dir / "easy").mkdir(parents=True, exist_ok=True)
    for name, stems in (("train", train), ("test", test)):
        paths = [str((out_npy / f"{s}.npy").resolve()) for s in sorted(stems)]
        payload = json.dumps(paths)
        (split_dir / f"{name}_split.json").write_text(payload, encoding="utf-8")
        # FL 脚本按 <data_root>/split/<easy|hard>/ 找，所以两处都写一份
        (split_dir / "easy" / f"{name}_split.json").write_text(payload, encoding="utf-8")
    print(f"split: train={len(train)} test={len(test)} -> {split_dir}")

    # mean_std.json（训练归一化用；逐分量真实 mean/std，与训练脚本共用同一套）
    mean_std = {}
    for key, dim in dims.items():
        entry = stats[key]
        if entry["n"] == 0:
            continue
        mean = entry["sum"] / entry["n"]
        var = np.maximum(entry["sumsq"] / entry["n"] - mean**2, 0.0)
        mean_std[key] = {
            "mean": [float(x) for x in mean],
            "std": [float(x) for x in np.sqrt(var)],
        }
    (args.out / "mean_std.json").write_text(
        json.dumps(mean_std, indent=2), encoding="utf-8"
    )
    print(
        "mean_std.json:",
        {k: [round(s, 4) for s in v["std"]] for k, v in mean_std.items()},
    )

    if errors:
        rel = np.array([abs(p - r) / r for _, p, r in errors])
        print(f"\n自校验（vs 官方 Cd，{len(errors)} 例）：")
        print(f"  平均相对误差 {rel.mean():.3%} | 中位数 {np.median(rel):.3%} | 最大 {rel.max():.3%}")
        for stem, pred, ref in errors[:3]:
            print(f"    {stem}: 我算 {pred:.6f} vs 官方 {ref:.6f}")
    print(f"\n完成：{sum(len(v) for v in written.values())} 例，用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
