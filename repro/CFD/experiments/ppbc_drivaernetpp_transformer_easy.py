import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from lightning.fabric import Fabric
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

import hydra
from omegaconf import OmegaConf

from datamodule.point_cloud_datamodule import PointCloudDataModule, PointCloudDataset
from model.transformer import Transformer
from util.cfd_utils import calc_force_coeff_wrapper
from util.fabric_utils import seed_everything

MODEL_CONFIG_FILE = REPO_ROOT / "config" / "model" / "transformer.yaml"
LOSS_CONFIG_FILE = REPO_ROOT / "config" / "loss" / "cfd_loss_wrapper.yaml"
METRIC_CONFIG_FILE = REPO_ROOT / "config" / "metric" / "cfd_metric_wrapper.yaml"
TRAIN_CONFIG_FILE = REPO_ROOT / "config" / "train.yaml"

BODY_TO_CLASS = {
    "E": ("estateback", 0),
    "N": ("notchback", 1),
    "F": ("fastback", 2),
}

MEAN_STD_DICT = {
    "pressure": {"mean": [0.0], "std": [500.0]},
    "wss": {"mean": [0.0], "std": [1.5]},
    "centroid": {"mean": [0.0], "std": [4.0]},
}


@dataclass(frozen=True)
class SampleRecord:
    path: str
    file_name: str
    body: str
    class_id: int


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def setup_run_log(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    log_file = log_path.open("a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    return log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FedAvg/PPBC validation with the repository Transformer on the "
            "DrivAerNet++ easy split. Vehicle body types are used as classes "
            "for Dirichlet client heterogeneity."
        )
    )
    parser.add_argument(
        "--data-root",
        default="/mnt/nfs_project_a/shared/datasets/DrivAerNetPP",
    )
    parser.add_argument("--split", default="easy", choices=["easy", "hard"])
    parser.add_argument(
        "--output-dir",
        default="experiments/results/ppbc_drivaernetpp_transformer_easy",
    )
    parser.add_argument("--methods", nargs="+", default=["fedavg", "ppbc"])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.5, 10.0])
    parser.add_argument("--num-clients", type=int, default=6)
    parser.add_argument(
        "--client-split-mode",
        default="dirichlet",
        choices=["dirichlet", "single_class"],
        help=(
            "dirichlet uses class-conditional Dirichlet heterogeneity. "
            "single_class guarantees every non-empty client contains exactly one body class."
        ),
    )
    parser.add_argument(
        "--single-class-allocation",
        default="proportional",
        choices=["equal_classes", "proportional"],
        help=(
            "How to allocate clients to body classes when --client-split-mode=single_class. "
            "proportional keeps pure clients while making client sample counts as even as possible."
        ),
    )
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--downsample-size", type=int, default=50000)
    parser.add_argument(
        "--test-downsample-size",
        type=int,
        default=-1,
        help="Test point downsample size. -1 matches the repository test.py baseline and uses all points.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--aggregation", default="equal", choices=["equal", "sample"])
    parser.add_argument("--pressure-weight", type=float, default=1.0)
    parser.add_argument("--wss-weight", type=float, default=1.0)

    parser.add_argument("--ppbc-iterations", type=int, default=5)
    parser.add_argument("--epoch-k", type=int, default=100)
    parser.add_argument("--iter-k", type=int, default=10)
    parser.add_argument("--theta", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--q-m", type=float, default=1.0)

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Stratified cap for quick checks. 0 uses the full train split.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=0,
        help="Stratified cap for quick checks. 0 uses the full test split.",
    )
    parser.add_argument(
        "--max-client-batches",
        type=int,
        default=0,
        help="Optional per-client batch cap for smoke tests. 0 means no cap.",
    )
    parser.add_argument("--eval-every", type=int, default=1)
    return parser.parse_args()


def check_device(args: argparse.Namespace) -> None:
    if args.device == "cpu":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --device cuda was requested.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_fabric(args: argparse.Namespace) -> Fabric:
    precision = "bf16-mixed" if args.precision == "bf16" else "32-true"
    accelerator = "cuda" if args.device == "cuda" else "cpu"
    devices = [0] if args.device == "cuda" else 1
    return Fabric(accelerator=accelerator, devices=devices, precision=precision)


def load_repo_cfg(args: argparse.Namespace):
    cfg = OmegaConf.create()
    cfg.seed = args.seed
    cfg.model = OmegaConf.load(MODEL_CONFIG_FILE)
    cfg.loss = OmegaConf.load(LOSS_CONFIG_FILE)
    cfg.metric = OmegaConf.load(METRIC_CONFIG_FILE)
    cfg.metric.mean_std_file = str(Path(args.data_root) / "mean_std.json")
    cfg.optimizer = OmegaConf.load(TRAIN_CONFIG_FILE).optimizer
    cfg.optimizer.lr = args.lr
    cfg.optimizer.weight_decay = args.weight_decay
    return cfg


def resolve_case_path(path: str, data_root: Path) -> str:
    if Path(path).exists():
        return path
    candidate = data_root / "npy" / Path(path).name
    if candidate.exists():
        return candidate.as_posix()
    raise FileNotFoundError(path)


def read_records(split_file: Path, data_root: Path, max_samples: int, seed: int):
    with split_file.open("r") as f:
        paths = json.load(f)
    records = []
    for raw_path in paths:
        path = resolve_case_path(raw_path, data_root)
        file_name = Path(path).stem
        body_prefix = file_name.split("_", maxsplit=1)[0]
        if body_prefix not in BODY_TO_CLASS:
            raise ValueError(f"Unknown body prefix {body_prefix!r} in {path}")
        body, class_id = BODY_TO_CLASS[body_prefix]
        records.append(SampleRecord(path, file_name, body, class_id))
    if max_samples and max_samples < len(records):
        records = stratified_cap(records, max_samples, seed)
    return records


def stratified_cap(
    records: list[SampleRecord], max_samples: int, seed: int
) -> list[SampleRecord]:
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[SampleRecord]] = defaultdict(list)
    for record in records:
        by_class[record.class_id].append(record)

    capped = []
    remaining = max_samples
    total = len(records)
    class_ids = sorted(by_class)
    for pos, class_id in enumerate(class_ids):
        class_records = by_class[class_id]
        if pos == len(class_ids) - 1:
            take = min(remaining, len(class_records))
        else:
            take = max(1, round(max_samples * len(class_records) / total))
            take = min(take, len(class_records), remaining - (len(class_ids) - pos - 1))
        choice = rng.choice(len(class_records), size=take, replace=False)
        capped.extend(class_records[i] for i in choice)
        remaining -= take
    return sorted(capped, key=lambda item: item.file_name)


def dirichlet_clients(
    classes: np.ndarray, alpha: float, num_clients: int, seed: int
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    clients = [[] for _ in range(num_clients)]
    for class_id in sorted(np.unique(classes).tolist()):
        class_indices = np.where(classes == class_id)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        raw_counts = proportions * len(class_indices)
        counts = np.floor(raw_counts).astype(int)
        remainder = len(class_indices) - int(counts.sum())
        if remainder > 0:
            order = np.argsort(raw_counts - counts)[::-1]
            counts[order[:remainder]] += 1
        cursor = 0
        for client_id, count in enumerate(counts):
            if count > 0:
                indices = class_indices[cursor : cursor + count].tolist()
                clients[client_id].extend(indices)
                cursor += count
    for indices in clients:
        rng.shuffle(indices)
    return clients


def single_class_clients(
    classes: np.ndarray,
    num_clients: int,
    seed: int,
    allocation: str,
) -> list[list[int]]:
    class_ids = sorted(np.unique(classes).tolist())
    if num_clients < len(class_ids):
        raise ValueError(
            "--num-clients must be at least the number of body classes "
            f"({len(class_ids)}) for --client-split-mode=single_class."
        )

    rng = np.random.default_rng(seed)
    class_client_counts = allocate_clients_per_class(
        classes, class_ids, num_clients, allocation
    )
    clients = [[] for _ in range(num_clients)]
    client_cursor = 0

    for class_id, class_num_clients in zip(class_ids, class_client_counts):
        class_indices = np.where(classes == class_id)[0]
        rng.shuffle(class_indices)
        split_indices = np.array_split(class_indices, class_num_clients)
        for indices in split_indices:
            clients[client_cursor] = indices.tolist()
            rng.shuffle(clients[client_cursor])
            client_cursor += 1

    return clients


def allocate_clients_per_class(
    classes: np.ndarray,
    class_ids: list[int],
    num_clients: int,
    allocation: str,
) -> list[int]:
    if allocation == "equal_classes":
        base = num_clients // len(class_ids)
        remainder = num_clients % len(class_ids)
        counts = [base] * len(class_ids)
        class_sizes = np.array([int(np.sum(classes == class_id)) for class_id in class_ids])
        for idx in np.argsort(class_sizes)[::-1][:remainder]:
            counts[int(idx)] += 1
        return counts

    class_sizes = np.array([int(np.sum(classes == class_id)) for class_id in class_ids])
    raw = class_sizes / class_sizes.sum() * num_clients
    counts = np.maximum(1, np.floor(raw).astype(int))
    while int(counts.sum()) > num_clients:
        candidates = np.where(counts > 1)[0]
        idx = candidates[np.argmin(raw[candidates] - counts[candidates])]
        counts[idx] -= 1
    while int(counts.sum()) < num_clients:
        idx = int(np.argmax(raw - counts))
        counts[idx] += 1
    return counts.tolist()


def build_clients(args: argparse.Namespace, classes: np.ndarray, alpha: float) -> list[list[int]]:
    seed = args.seed + int(alpha * 1000)
    if args.client_split_mode == "single_class":
        return single_class_clients(
            classes,
            args.num_clients,
            seed,
            args.single_class_allocation,
        )
    return dirichlet_clients(classes, alpha, args.num_clients, seed)


def client_distribution(clients: list[list[int]], classes: np.ndarray) -> np.ndarray:
    counts = np.zeros((len(clients), len(BODY_TO_CLASS)), dtype=np.int64)
    for client_id, indices in enumerate(clients):
        for index in indices:
            counts[client_id, classes[index]] += 1
    return counts


def distribution_stats(counts: np.ndarray) -> dict[str, float | int]:
    totals = counts.sum(axis=1)
    non_empty = totals > 0
    proportions = counts[non_empty] / totals[non_empty, None]
    log_proportions = np.zeros_like(proportions, dtype=np.float64)
    np.log(proportions, out=log_proportions, where=proportions > 0)
    entropy = -np.sum(proportions * log_proportions, axis=1)
    max_entropy = math.log(counts.shape[1])
    return {
        "non_empty_clients": int(non_empty.sum()),
        "mean_client_size": float(totals[non_empty].mean()) if non_empty.any() else 0.0,
        "min_client_size": int(totals[non_empty].min()) if non_empty.any() else 0,
        "max_client_size": int(totals[non_empty].max()) if non_empty.any() else 0,
        "mean_normalized_entropy": float((entropy / max_entropy).mean())
        if non_empty.any()
        else 0.0,
    }


def validate_client_purity(
    clients: list[list[int]],
    classes: np.ndarray,
    mode: str,
) -> None:
    if mode != "single_class":
        return
    mixed_clients = []
    for client_id, indices in enumerate(clients):
        if indices and len(set(classes[indices].tolist())) > 1:
            mixed_clients.append(client_id)
    if mixed_clients:
        raise RuntimeError(
            "single_class split produced mixed-class clients: "
            f"{mixed_clients}"
        )


def make_dataset(
    records: list[SampleRecord], downsample_size: int
) -> PointCloudDataset:
    return PointCloudDataset(
        [record.path for record in records],
        MEAN_STD_DICT,
        downsample_size=downsample_size,
    )


def make_model() -> Transformer:
    return hydra.utils.instantiate(OmegaConf.load(MODEL_CONFIG_FILE))


def make_optimizer(model: torch.nn.Module, repo_cfg):
    return hydra.utils.instantiate(repo_cfg.optimizer, params=model.parameters())


def make_loss_wrapper(args: argparse.Namespace, repo_cfg):
    loss_cfg = OmegaConf.create(OmegaConf.to_container(repo_cfg.loss, resolve=True))
    loss_cfg.mse_loss_pressure = args.pressure_weight
    loss_cfg.mse_loss_wss = args.wss_weight
    return hydra.utils.instantiate(loss_cfg)


def make_metric_wrapper(repo_cfg):
    return hydra.utils.instantiate(repo_cfg.metric)


def clone_state_cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def zero_state_like(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: torch.zeros_like(value, device="cpu")
        for key, value in state.items()
        if torch.is_floating_point(value)
    }


def add_scaled_state(
    target: dict[str, torch.Tensor],
    update: dict[str, torch.Tensor],
    scale: float,
) -> None:
    for key, value in update.items():
        target[key].add_(value, alpha=scale)


def state_plus_update(
    state: dict[str, torch.Tensor],
    update: dict[str, torch.Tensor],
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    new_state = clone_state_cpu(state)
    for key, value in update.items():
        new_state[key].add_(value, alpha=scale)
    return new_state


def train_client(
    base_state: dict[str, torch.Tensor],
    dataset: PointCloudDataset,
    indices: list[int],
    args: argparse.Namespace,
    repo_cfg,
    fabric: Fabric,
) -> tuple[dict[str, torch.Tensor], float]:
    model = make_model()
    model.load_state_dict(base_state)
    optimizer = make_optimizer(model, repo_cfg)
    model, optimizer = fabric.setup(model, optimizer)

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=fabric.device.type == "cuda",
        drop_last=False,
    )
    loader = fabric.setup_dataloaders(loader)
    loss_wrapper = make_loss_wrapper(args, repo_cfg)
    model.train()

    total_loss = 0.0
    total_batches = 0
    for _ in range(args.local_epochs):
        for batch_idx, data in enumerate(loader):
            if args.max_client_batches and batch_idx >= args.max_client_batches:
                break
            inputs, targets, others = data
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss, _ = loss_wrapper(inputs, outputs, targets, others)
            fabric.backward(loss)
            if args.grad_clip > 0:
                fabric.clip_gradients(
                    model, optimizer, max_norm=args.grad_clip, norm_type=2
                )
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_batches += 1

    local_state = model.state_dict()
    grad = {}
    for key, base_value in base_state.items():
        if torch.is_floating_point(base_value):
            grad[key] = local_state[key].detach().cpu() - base_value
    del model
    if fabric.device.type == "cuda":
        torch.cuda.empty_cache()
    return grad, total_loss / max(total_batches, 1)


def client_weight(
    client_id: int,
    active_clients: list[int],
    clients: list[list[int]],
    total_samples: int,
    aggregation: str,
) -> float:
    if aggregation == "sample":
        return len(clients[client_id]) / max(total_samples, 1)
    return 1.0 / max(len(active_clients), 1)


def evaluate_state(
    state: dict[str, torch.Tensor],
    dataset: PointCloudDataset,
    records: list[SampleRecord],
    args: argparse.Namespace,
    repo_cfg,
    fabric: Fabric,
    datamodule: PointCloudDataModule,
    method: str,
    alpha: float,
    round_idx: int,
    train_loss: float | None = None,
) -> list[dict[str, float | int | str]]:
    model = make_model()
    model.load_state_dict(state)
    model = fabric.setup_module(model)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=fabric.device.type == "cuda",
        drop_last=False,
    )
    loader = fabric.setup_dataloaders(loader)
    metric_wrapper = make_metric_wrapper(repo_cfg)
    case_rows = []

    with torch.no_grad():
        for inputs, targets, others in loader:
            file_paths = others["file_path"]
            outputs = model(inputs)
            datamodule._denormalize(outputs, targets, others)

            calc_force_coeff_wrapper(inputs, outputs, targets, others)
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
            metric_dict = metric_wrapper(outputs, targets)
            case_rows.extend(case_result_rows(file_paths, outputs, targets, metric_dict))

    del model
    if fabric.device.type == "cuda":
        torch.cuda.empty_cache()

    base = {
        "method": method,
        "alpha": alpha,
        "round": round_idx,
        "train_loss": float("nan") if train_loss is None else train_loss,
    }
    rows = [metric_row(base, "all", case_rows)]
    for _, (body_name, _) in BODY_TO_CLASS.items():
        body_rows = [row for row in case_rows if row["body"] == body_name]
        rows.append(metric_row(base, body_name, body_rows))
    return rows


def case_result_rows(
    file_paths: list[str],
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    metric_dict: dict[str, torch.Tensor],
) -> list[dict[str, float | str]]:
    rows = []
    for idx, file_path in enumerate(file_paths):
        prefix = Path(file_path).stem.split("_", maxsplit=1)[0]
        row = {
            "file_name": Path(file_path).stem,
            "body": BODY_TO_CLASS[prefix][0],
            "c_p_pred": float(outputs["c_p"][idx].detach().cpu()),
            "c_p_gt": float(targets["c_p"][idx].detach().cpu()),
            "c_wss_pred": float(outputs["c_wss"][idx].detach().cpu()),
            "c_wss_gt": float(targets["c_wss"][idx].detach().cpu()),
            "c_d_pred": float(outputs["c_d"][idx].detach().cpu()),
            "c_d_gt": float(targets["c_d"][idx].detach().cpu()),
        }
        for metric_name, metric in metric_dict.items():
            row[metric_name] = float(metric.detach().cpu())
        rows.append(row)
    return rows


def metric_row(
    base: dict[str, float | int | str],
    body: str,
    case_rows: list[dict[str, float | str]],
) -> dict[str, float | int | str]:
    row = dict(base)
    row["body"] = body
    row["n"] = len(case_rows)
    if not case_rows:
        row.update(
            {
                "c_d_mae": float("nan"),
                "c_d_rmse": float("nan"),
                "c_d_r2": float("nan"),
                "c_p_mae": float("nan"),
                "c_p_rmse": float("nan"),
                "c_wss_mae": float("nan"),
                "c_wss_rmse": float("nan"),
                "mse_p": float("nan"),
                "mse_wss": float("nan"),
                "rae_c_p": float("nan"),
                "rae_c_wss": float("nan"),
                "rae_c_d": float("nan"),
            }
        )
        return row
    c_d_pred = np.array([case["c_d_pred"] for case in case_rows], dtype=np.float64)
    c_d_gt = np.array([case["c_d_gt"] for case in case_rows], dtype=np.float64)
    c_p_pred = np.array([case["c_p_pred"] for case in case_rows], dtype=np.float64)
    c_p_gt = np.array([case["c_p_gt"] for case in case_rows], dtype=np.float64)
    c_wss_pred = np.array([case["c_wss_pred"] for case in case_rows], dtype=np.float64)
    c_wss_gt = np.array([case["c_wss_gt"] for case in case_rows], dtype=np.float64)
    row["c_d_mae"] = mae(c_d_pred, c_d_gt)
    row["c_d_rmse"] = rmse(c_d_pred, c_d_gt)
    row["c_d_r2"] = r2(c_d_pred, c_d_gt)
    row["c_p_mae"] = mae(c_p_pred, c_p_gt)
    row["c_p_rmse"] = rmse(c_p_pred, c_p_gt)
    row["c_wss_mae"] = mae(c_wss_pred, c_wss_gt)
    row["c_wss_rmse"] = rmse(c_wss_pred, c_wss_gt)
    for metric_name in ("mse_p", "mse_wss", "rae_c_p", "rae_c_wss", "rae_c_d"):
        row[metric_name] = mean_case_metric(case_rows, metric_name)
    return row


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target))) if len(pred) else float("nan")


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2))) if len(pred) else float("nan")


def r2(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) == 0:
        return float("nan")
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def mean_case_metric(case_rows: list[dict[str, float | str]], metric_name: str) -> float:
    values = [float(case[metric_name]) for case in case_rows if metric_name in case]
    return float(np.mean(values)) if values else float("nan")


def run_fedavg(
    initial_state: dict[str, torch.Tensor],
    train_dataset: PointCloudDataset,
    test_dataset: PointCloudDataset,
    test_records: list[SampleRecord],
    clients: list[list[int]],
    args: argparse.Namespace,
    repo_cfg,
    fabric: Fabric,
    datamodule: PointCloudDataModule,
    alpha: float,
) -> list[dict[str, float | int | str]]:
    state = clone_state_cpu(initial_state)
    active_clients = [client_id for client_id, indices in enumerate(clients) if indices]
    total_samples = sum(len(clients[client_id]) for client_id in active_clients)
    history = evaluate_state(
        state,
        test_dataset,
        test_records,
        args,
        repo_cfg,
        fabric,
        datamodule,
        "fedavg",
        alpha,
        0,
    )

    for round_idx in range(1, args.rounds + 1):
        aggregate_update = zero_state_like(state)
        losses = []
        for client_id in active_clients:
            grad, loss = train_client(
                state, train_dataset, clients[client_id], args, repo_cfg, fabric
            )
            weight = client_weight(
                client_id, active_clients, clients, total_samples, args.aggregation
            )
            add_scaled_state(aggregate_update, grad, weight)
            losses.append(loss)
        state = state_plus_update(state, aggregate_update)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        if should_evaluate(round_idx, args):
            rows = evaluate_state(
                state,
                test_dataset,
                test_records,
                args,
                repo_cfg,
                fabric,
                datamodule,
                "fedavg",
                alpha,
                round_idx,
                train_loss,
            )
            history.extend(rows)
            print_round(rows[0], args.rounds)
    return history


def run_ppbc(
    initial_state: dict[str, torch.Tensor],
    train_dataset: PointCloudDataset,
    test_dataset: PointCloudDataset,
    test_records: list[SampleRecord],
    clients: list[list[int]],
    args: argparse.Namespace,
    repo_cfg,
    fabric: Fabric,
    datamodule: PointCloudDataModule,
    alpha: float,
) -> list[dict[str, float | int | str]]:
    state = clone_state_cpu(initial_state)
    active_clients = [client_id for client_id, indices in enumerate(clients) if indices]
    total_samples = sum(len(clients[client_id]) for client_id in active_clients)
    rng = np.random.default_rng(args.seed + int(alpha * 1000) + 17)
    final_errors = {client_id: zero_state_like(state) for client_id in active_clients}
    history = evaluate_state(
        state,
        test_dataset,
        test_records,
        args,
        repo_cfg,
        fabric,
        datamodule,
        "ppbc",
        alpha,
        0,
    )

    for round_idx in range(1, args.rounds + 1):
        working_state = clone_state_cpu(state)
        if round_idx > 1:
            for client_id in active_clients:
                add_scaled_state(working_state, final_errors[client_id], 0.5)

        epoch_clients = select_clients(rng, active_clients, args.epoch_k)
        current_errors = {
            client_id: zero_state_like(state) for client_id in active_clients
        }
        q_mask = {
            client_id: float(rng.random() < args.q_m) / max(args.q_m, 1.0e-12)
            for client_id in active_clients
        }
        round_losses = []

        for iter_idx in range(args.ppbc_iterations):
            selected_clients = set(select_clients(rng, epoch_clients, args.iter_k))
            aggregate_update = zero_state_like(state)
            for client_id in active_clients:
                grad, loss = train_client(
                    working_state,
                    train_dataset,
                    clients[client_id],
                    args,
                    repo_cfg,
                    fabric,
                )
                round_losses.append(loss)
                base_weight = client_weight(
                    client_id, active_clients, clients, total_samples, args.aggregation
                )
                iter_weight = base_weight if client_id in selected_clients else 0.0
                prob = q_mask[client_id]
                error_scale = (1.0 - args.theta) * (base_weight - iter_weight) * prob
                add_scaled_state(current_errors[client_id], grad, error_scale)
                grad_scale = args.gamma * (1.0 - args.theta) * iter_weight * prob
                add_scaled_state(aggregate_update, grad, grad_scale)
                add_scaled_state(
                    aggregate_update,
                    final_errors[client_id],
                    args.gamma * args.theta * 0.5,
                )
            working_state = state_plus_update(working_state, aggregate_update)
            if iter_idx == args.ppbc_iterations - 1:
                final_errors = current_errors

        state = working_state
        train_loss = float(np.mean(round_losses)) if round_losses else float("nan")
        if should_evaluate(round_idx, args):
            rows = evaluate_state(
                state,
                test_dataset,
                test_records,
                args,
                repo_cfg,
                fabric,
                datamodule,
                "ppbc",
                alpha,
                round_idx,
                train_loss,
            )
            history.extend(rows)
            print_round(rows[0], args.rounds)
    return history


def select_clients(
    rng: np.random.Generator, candidates: list[int], k: int
) -> list[int]:
    if k <= 0 or k >= len(candidates):
        return list(candidates)
    return rng.choice(candidates, size=k, replace=False).tolist()


def should_evaluate(round_idx: int, args: argparse.Namespace) -> bool:
    return round_idx == args.rounds or (
        args.eval_every > 0 and round_idx % args.eval_every == 0
    )


def print_round(row: dict[str, float | int | str], total_rounds: int) -> None:
    metric_names = [
        "c_d_mae",
        "c_d_rmse",
        "c_d_r2",
        "c_p_mae",
        "c_p_rmse",
        "c_wss_mae",
        "c_wss_rmse",
        "mse_p",
        "mse_wss",
        "rae_c_p",
        "rae_c_wss",
        "rae_c_d",
    ]
    metrics = ", ".join(
        f"{metric_name}: {format_metric(row[metric_name])}"
        for metric_name in metric_names
    )
    print(
        f"[{str(row['method']).upper()}] alpha={float(row['alpha']):g}, "
        f"round={row['round']}/{total_rounds}, body={row['body']}, n={row['n']}, "
        f"train_loss: {format_metric(row['train_loss'])}, <Metrics> {metrics}",
        flush=True,
    )


def format_metric(value: float | int | str) -> str:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value_float):
        return "nan"
    return f"{value_float:.4e}"


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def final_summary(history: list[dict[str, float | int | str]], rounds: int):
    return [
        row
        for row in history
        if int(row["round"]) == rounds and str(row["body"]) == "all"
    ]


def main() -> None:
    args = parse_args()
    # [patch] 归一化统计量以数据根目录的 mean_std.json 为准（训练与指标共用一套，§9）
    _mean_std_path = Path(args.data_root) / "mean_std.json"
    if _mean_std_path.exists():
        global MEAN_STD_DICT
        MEAN_STD_DICT = json.loads(_mean_std_path.read_text(encoding="utf-8"))
        print(f"[patch] 已从 {_mean_std_path} 载入归一化统计量", flush=True)
    if not set(args.methods).issubset({"fedavg", "ppbc"}):
        raise ValueError("--methods must contain only fedavg and/or ppbc")
    if args.eval_batch_size != 1:
        raise ValueError(
            "--eval-batch-size must be 1 to match the repository test.py metric semantics."
        )
    output_dir = Path(args.output_dir)
    log_path = setup_run_log(output_dir)
    print(f"Logging to {log_path}", flush=True)
    check_device(args)
    repo_cfg = load_repo_cfg(args)
    fabric = make_fabric(args)
    fabric.launch()
    seed_everything(repo_cfg, fabric, workers=True)
    set_seed(args.seed)

    data_root = Path(args.data_root)
    split_dir = data_root / "split" / args.split
    train_records = read_records(
        split_dir / "train_split.json", data_root, args.max_train_samples, args.seed
    )
    test_records = read_records(
        split_dir / "test_split.json", data_root, args.max_test_samples, args.seed + 1
    )
    train_classes = np.array(
        [record.class_id for record in train_records], dtype=np.int64
    )
    train_downsample = args.downsample_size
    test_downsample = args.test_downsample_size
    train_dataset = make_dataset(train_records, train_downsample)
    test_dataset = make_dataset(test_records, test_downsample)
    datamodule = PointCloudDataModule(MEAN_STD_DICT)

    fabric.print(
        f"Loaded DrivAerNet++ {args.split}: {len(train_records)} train, "
        f"{len(test_records)} test. Model=repository Transformer via Fabric.",
    )

    initial_model = make_model()
    initial_state = clone_state_cpu(initial_model.state_dict())
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    config["train_cases"] = len(train_records)
    config["test_cases"] = len(test_records)
    config["mean_std_dict"] = MEAN_STD_DICT
    config["model_config_file"] = MODEL_CONFIG_FILE.as_posix()
    config["model_config"] = OmegaConf.to_container(
        OmegaConf.load(MODEL_CONFIG_FILE), resolve=True
    )
    config["metric_config_file"] = METRIC_CONFIG_FILE.as_posix()
    config["metric_config"] = OmegaConf.to_container(repo_cfg.metric, resolve=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    all_history = []
    distribution_rows = []
    for alpha in args.alphas:
        clients = build_clients(args, train_classes, alpha)
        validate_client_purity(clients, train_classes, args.client_split_mode)
        counts = client_distribution(clients, train_classes)
        stats = distribution_stats(counts)
        stats.update(
            {
                "alpha": alpha,
                "client_split_mode": args.client_split_mode,
                "single_class_allocation": args.single_class_allocation,
            }
        )
        distribution_rows.append(stats)
        np.savetxt(
            output_dir / f"client_class_counts_alpha_{alpha:g}.csv",
            counts,
            delimiter=",",
            fmt="%d",
            header="estateback,notchback,fastback",
            comments="",
        )
        print(f"alpha={alpha:g} distribution: {stats}", flush=True)

        if "fedavg" in args.methods:
            set_seed(args.seed)
            all_history.extend(
                run_fedavg(
                    initial_state,
                    train_dataset,
                    test_dataset,
                    test_records,
                    clients,
                    args,
                    repo_cfg,
                    fabric,
                    datamodule,
                    alpha,
                )
            )
        if "ppbc" in args.methods:
            set_seed(args.seed)
            all_history.extend(
                run_ppbc(
                    initial_state,
                    train_dataset,
                    test_dataset,
                    test_records,
                    clients,
                    args,
                    repo_cfg,
                    fabric,
                    datamodule,
                    alpha,
                )
            )

    write_csv(output_dir / "history.csv", all_history)
    write_csv(output_dir / "summary.csv", final_summary(all_history, args.rounds))
    write_csv(output_dir / "distribution_summary.csv", distribution_rows)
    print(f"Wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
