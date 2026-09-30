import logging
import os
from pathlib import Path

import torch
from lightning.fabric import Fabric
from tqdm import tqdm


def seed_everything(cfg, fabric: Fabric, workers: bool = True):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    fabric.seed_everything(cfg.seed, workers=workers)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def load_checkpoint(
    cfg,
    fabric: Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    train_dataloader: torch.utils.data.DataLoader = None,
    test_dataloader: torch.utils.data.DataLoader = None,
):
    start_epoch = 0
    if cfg.ckpt_file:
        state = {"model": model, "epoch": 0}
        if cfg.mode == "train":
            state["optimizer"] = optimizer

        fabric.load(cfg.ckpt_file, state)
        if fabric.is_global_zero:
            logging.info(
                f"Loaded model from checkpoint {cfg.ckpt_file} at epoch {state['epoch']}"
            )

        if cfg.mode == "train":
            start_epoch = state["epoch"] + 1

            for epoch in tqdm(
                range(0, start_epoch),
                desc="Resuming",
                dynamic_ncols=True,
                leave=False,
                disable=not fabric.is_global_zero,
            ):
                for _ in train_dataloader:
                    break

                if (epoch + 1) % cfg.test_interval == 0:
                    for _ in test_dataloader:
                        break

            fabric.barrier()

            if fabric.is_global_zero:
                logging.info(f"Resuming training from epoch {start_epoch}")

    return start_epoch


def setup_logger(
    logger_type: str, name: str, log_dir: str, config: dict = None, resume: bool = False
):
    log_dir = Path(log_dir)

    if logger_type == "":
        logger = None
    elif logger_type == "wandb":
        from wandb.integration.lightning.fabric import WandbLogger

        logger = WandbLogger(
            project=name,
            name=log_dir.name,
            config=config,
            id=log_dir.name if resume else None,
            resume="must" if resume else "never",
        )
    elif logger_type == "swanlab":
        from swanlab.integration.pytorch_lightning import SwanLabLogger

        logger = SwanLabLogger(
            project=name,
            experiment_name=log_dir.name,
            config=config,
            id=log_dir.name if resume else None,
            resume="must" if resume else "never",
        )
    else:
        raise ValueError(f"Unknown logger type: {logger_type}")

    return logger


@torch.no_grad()
def log_results(
    fabric: Fabric,
    mode: str,
    loss_dict: dict[str, torch.Tensor],
    metric_dict: dict[str, torch.Tensor],
    epoch: int,
    lr: float = None,
):
    log_dict = {}

    log_dict.update(
        {f"{mode}_loss/{loss_name}": loss for loss_name, loss in loss_dict.items()}
    )
    log_dict.update(
        {
            f"{mode}_metric/{metric_name}": metric
            for metric_name, metric in metric_dict.items()
        }
    )
    if lr is not None:
        log_dict["trainer/lr"] = lr

    fabric.log_dict(log_dict, step=epoch)


@torch.no_grad()
def print_results(
    fabric: Fabric,
    name: str,
    loss_dict: dict[str, torch.Tensor],
    metric_dict: dict[str, torch.Tensor],
    epoch: int,
):
    if fabric.is_global_zero:
        total_loss = loss_dict["total_loss"]
        logging.info(
            f"[{name.upper()}] Epoch: {epoch}, Total Loss: {total_loss.item():.4e}"
        )
        logging.info(
            f"<Losses> {', '.join([f'{loss_name}: {loss.item():.4e}' for loss_name, loss in loss_dict.items() if loss_name != 'total_loss'])}"
        )
        logging.info(
            f"<Metrics> {', '.join([f'{metric_name}: {metric.item():.4e}' for metric_name, metric in metric_dict.items()])}"
        )
