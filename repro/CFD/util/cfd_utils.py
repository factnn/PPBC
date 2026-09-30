from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from .cfd_utils_impl import (
    CFDLossWrapper,
    CFDMetricWrapper,
    calc_force_coeff,
    correct_wss_dir,
)


@torch.no_grad()
def save_results(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    others: dict[str, torch.Tensor],
    save_dir: Path,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    if "pressure" in outputs:
        pressure_pred = outputs["pressure"] * others["pressure_scale"]  # (B, N, 1)
        pressure_gt = targets["pressure"] * others["pressure_scale"]  # (B, N, 1)
        pressure_diff = torch.abs(pressure_pred - pressure_gt)  # (B, N, 1)

    if "wss" in outputs:
        wss_pred = outputs["wss"] * others["wss_scale"]  # (B, N, 3)
        wss_gt = targets["wss"] * others["wss_scale"]  # (B, N, 3)
        wss_diff = torch.norm(wss_pred - wss_gt, p=2, dim=-1, keepdim=True)  # (B, N, 1)

    B = len(others["file_path"])
    for b_i in range(B):
        file_path = Path(others["file_path"][b_i])
        poly_data: pv.PolyData = pv.read(file_path.with_suffix(".vtk"))

        if "pressure" in outputs:
            poly_data.cell_data["pressure_pred"] = pressure_pred[b_i].cpu().numpy()
            poly_data.cell_data["pressure_gt"] = pressure_gt[b_i].cpu().numpy()
            poly_data.cell_data["pressure_diff"] = pressure_diff[b_i].cpu().numpy()

        if "wss" in outputs:
            poly_data.cell_data["wss_pred"] = wss_pred[b_i].cpu().numpy()
            poly_data.cell_data["wss_gt"] = wss_gt[b_i].cpu().numpy()
            poly_data.cell_data["wss_diff"] = wss_diff[b_i].cpu().numpy()

        poly_data.save(save_dir / f"{file_path.stem}.vtk")


def calc_R_squared(
    predictions: torch.Tensor | np.ndarray, targets: torch.Tensor | np.ndarray
):
    # predictions: (N,), targets: (N,)

    ss_res = ((targets - predictions) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    return 1 - (ss_res / ss_tot)


@torch.no_grad()
def calc_force_coeff_wrapper(
    inputs: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    others: dict[str, torch.Tensor],
):
    if "pressure" in outputs:
        outputs["c_p"] = calc_force_coeff(
            others["air_density"],
            others["flow_dir"],
            others["flow_speed"],
            others["frontal_area"],
            inputs["area"],
            outputs["pressure_denorm"] * inputs["normal"],
        )
        targets["c_p"] = calc_force_coeff(
            others["air_density"],
            others["flow_dir"],
            others["flow_speed"],
            others["frontal_area"],
            inputs["area"],
            targets["pressure_denorm"] * inputs["normal"],
        )

    if "wss" in outputs:
        outputs["wss"] = correct_wss_dir(outputs["wss"], inputs["normal"])

        outputs["c_wss"] = calc_force_coeff(
            others["air_density"],
            others["flow_dir"],
            others["flow_speed"],
            others["frontal_area"],
            inputs["area"],
            outputs["wss_denorm"],
        )
        targets["c_wss"] = calc_force_coeff(
            others["air_density"],
            others["flow_dir"],
            others["flow_speed"],
            others["frontal_area"],
            inputs["area"],
            targets["wss_denorm"],
        )

    if "c_p" in outputs and "c_wss" in outputs:
        outputs["c_d"] = outputs["c_p"] + outputs["c_wss"]
        targets["c_d"] = targets["c_p"] + targets["c_wss"]
