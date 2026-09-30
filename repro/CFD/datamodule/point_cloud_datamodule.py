import numpy as np
import torch
import torch.utils.data

from .base_datamodule import BaseDataModule, BaseDataset


class PointCloudDataset(BaseDataset):
    def __init__(
        self,
        files: list[str],
        mean_std_dict: dict[str, dict[str, list[float] | float]],
        downsample_size: int = -1,
    ):
        super().__init__(files, downsample_size)

        self.mean_std_dict = self._parse_mean_std_dict(mean_std_dict)

    def _parse_mean_std_dict(self, mean_std_dict: dict[str, dict[str, list[float]]]):
        mean_std_dict_new = {}
        for name, mean_std in mean_std_dict.items():
            mean_std_dict_new[name] = {
                "mean": torch.tensor(
                    np.array(mean_std["mean"], dtype=np.float32)
                ).reshape(1, -1),
                "std": torch.tensor(
                    np.array(mean_std["std"], dtype=np.float32)
                ).reshape(1, -1),
            }
        return mean_std_dict_new

    def __getitem__(self, idx: int):
        data_file = self.files[idx]

        npz_data = np.load(data_file.replace(".npy", ".npz"))

        num_points = int(npz_data["num_points"])

        center = torch.from_numpy(npz_data["center"]).to(torch.float32)
        flow_dir = torch.from_numpy(npz_data["flow_dir"]).to(torch.float32)
        flow_speed = torch.from_numpy(npz_data["flow_speed"]).to(torch.float32)
        air_density = torch.from_numpy(npz_data["air_density"]).to(torch.float32)
        frontal_area = torch.from_numpy(npz_data["frontal_area"]).to(torch.float32)

        npy_data = np.load(data_file)
        num_points_total = num_points          # [patch] 降采样前的真实点数
        if self.downsample_size > 0 and self.downsample_size < num_points:
            sampled_indices = np.sort(
                np.random.choice(num_points, self.downsample_size, replace=False)
            )
            npy_data = np.ascontiguousarray(npy_data[sampled_indices])
        else:
            npy_data = np.ascontiguousarray(npy_data)
        num_points = npy_data.shape[0]

        centroid = torch.from_numpy(npy_data["centroid"]).to(torch.float32)
        area = torch.from_numpy(npy_data["area"]).to(torch.float32)
        normal = torch.from_numpy(npy_data["normal"]).to(torch.float32)
        pressure = torch.from_numpy(npy_data["pressure"]).to(torch.float32)
        wss = torch.from_numpy(npy_data["wss"]).to(torch.float32)

        centroid = (centroid - center) / self.mean_std_dict["centroid"]["std"]
        pressure = (
            pressure - self.mean_std_dict["pressure"]["mean"]
        ) / self.mean_std_dict["pressure"]["std"]
        wss = (wss - self.mean_std_dict["wss"]["mean"]) / self.mean_std_dict["wss"][
            "std"
        ]

        inputs = {
            "centroid": centroid,  # (N, 3)
            "normal": normal,  # (N, 3)
            "area": area,  # (N, 1)
        }
        targets = {
            "pressure": pressure,  # (N, 1)
            "wss": wss,  # (N, 3)
        }
        others = {
            "file_path": data_file,
            "num_points_total": torch.tensor(num_points_total),   # [patch] 见 §7
            "mean_std_dict": self.mean_std_dict,
            "air_density": air_density,  # (,)
            "flow_dir": flow_dir,  # (,)
            "flow_speed": flow_speed,  # (,)
            "frontal_area": frontal_area,  # (,)
        }

        return inputs, targets, others

    def __len__(self):
        return len(self.files)


class PointCloudDataModule(BaseDataModule):
    def __init__(
        self,
        mean_std_dict: dict[str, dict[str, list[float] | float]],
        train_split_file: str = "",
        test_split_file: str = "",
        train_downsample_size: int = -1,
        test_downsample_size: int = -1,
    ):
        self.mean_std_dict = mean_std_dict

        super().__init__(
            train_split_file=train_split_file,
            test_split_file=test_split_file,
            train_downsample_size=train_downsample_size,
            test_downsample_size=test_downsample_size,
        )

    def _instantiate_dataset(
        self, files: list[str], downsample_size: int = -1
    ) -> torch.utils.data.Dataset:
        return PointCloudDataset(files, self.mean_std_dict, downsample_size)

    def _denormalize(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        others: dict[str, torch.Tensor],
    ):
        if "pressure" in outputs:
            outputs["pressure_denorm"] = (
                outputs["pressure"] * others["mean_std_dict"]["pressure"]["std"]
                + others["mean_std_dict"]["pressure"]["mean"]
            )
            targets["pressure_denorm"] = (
                targets["pressure"] * others["mean_std_dict"]["pressure"]["std"]
                + others["mean_std_dict"]["pressure"]["mean"]
            )

        if "wss" in outputs:
            outputs["wss_denorm"] = (
                outputs["wss"] * others["mean_std_dict"]["wss"]["std"]
                + others["mean_std_dict"]["wss"]["mean"]
            )
            targets["wss_denorm"] = (
                targets["wss"] * others["mean_std_dict"]["wss"]["std"]
                + others["mean_std_dict"]["wss"]["mean"]
            )
