import json
from pathlib import Path

import torch
import torch.utils.data


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, files: list[str], downsample_size: int = -1):
        self.files = files
        self.downsample_size = downsample_size


class BaseDataModule:
    def __init__(
        self,
        train_split_file: str = "",
        test_split_file: str = "",
        train_downsample_size: int = -1,
        test_downsample_size: int = -1,
    ):
        train_files = self._get_files(train_split_file)
        test_files = self._get_files(test_split_file)

        self.train_dataset = (
            self._instantiate_dataset(train_files, train_downsample_size)
            if len(train_files) > 0
            else None
        )

        self.test_dataset = (
            self._instantiate_dataset(test_files, test_downsample_size)
            if len(test_files) > 0
            else None
        )

    def _instantiate_dataset(
        self, files: list[str], downsample_size: int = -1
    ) -> torch.utils.data.Dataset:
        raise NotImplementedError

    def train_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(self.train_dataset, *args, **kwargs)

    def test_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(self.test_dataset, *args, **kwargs)

    def _get_files(self, split_file: str) -> list[str]:
        files = []
        if split_file and Path(split_file).exists():
            with open(split_file, "r") as f:
                files = json.load(f)
        return files

    def _denormalize(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        others: dict[str, torch.Tensor],
    ):
        raise NotImplementedError(
            "Denormalization method not implemented in current datamodule"
        )
