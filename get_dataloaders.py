from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import torch
import numpy as np
import os


class CustomDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        block_size: int,
        is_train: bool,
        mask_token: int,
        mask_min_max_train: tuple = (0.01, 0.8),
        mask_only_end: float = 0.1,
    ):
        self.block_size = block_size
        self.data_path = data_path
        self.mask_token = mask_token
        if is_train:
            self.filelist = [
                file for file in os.listdir(data_path) if "training" in file
            ]
            self.mask_min_max = mask_min_max_train
            self.mask_only_end = mask_only_end
        else:
            self.filelist = [
                file for file in os.listdir(data_path) if "validation" in file
            ]
            # We don't want a high variance between validation data
            # So mask_min_max will have much less range than training process
            self.mask_min_max = (0.1, 0.3)
            self.mask_only_end = 0.5

        self.meta_data = {}
        self.populate_meta_data()

    def populate_meta_data(self):
        for file in self.filelist:
            file_path = self.data_path / file
            data = np.load(file_path, mmap_mode="r")
            num_shard = int(str(file_path).split("_")[-1].split(".")[0])
            self.meta_data[num_shard] = {"file_path": file_path}
            self.meta_data[num_shard]["num_tokens"] = len(data)

        total_blocks = 0
        for num_shard in sorted(self.meta_data.keys()):
            num_blocks = self.get_num_blocks(num_shard)
            self.meta_data[num_shard]["start_index"] = total_blocks
            self.meta_data[num_shard]["end_index"] = total_blocks + num_blocks - 1
            total_blocks += num_blocks

    def get_num_blocks(self, num_shard):
        num_blocks = self.meta_data[num_shard]["num_tokens"] // self.block_size
        if self.meta_data[num_shard]["num_tokens"] % self.block_size:
            return num_blocks + 1
        else:
            return num_blocks

    def __len__(self):
        last_key = max(self.meta_data.keys())
        return self.meta_data[last_key]["end_index"] + 1

    def __getitem__(self, index):
        num_shard = 1
        if index == -1:
            index = len(self) - 1
        try:
            while not (
                self.meta_data[num_shard]["start_index"]
                <= index
                <= self.meta_data[num_shard]["end_index"]
            ):
                num_shard += 1
        except KeyError:
            raise IndexError(f"{self.__class__.__name__} index out of range")

        data = np.load(self.meta_data[num_shard]["file_path"], mmap_mode="r")
        norm_index = index - self.meta_data[num_shard]["start_index"]

        y = data[norm_index * self.block_size : (norm_index + 1) * self.block_size]

        if len(y) < self.block_size:
            y = data[-self.block_size :]

        y = torch.tensor(y, dtype=torch.long)

        min_mask, max_mask = self.mask_min_max
        mask_proportion = torch.rand(1).item() * (max_mask - min_mask) + min_mask
        num_masks = int(mask_proportion * self.block_size)

        if torch.rand(1).item() <= self.mask_only_end:
            ids = torch.arange(self.block_size)[-num_masks:]
        else:
            ids = torch.randperm(self.block_size)[:num_masks]

        x = y.clone()
        x[ids] = self.mask_token

        return x, y


def create_dataloaders(config, data_path, mask_token, train_shuffle=True):
    data_path = Path(data_path)

    train_dataset = CustomDataset(
        data_path,
        config.block_size,
        is_train=True,
        mask_token=mask_token,
    )
    val_dataset = CustomDataset(
        data_path,
        config.block_size,
        is_train=False,
        mask_token=mask_token,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=train_shuffle)
    val_sampler = DistributedSampler(val_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        config.batch_size,
        num_workers=config.dataloader_num_workers,
        pin_memory=True,
        shuffle=False,
        sampler=train_sampler,
    )

    val_dataloader = DataLoader(
        val_dataset,
        config.batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        pin_memory=True,
        sampler=val_sampler,
    )

    return train_dataloader, val_dataloader, train_sampler
