from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import torch
import numpy as np
import os

def mask_tokens(x: torch.Tensor, mask_range: tuple, mask_token: int):
        tensor_length = x.size(0)
        min_mask, max_mask = mask_range
        mask_ratio = torch.rand(1).item() * (max_mask - min_mask) + min_mask
        num_masks = int(mask_ratio * tensor_length)
        mask_ids = torch.randperm(tensor_length)[:num_masks]
        x[mask_ids] = mask_token
        return x.clone()

def add_noise(x: torch.Tensor, noise_range: tuple, mask_token: int) -> torch.Tensor:
    min_noise, max_noise = noise_range
    assert 0 <= min_noise < max_noise <= 1.0
    noise_ratio = torch.rand(1).item() * (max_noise - min_noise) + min_noise
    candidate_indices = (x != mask_token).nonzero(as_tuple=True)[0]
    num_candidates = candidate_indices.numel()
    num_noise = int(noise_ratio * num_candidates)
    noise_indices = candidate_indices[torch.randperm(num_candidates)[:num_noise]]
    if num_noise > 0:
        #Make sure mask_token is the largest token in the tokenizer
        noise = torch.randint(0, mask_token, size=(num_noise, ))
        x[noise_indices] = noise
    return x.clone()

class CustomDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        block_size: int,
        is_train: bool,
        mask_token: int,
        mask_range_train: tuple = (0.01, 0.8),
        noise_range_train: float = (0.0, 0.3),
    ):
        self.block_size = block_size
        self.data_path = data_path
        self.mask_token = mask_token
        self.is_train = is_train
        if is_train:
            self.filelist = [
                file for file in os.listdir(data_path) if "training" in file
            ]
            self.mask_range = mask_range_train
            self.noise_range = noise_range_train
        else:
            self.filelist = [
                file for file in os.listdir(data_path) if "validation" in file
            ]
            # We don't want a high variance between validation data
            # So mask_range_ will have much less range than training process
            self.mask_range = (0.25, 0.3)

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
        x = y.clone()

        #Mask some of the tokens
        x = mask_tokens(x, self.mask_range, self.mask_token)

        #Add noise to the training data
        if self.is_train:
            x = add_noise(x, self.noise_range, self.mask_token)

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
