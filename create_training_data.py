import argparse
from pathlib import Path
import random
import json
import pickle
import shutil
import numpy as np
import functools
from concurrent import futures
import multiprocessing as mp
import os

from get_tokenizer import Tokenizer
from get_augmentor import Augmentor


def get_tasks(data_path):
    for cur_dir_path, _, sub_files in os.walk(data_path):
        for file in sub_files:
            if file.endswith(".json"):
                file_path = Path(cur_dir_path) / file
            else:
                continue
            task = json.loads(file_path.read_text())
            if (
                isinstance(task, list) and "input" in task[0].keys()
            ) or "train" in task.keys():
                yield task
                

def process_syn_task(task, tokenizer: Tokenizer, augmentor: Augmentor = None):
    if augmentor is not None:
        random.shuffle(task)
        augmentor(task)
    task = tokenizer.encode(task)
    np_task = np.array(task, dtype=np.uint8)
    return np_task


def save_syn_shard(data, shard_size, output_path, start_index):
    random.shuffle(data)
    np_data = np.concatenate(data)
    num_shards = len(np_data) // shard_size
    shards = np.array_split(np_data, num_shards)
    for i, shard in enumerate(shards, start=start_index):
        file_path = output_path / f"training_{i}.npy"
        print(f"File {file_path} is created.")
        np.save(file_path, shard)
    last_index = i
    return last_index


def create_syn_data(
    output_path,
    source_path,
    tokenizer,
    shard_size,
    num_aug,
    num_workers,
):
    data_path = Path(source_path)
    output_path = Path(output_path)
    augmentor = Augmentor()

    # Without Augmentation
    process_without_aug = functools.partial(process_syn_task, tokenizer=tokenizer)
    tasks = get_tasks(data_path)
    with mp.Pool(processes=num_workers) as executor:
        data = list(executor.map(process_without_aug, tasks))
    last_index = save_syn_shard(data, shard_size, output_path, start_index=1)

    # With Augmentation
    process_with_aug = functools.partial(
        process_syn_task, tokenizer=tokenizer, augmentor=augmentor
    )
    for _ in range(num_aug):
        tasks = get_tasks(data_path)
        with mp.Pool(processes=num_workers) as executor:
            data = list(executor.map(process_with_aug, tasks))
        last_index = save_syn_shard(data, shard_size, output_path, last_index + 1)
    return last_index


def create_data(
    output_file_path,
    source_path,
    tokenizer,
    is_train,
    rolled,
    augmented,
    num_repeat,
):
    data_path = Path(source_path)
    output_file_path = Path(output_file_path)
    augmentor = Augmentor()

    data = []
    for _ in range(num_repeat):
        tasks = get_tasks(data_path)
        for task in tasks:
            if is_train:
                task = task["train"]
                random.shuffle(task)
            else:
                task = task["test"]
            if augmented:
                augmentor(task)  # In-place change
            task = tokenizer.encode(task)
            np_task = np.array(task, dtype=np.uint8)
            data.append(np_task)

    if is_train:
        random.shuffle(data)
    data = np.concatenate(data)

    if rolled:
        data = np.roll(data, shift=random.randint(0, 50000))

    np.save(output_file_path, data)

    return output_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--syn_data_path", type=str, default="data/re_arc/tasks")
    parser.add_argument("--syn_shard_size", type=int, default=int(1e7))
    parser.add_argument("--syn_num_aug", type=int, default=1)
    parser.add_argument("--train_data_path", type=str, default="data/ARC-AGI-2/data")
    parser.add_argument(
        "--val_data_path", type=str, default="data/ARC-AGI-2/data/training"
    )
    parser.add_argument("--processed_data_path", type=str, default="data/pretraining")
    parser.add_argument("--aug_num_shards", type=int, default=10)
    parser.add_argument("--aug_num_repeat_per_shard", type=int, default=1)
    parser.add_argument("--vocab_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--tokenizer_save_path", type=str, default="")
    parser.add_argument("--only_validation_data", type=int, choices={0, 1}, default=0)

    args = parser.parse_args()
    tokenizer = Tokenizer(args.vocab_size)

    PROCESSED_DATA_PATH = Path(args.processed_data_path)
    SYN_DATA_PATH = Path(args.syn_data_path)
    TRAIN_DATA_PATH = Path(args.train_data_path)
    VAL_DATA_PATH = Path(args.val_data_path)

    if not args.only_validation_data:
        if PROCESSED_DATA_PATH.exists():
            shutil.rmtree(PROCESSED_DATA_PATH)
        PROCESSED_DATA_PATH.mkdir(parents=True)

        print("Training data is being created.")
        last_shard_num = create_syn_data(
            output_path=PROCESSED_DATA_PATH,
            source_path=SYN_DATA_PATH,
            tokenizer=tokenizer,
            shard_size=args.syn_shard_size,
            num_aug=args.syn_num_aug,
            num_workers=args.num_workers,
        )
        print("Finished with creating synthetic data.")

        train_create_data = functools.partial(
            create_data,
            source_path=TRAIN_DATA_PATH,
            tokenizer=tokenizer,
            is_train=True,
            rolled=True,
            augmented=True,
            num_repeat=args.aug_num_repeat_per_shard,
        )

        training_file_paths = [
            PROCESSED_DATA_PATH / f"training_{i}.npy"
            for i in range(last_shard_num + 1, args.aug_num_shards + last_shard_num + 1)
        ]
        with futures.ProcessPoolExecutor(args.num_workers) as executor:
            for outfile_path in executor.map(train_create_data, training_file_paths):
                print(f"File {outfile_path} is created.")
        print("Finished with creating augmented competition data.")

    # Validation data is being created
    create_data(
        source_path=VAL_DATA_PATH,
        tokenizer=tokenizer,
        output_file_path=PROCESSED_DATA_PATH / "validation_1.npy",
        is_train=False,
        rolled=False,
        augmented=False,
        num_repeat=1,
    )

    if args.tokenizer_save_path:
        tokenizer.train(
            data_path=PROCESSED_DATA_PATH,
            num_workers=args.num_workers,
        )

        tokenizer_save_path = Path(args.tokenizer_save_path)
        if not tokenizer_save_path.parent.exists():
            tokenizer_save_path.parent.mkdir(parents=True)

        with open(tokenizer_save_path, "wb") as fhandle:
            pickle.dump(tokenizer, fhandle)
