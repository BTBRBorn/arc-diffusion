from collections import Counter
from collections.abc import Iterable
import multiprocessing as mp
import numpy as np
from pathlib import Path
import os
import functools
import time
from collections.abc import Container


def get_pair_stats(tokens: Iterable[int]):
    return Counter((int(i), int(j)) for i, j in zip(tokens[:-1], tokens[1:]))


def get_file_stats(file_path):
    tokens = np.load(file_path)
    return get_pair_stats(tokens)


def get_total_stats(data_path: Path, num_workers) -> Counter[tuple, int]:
    total_counter = Counter()
    filelist = os.listdir(data_path)
    with mp.Pool(num_workers) as pool:
        for ct in pool.imap(get_file_stats, (data_path / file for file in filelist)):
            total_counter += ct
    return total_counter


def merge(tokens: Iterable[np.uint8], pair: tuple[int, int], replace_token: int):
    new_tokens = []
    index = 0
    while len(tokens) >= 2:
        if (
            index < len(tokens) - 1
            and int(tokens[index]) == pair[0]
            and int(tokens[index + 1]) == pair[1]
        ):
            new_tokens.append(replace_token)
            index += 2
        elif index != len(tokens):
            new_tokens.append(tokens[index])
            index += 1
        else:
            break
    return new_tokens


def merge_save_file(file_path: Path, pair: tuple[int, int], replace_token: int):
    tokens = np.load(file_path)
    tokens = merge(tokens, pair, replace_token)
    tokens = np.array(tokens, dtype=np.uint8)
    np.save(file_path, tokens)


def merge_save_files(
    data_path: Path, pair: tuple[int, int], replace_token: int, num_workers
):
    filelist = os.listdir(data_path)
    with mp.Pool(num_workers) as pool:
        for _ in pool.imap(
            functools.partial(merge_save_file, pair=pair, replace_token=replace_token),
            [data_path / file for file in filelist],
        ):
            continue


class Tokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

        self.special_tokens = {
            "start_of_input": None,
            "end_of_input": None,
            "start_of_output": None,
            "end_of_output": None,
            "row_indicator": None,
            "context_indicator": None,
        }

        last_token = vocab_size - len(self.special_tokens)
        for key in self.special_tokens.keys():
            self.special_tokens[key] = last_token
            last_token += 1

        self.merges = {}
        self.vocab = {i: i for i in range(10)}
        for token in self.special_tokens.values():
            self.vocab[token] = token

    def _flatten(self, array, with_rows=False):
        is_nested = all([isinstance(x, Container) for x in array])
        if not is_nested:
            return list(array)
        flat_l = []
        if with_rows:
            for i, e in enumerate(array):
                if i != len(array) - 1:
                    flat_l.extend(e + [self.special_tokens["row_indicator"]])
                else:
                    flat_l.extend(e)
        else:
            for e in array:
                flat_l.extend(e)
        return flat_l

    def encode(self, array: list[dict]):
        data = self._flatten(
            [
                [self.special_tokens["start_of_input"]]
                + self._flatten(e["input"], with_rows=True)
                + [self.special_tokens["end_of_input"]]
                + [self.special_tokens["start_of_output"]]
                + self._flatten(e["output"], with_rows=True)
                + [self.special_tokens["end_of_output"]]
                for e in array
            ]
        )

        data = [self.special_tokens["context_indicator"]] + data

        while True:
            ct = get_pair_stats(data)
            pair = min(ct, key=lambda x: self.merges.get(x, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            data = merge(data, pair, idx)

        return data

    def decode(self, comp_tokens: list[int], only_last_output=False):
        tokens = []
        for token in comp_tokens:
            value = self.vocab[token]
            if isinstance(value, Container):
                tokens.extend(value)
            else:
                tokens.append(value)

        if only_last_output:

            for idx, token in enumerate(tokens[::-1], start=1):
                if token == self.special_tokens["start_of_output"]:
                    output_index = idx
                    break
            tokens = tokens[-output_index:]

        examples = []
        context = None
        for token in tokens:
            if token == self.special_tokens["start_of_input"]:
                example = {"input": []}
                row = []
                context = "input"
            elif token == self.special_tokens["end_of_input"]:
                example["input"].append(row)
            elif token == self.special_tokens["start_of_output"]:
                try:
                    example["output"] = []
                except UnboundLocalError:
                    example = {"output": []}
                row = []
                context = "output"
            elif token == self.special_tokens["end_of_output"]:
                example["output"].append(row)
                examples.append(example)
            elif token == self.special_tokens["row_indicator"]:
                example[context].append(row)
                row = []
            elif token == self.special_tokens["context_indicator"]:
                continue
            else:
                row.append(token)

        return examples

    def train(self, data_path: Path, num_workers=8, verbose=True):
        replace_token = 10
        while replace_token != self.special_tokens["start_of_input"]:
            start = time.perf_counter()
            ct = get_total_stats(data_path, num_workers)
            pair = ct.most_common(1)[0][0]
            self.merges[pair] = replace_token
            merge_save_files(data_path, pair, replace_token, num_workers)
            end = time.perf_counter()
            if verbose:
                print(
                    f"Pair {pair} is replaced with {replace_token} in {end - start:.4f} seconds"
                )
            replace_token += 1

        for pair, idx in self.merges.items():
            v1, v2 = self.vocab[pair[0]], self.vocab[pair[1]]
            self.vocab[idx] = (
                (v1 if isinstance(v1, Container) else (v1,)) + 
                (v2 if isinstance(v2, Container) else (v2,))
            )
