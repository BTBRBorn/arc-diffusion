# Multi-gpu inference script
# This script is setup to run with torchrun

import torch
import torch.nn.functional as F
from copy import deepcopy
from utils import load_checkpoint
import os
from pathlib import Path
import json
import argparse
from collections import deque
import math
import random

# Modules are needed for multi-gpu inference
import torch.distributed as dist


class Evaluator:
    def __init__(self, path_to_checkpoint, task_paths, k_beam, device):
        self.path_to_checkpoint = Path(path_to_checkpoint)
        self.task_paths = task_paths
        self.k_beam = k_beam
        self.device = device
        self.checkpoint = load_checkpoint(
            path_to_checkpoint,
            device,
            compile_model=False,
        )

    def _create_context(self, task, test_index, tokenizer):
        new_task = {}
        new_task["context"] = deepcopy(task["train"])
        test_input = deepcopy(
            {"input": task["test"][test_index]["input"], "output": [[]]}
        )
        new_task["context"].append(test_input)
        tokens = tokenizer.encode(new_task["context"])
        return tokens[:-1], len(tokens[:-1])

    @staticmethod
    def _bfs(
        model,
        context: torch.Tensor,
        config,
        tokenizer,
        tokens_threshold,
        prob_threshold=1.0,
    ):
        contexts = deque([(context, 0.0, 0)])
        solutions = []
        with torch.inference_mode():
            while contexts:
                context, score, counter = contexts.popleft()
                if counter > tokens_threshold:
                    continue
                context = context[:, -config.block_size :]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(context)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                # Bundle next_tokens and their probability together
                probs = probs.view(-1)
                mask = probs > prob_threshold
                tokens = mask.nonzero(as_tuple=True)[0]
                if len(tokens) == 0:
                    tokens = torch.argmax(probs).unsqueeze(dim=0)
                    mask = tokens
                next_tokens = zip(tokens, probs[mask])
                for next_token, prob in next_tokens:
                    next_context = torch.cat((context, next_token.view(1, -1)), dim=-1)
                    new_score = score + math.log(prob.item())
                    new_counter = counter + 1
                    if next_token.item() != tokenizer.special_tokens["end_of_output"]:
                        contexts.append((next_context, new_score, new_counter))
                    else:
                        solution = tokenizer.decode(
                            next_context.tolist()[0], only_last_output=True
                        )[0]["output"]
                        new_score /= len(solution)
                        solutions.append((solution, new_score))
        return solutions

    @staticmethod
    def _beam_search(
        model: torch.nn.Module,
        context: torch.Tensor,
        config,
        tokenizer,
        tokens_threshold,
        k_beam,
    ):
        # canditates = [(context, score, is_finished)]
        beams = [(context, 0.0, False)]
        end_of_output = tokenizer.special_tokens["end_of_output"]
        with torch.inference_mode():
            for _ in range(tokens_threshold):
                candidates = []
                not_finished_seqs = [seq for seq, _, finished in beams if not finished]
                # If every beam ends with end_of_output token break early
                if len(not_finished_seqs) == 0:
                    break
                context = torch.cat(not_finished_seqs, dim=0)
                context = context[:, -config.block_size :]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(context)[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                top_log_probs, top_tokens = torch.topk(log_probs, k=k_beam, dim=-1)

                i = 0
                for seq, score, _ in beams:
                    if seq[0, -1].item() == end_of_output:
                        candidates.append((seq, score, True))
                        continue

                    for log_prob, token in zip(top_log_probs[i], top_tokens[i]):
                        next_seq = torch.cat((seq, token.view(1, 1)), dim=-1)
                        next_score = score + log_prob.item()
                        candidates.append(
                            (
                                next_seq,
                                next_score,
                                True if token.item() == end_of_output else False,
                            )
                        )
                    i += 1

                beams = sorted(
                    candidates,
                    key=lambda x: x[1] / x[0].shape[1],
                    reverse=True,
                )[:k_beam]

        solutions = []
        for seq, score, finished in beams:
            if finished:
                try:
                    solution = tokenizer.decode(seq.tolist()[0], only_last_output=True)[
                        0
                    ]["output"]
                    norm_score = score / seq.shape[1]
                    solutions.append((solution, norm_score))
                except KeyError as err:
                    print("KeyError:", err)
                    continue

        return solutions

    def _generate_solutions(
        self,
        model,
        task,
        test_index,
        tokens_threshold=2000,
    ):
        tokenizer = self.checkpoint["tokenizer"]
        config = self.checkpoint["config"]
        context, con_len = self._create_context(task, test_index, tokenizer)
        context = torch.tensor(context, device=self.device).view(1, -1)
        solutions = self._beam_search(
            model,
            context,
            config,
            tokenizer,
            tokens_threshold,
            self.k_beam,
        )

        if len(solutions) == 0:
            return None, con_len
        else:
            return [
                s[0] for s in sorted(solutions, key=lambda x: x[1], reverse=True)[:2]
            ], con_len

    def _check_solution(self, output, solution):
        if solution is None:
            return False
        return output == solution

    def _is_2d_array(self, array):
        lengths = set([len(e) for e in array])
        return len(lengths) == 1

    def _check_pixel_values(self, output, solution):
        if solution is None:
            return 0.0
        if self._is_2d_array(solution) and (
            len(output) == len(solution) and len(output[0]) == len(solution[0])
        ):
            total_pixels = len(output) * len(output[0])
            matched_pixels = 0
            for i in range(len(output)):
                for j in range(len(output[0])):
                    if output[i][j] == solution[i][j]:
                        matched_pixels += 1
            return matched_pixels / total_pixels
        else:
            return 0.0

    def evaluate(self, verbose=False):
        model = self.checkpoint["model"]
        task_acc, pixel_acc = [], []
        total_tasks = len(self.task_paths)
        for task_number, task_path in enumerate(self.task_paths, start=1):
            with open(task_path, "r") as fhandle:
                task = json.load(fhandle)

            for tx in range(len(task["test"])):
                output = task["test"][tx]["output"]

                solutions, con_len = self._generate_solutions(model, task, tx)
                acc = (
                    max(self._check_solution(output, s) for s in solutions)
                    if solutions is not None
                    else 0.0
                )
                p_acc = (
                    max(self._check_pixel_values(output, s) for s in solutions)
                    if solutions is not None
                    else 0.0
                )
                task_acc.append(acc)
                pixel_acc.append(p_acc)
                if verbose:
                    print(
                        f"Task({task_path.name}) {task_number!s:>3s}/{total_tasks}: test {tx + 1}, "
                        + f"context length: {con_len!s:>4s}, task solved: {task_acc[-1]}, "
                        + f"pixel accuracy percentage: {pixel_acc[-1] * 100:.2f}%"
                    )

        task_acc_sum = torch.tensor(
            sum(task_acc), device=self.device, dtype=torch.float32
        )
        pixel_acc_sum = torch.tensor(
            sum(pixel_acc), device=self.device, dtype=torch.float32
        )

        return task_acc_sum, pixel_acc_sum


def get_balanced_filelists(all_tasks: list, numsplits: int):
    files_with_size = []
    total_tasks = 0
    for file_path in all_tasks:
        task = json.loads(file_path.read_text())
        num_tests = len(task["test"])
        total_tasks += num_tests
        file_size = os.path.getsize(file_path)
        files_with_size.append((file_path, file_size * num_tests))
    files_with_size.sort(key=lambda x: x[1], reverse=True)
    gpu_bins = [[] for _ in range(numsplits)]
    gpu_sizes = [0 for _ in range(numsplits)]
    for file_path, size in files_with_size:
        idx = min(range(len(gpu_sizes)), key=lambda i: gpu_sizes[i])
        gpu_bins[idx].append(file_path)
        gpu_sizes[idx] += size

    for bin in gpu_bins:
        random.shuffle(bin)

    return gpu_bins, gpu_sizes, total_tasks


if __name__ == "__main__":
    random.seed(1)

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--tasks_path", type=str)
    parser.add_argument("--verbose", type=int, choices={0, 1}, default=1)
    parser.add_argument("--k_beam", type=int, default=1)

    args = parser.parse_args()

    dist.init_process_group(backend="nccl")

    # torchrun will handle setting up environment variables
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    master_process = rank == 0
    torch.cuda.set_device(device)

    data_path = Path(args.tasks_path)
    all_tasks = [data_path / file for file in os.listdir(data_path)]

    gpu_bins, gpu_sizes, total_tasks = get_balanced_filelists(all_tasks, world_size)
    task_paths = gpu_bins[rank]
    if master_process:
        for rank in range(world_size):
            print(
                f"Rank {rank} gpu will be processing: {gpu_sizes[rank] / 1e6: .2f} MBs."
            )

    evaluator = Evaluator(
        path_to_checkpoint=args.checkpoint_path,
        task_paths=task_paths,
        k_beam=args.k_beam,
        device=device,
    )
    task_acc_sum, pixel_acc_sum = evaluator.evaluate(verbose=bool(args.verbose))

    dist.reduce(task_acc_sum, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(pixel_acc_sum, dst=0, op=dist.ReduceOp.SUM)

    if master_process:
        task_acc_avg = task_acc_sum.item() / total_tasks
        pixel_acc_avg = pixel_acc_sum.item() / total_tasks
        print(
            f"Overall accuracy: {100 * task_acc_avg:.2f}%, "
            + f"Overall pixel accuracy: {100 * pixel_acc_avg:.2f}%"
        )

    dist.destroy_process_group()
