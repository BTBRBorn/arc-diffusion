import os
import numpy as np
import random
import itertools
from pathlib import Path
from argparse import ArgumentParser
import json
from collections.abc import Iterable


# Helper Functions
def trim_array(array, *, left=0, right=0, up=0, down=0):
    down = None if down == 0 else -down
    right = None if right == 0 else -right
    up = None if up == 0 else up
    left = None if left == 0 else left
    return array[up:down, left:right]


def return_example(arc_func, *args, idx=0):
    gen = arc_func(*args)
    task = next(gen)
    input_array, output_array = task[idx]["input"], task[idx]["output"]
    return np.array(input_array), np.array(output_array)


def random_rectangle_zeros(min_size=2, max_size=8):
    n_rows, n_cols = (
        random.randint(min_size, max_size),
        random.randint(min_size, max_size),
    )
    return np.zeros(shape=(n_rows, n_cols))


def random_pad(array, max_padding):
    num_pad = random.randint(1, max_padding)
    out_array = np.pad(array, pad_width=num_pad)
    return out_array, num_pad


def random_roll(array, max_shift):
    x_shift, y_shift = (
        random.randint(-max_shift, max_shift),
        random.randint(-max_shift, max_shift),
    )
    array = np.roll(array, shift=(x_shift, y_shift), axis=(0, 1))
    return array, (x_shift, y_shift)


def random_location(n_rows, n_cols):
    return random.randint(0, n_rows - 1), random.randint(0, n_cols - 1)


def random_impute(array, ratio, values: tuple[int, ...] = (1,)):
    values_cycle = itertools.cycle(values)
    num_imps = int(array.size * ratio)
    num_imps = num_imps if num_imps else 1
    locations = []
    while len(locations) != num_imps:
        i, j = random_location(*array.shape)
        if (i, j) in locations:
            continue
        else:
            array[i, j] = next(values_cycle)
            locations.append((i, j))


def impute_diagonal(array, start_position: tuple[int, int], direction):
    if direction == "ne":
        sums = (-1, 1)
    elif direction == "sw":
        sums = (1, -1)
    elif direction == "nw":
        sums = (-1, -1)
    elif direction == "se":
        sums = (1, 1)
    else:
        raise ValueError
    i, j = start_position
    while True:
        try:
            i, j = i + sums[0], j + sums[1]
            if i < 0 or j < 0:
                break
            array[i, j] = 1
        except IndexError:
            break


# Arc task generator function template
def stub_func(num_repeat):
    while True:
        examples = []
        for _ in range(num_repeat):
            input_array = random_rectangle_zeros()
            output_array = random_rectangle_zeros()
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


### ARC task generators starts from this line ###
def rect_encap(num_repeat, max_rec_size=(8, 8), max_padding=5):
    while True:
        examples = []
        for _ in range(num_repeat):
            n_rows, n_cols = (
                random.randint(1, max_rec_size[0]),
                random.randint(1, max_rec_size[1]),
            )
            output_array = np.ones(shape=(n_rows, n_cols))
            # pad the output_array to get input array
            input_array, num_pad = random_pad(output_array, max_padding)
            # roll the output array
            input_array, shifts = random_roll(input_array, num_pad)
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


def rect_encap_v2(num_repeat, max_rec_size=(8, 8), max_padding=5):
    while True:
        examples = []
        for _ in range(num_repeat):
            n_rows, n_cols = (
                random.randint(1, max_rec_size[0]),
                random.randint(1, max_rec_size[1]),
            )
            # This one has random output array
            output_array = np.random.randint(low=1, high=10, size=(n_rows, n_cols))
            # pad the output_array to get input array
            input_array, num_pad = random_pad(output_array, max_padding)
            # roll the output array
            input_array, shifts = random_roll(input_array, num_pad)
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


def fill_rectangle(num_repeat):
    while True:
        examples = []
        for _ in range(num_repeat):
            zeros = random_rectangle_zeros()
            input_array = np.pad(zeros, pad_width=1, constant_values=1)
            output_array = np.where(input_array == 0, 2, input_array)
            input_array, num_pad = random_pad(input_array, max_padding=5)
            input_array, shifts = random_roll(input_array, max_shift=num_pad)
            output_array = np.pad(output_array, pad_width=num_pad)
            output_array = np.roll(output_array, shift=shifts, axis=(0, 1))
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


def vertical_lines(num_repeat):
    while True:
        examples = []
        for _ in range(num_repeat):
            input_array = random_rectangle_zeros(max_size=10)
            n_rows, n_cols = input_array.shape
            num_lines = random.randint(1, n_cols - 1)
            js = []
            for _ in range(num_lines):
                i, j = random_location(n_rows, n_cols)
                js.append(j)
                input_array[i, j] = 1
            output_array = np.copy(input_array)
            for j in js:
                output_array[:, j] = 1
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


def horizontal_lines(num_repeat):
    while True:
        examples = []
        for _ in range(num_repeat):
            input_array = random_rectangle_zeros(max_size=10)
            n_rows, n_cols = input_array.shape
            num_lines = random.randint(1, n_rows - 1)
            rows = []
            for _ in range(num_lines):
                i, j = random_location(n_rows, n_cols)
                rows.append(i)
                input_array[i, j] = 1
            output_array = np.copy(input_array)
            for i in rows:
                output_array[i, :] = 1
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


def draw_diagonal_1(num_repeat, min_size=4, max_size=20):
    while True:
        examples = []
        for _ in range(num_repeat):
            input_array = random_rectangle_zeros(min_size=min_size, max_size=max_size)
            ratio = 0.05
            num_changes = int(ratio * input_array.size)
            num_changes = num_changes if num_changes else 1
            locations = []
            for _ in range(num_changes):
                i, j = random_location(*input_array.shape)
                locations.append((i, j))
                input_array[i, j] = 1
            output_array = np.copy(input_array)
            for location in locations:
                impute_diagonal(output_array, location, "ne")
                impute_diagonal(output_array, location, "sw")
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


def draw_diagonal_2(num_repeat, min_size=4, max_size=20):
    while True:
        examples = []
        for _ in range(num_repeat):
            input_array = random_rectangle_zeros(min_size=min_size, max_size=max_size)
            ratio = 0.05
            num_changes = int(ratio * input_array.size)
            num_changes = num_changes if num_changes else 1
            locations = []
            for _ in range(num_changes):
                i, j = random_location(*input_array.shape)
                locations.append((i, j))
                input_array[i, j] = 1
            output_array = np.copy(input_array)
            for location in locations:
                impute_diagonal(output_array, location, "nw")
                impute_diagonal(output_array, location, "se")
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


def draw_xs(num_repeat, min_size=4, max_size=20):
    while True:
        examples = []
        for _ in range(num_repeat):
            input_array = random_rectangle_zeros(min_size=min_size, max_size=max_size)
            ratio = 0.05
            num_changes = int(ratio * input_array.size)
            num_changes = num_changes if num_changes else 1
            locations = []
            for _ in range(num_changes):
                i, j = random_location(*input_array.shape)
                locations.append((i, j))
                input_array[i, j] = 1
            output_array = np.copy(input_array)
            for location in locations:
                impute_diagonal(output_array, location, "ne")
                impute_diagonal(output_array, location, "sw")
                impute_diagonal(output_array, location, "nw")
                impute_diagonal(output_array, location, "se")
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


def color_mapping_columns(num_repeat, min_size=3, max_size=8):
    colors = list(range(1, 10))
    random.shuffle(colors)
    mappings = {colors.pop(): colors.pop() for i in range(4)}
    for k, v in tuple(mappings.items()):
        mappings[v] = k
    mappings[colors[-1]] = colors[-1]
    while True:
        examples = []
        for _ in range(num_repeat):
            input_array = random_rectangle_zeros(min_size, max_size)
            output_array = np.copy(input_array)
            n_cols = input_array.shape[1]
            for j in range(n_cols):
                input_color = random.randint(1, 9)
                input_array[:, j] = input_color
                output_array[:, j] = mappings[input_color]
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


def color_mapping_rows(num_repeat, min_size=3, max_size=8):
    colors = list(range(1, 10))
    random.shuffle(colors)
    mappings = {colors.pop(): colors.pop() for i in range(4)}
    for k, v in tuple(mappings.items()):
        mappings[v] = k
    mappings[colors[-1]] = colors[-1]
    while True:
        examples = []
        for _ in range(num_repeat):
            input_array = random_rectangle_zeros(min_size, max_size)
            output_array = np.copy(input_array)
            n_rows = input_array.shape[0]
            for i in range(n_rows):
                input_color = random.randint(1, 9)
                input_array[i, :] = input_color
                output_array[i, :] = mappings[input_color]
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


def color_mapping_dots(num_repeat, min_size=4, max_size=10):
    colors = list(range(1, 10))
    random.shuffle(colors)
    mappings = {colors.pop(): colors.pop() for i in range(4)}
    for k, v in tuple(mappings.items()):
        mappings[v] = k
    mappings[colors[-1]] = colors[-1]
    while True:
        examples = []
        for _ in range(num_repeat):
            input_array = random_rectangle_zeros(min_size, max_size)
            output_array = np.copy(input_array)
            n_dots = int(0.2 * input_array.size)
            n_dots = n_dots if n_dots else 1
            for i in range(n_dots):
                input_color = random.randint(1, 9)
                i, j = random_location(*input_array.shape)
                input_array[i, j] = input_color
                output_array[i, j] = mappings[input_color]
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


def copy_rectangle(num_repeat, min_size=2, max_size=8):
    while True:
        examples = []
        for _ in range(num_repeat):
            input_array = random_rectangle_zeros(min_size, max_size)
            n_rows, n_cols = input_array.shape
            output_array = np.concatenate((input_array,) * n_rows, axis=0)
            output_array = np.concatenate((output_array,) * n_cols, axis=1)
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


def copy_one_color(num_repeat, min_size=2, max_size=8):
    while True:
        examples = []
        for _ in range(num_repeat):
            input_array = random_rectangle_zeros(min_size, max_size)
            random_impute(input_array, ratio=random.random(), values=(1,))
            n_rows, n_cols = input_array.shape
            output_array = np.concatenate((input_array,) * n_rows, axis=0)
            output_array = np.concatenate((output_array,) * n_cols, axis=1)
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


def copy_multi_color(num_repeat, min_size=2, max_size=8):
    while True:
        examples = []
        for _ in range(num_repeat):
            input_array = random_rectangle_zeros(min_size, max_size)
            random_impute(input_array, ratio=random.random(), values=(1, 2, 3))
            n_rows, n_cols = input_array.shape
            output_array = np.concatenate((input_array,) * n_rows, axis=0)
            output_array = np.concatenate((output_array,) * n_cols, axis=1)
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


def flip_horizontal_one_color(num_repeat, min_size=4, max_size=20):
    while True:
        examples = []
        for _ in range(num_repeat):
            input_array = random_rectangle_zeros(min_size=min_size, max_size=max_size)
            random_impute(input_array, ratio=random.uniform(0.2, 0.8))
            output_array = np.flip(input_array, axis=0)
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


def flip_vertical_one_color(num_repeat, min_size=4, max_size=20):
    while True:
        examples = []
        for _ in range(num_repeat):
            input_array = random_rectangle_zeros(min_size=min_size, max_size=max_size)
            random_impute(input_array, ratio=random.uniform(0.2, 0.8))
            output_array = np.flip(input_array, axis=1)
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


def flip_horizontal_multi_color(num_repeat, min_size=4, max_size=20):
    while True:
        examples = []
        for _ in range(num_repeat):
            input_array = random_rectangle_zeros(min_size=min_size, max_size=max_size)
            random_impute(input_array, ratio=random.uniform(0.2, 0.8), values=(1, 2, 3))
            output_array = np.flip(input_array, axis=0)
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


def flip_vertical_multi_color(num_repeat, min_size=4, max_size=20):
    while True:
        examples = []
        for _ in range(num_repeat):
            input_array = random_rectangle_zeros(min_size=min_size, max_size=max_size)
            random_impute(input_array, ratio=random.uniform(0.2, 0.8), values=(1, 2, 3))
            output_array = np.flip(input_array, axis=1)
            examples.append(
                {"input": input_array.tolist(), "output": output_array.tolist()}
            )
        yield examples


class ArcGenerator:
    def __init__(self, num_repeat: int | Iterable[int], output_path: Path, verbose=False):
        self.verbose = verbose
        self.output_path = Path(output_path)
        self.task_generators = [
            rect_encap(num_repeat),
            rect_encap_v2(num_repeat),
            fill_rectangle(num_repeat),
            vertical_lines(num_repeat),
            horizontal_lines(num_repeat),
            draw_diagonal_1(num_repeat),
            draw_diagonal_2(num_repeat),
            draw_xs(num_repeat),
            color_mapping_columns(num_repeat),
            color_mapping_rows(num_repeat),
            color_mapping_dots(num_repeat),
            copy_rectangle(num_repeat),
            copy_one_color(num_repeat),
            copy_multi_color(num_repeat),
            flip_vertical_one_color(num_repeat),
            flip_vertical_multi_color(num_repeat),
            flip_horizontal_one_color(num_repeat),
            flip_horizontal_multi_color(num_repeat),
        ]
        if verbose:
            print(f"{len(self.task_generators)} generators initiated.")

    def __call__(self):
        for i, task_generator in enumerate(self.task_generators, start=1):
            task = next(task_generator)
            file_path = self.output_path / f"task_{i}.json"
            file_path.write_text(json.dumps(task))
            if self.verbose:
                print(f"File {file_path} is created.")


if __name__ == "__main__":
    arg_parser = ArgumentParser()

    arg_parser.add_argument("--output_folder", type=str)
    arg_parser.add_argument("--examples_per_task", type=int, default=100)
    arg_parser.add_argument("--verbose", type=str, choices={0, 1}, default=1)

    args = arg_parser.parse_args()

    os.makedirs(args.output_folder)

    arc_generator = ArcGenerator(args.examples_per_task, args.output_folder, verbose=args.verbose)
    arc_generator()
