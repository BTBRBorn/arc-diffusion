import random
import numpy as np


class Augmentor:
    def __init__(self):
        self.colors = list(range(0, 10))

    def _flatten(self, array):
        flat_l = []
        for e in array:
            flat_l.extend(e)
        return flat_l

    def _get_mappings(self, task):
        flattened = []
        for example in task:
            flattened.extend(
                self._flatten(example["input"]) + self._flatten(example["output"])
            )
        color_set = set(flattened)
        mappings = {}
        copy_colors = list(self.colors)
        for c in color_set:
            new_c = random.choice(copy_colors)
            mappings[c] = new_c
            copy_colors.remove(new_c)
        return mappings

    def _change_array(self, array, mappings):
        n_rows, n_columns = len(array), len(array[0])
        for i in range(n_rows):
            for j in range(n_columns):
                array[i][j] = mappings[array[i][j]]

    def _change_one_example(self, example, mappings):
        self._change_array(example["input"], mappings)
        self._change_array(example["output"], mappings)

    def _change_colors(self, task: list[dict]):
        mappings = self._get_mappings(task)
        for example in task:
            self._change_one_example(example, mappings)

    def _rotate90_array(self, array, k):
        np_array = np.array(array)
        return np.rot90(np_array, k=k).tolist()

    def _rotate90_example(self, example, k):
        example["input"] = self._rotate90_array(example["input"], k=k)
        example["output"] = self._rotate90_array(example["output"], k=k)

    def _rotate90(self, task):
        k = random.randint(1, 4)
        for example in task:
            self._rotate90_example(example, k=k)

    def _reflect_array(self, array, axis):
        np_array = np.array(array)
        return np.flip(np_array, axis=axis).tolist()

    def _reflect_example(self, example, axis):
        example["input"] = self._reflect_array(example["input"], axis=axis)
        example["output"] = self._reflect_array(example["output"], axis=axis)

    def _reflect_task(self, task):
        axis = random.choice((-100, 0, 1, (0, 1)))
        # If axis == -100 don't do anything
        if axis == -100:
            return None
        for example in task:
            self._reflect_example(example, axis=axis)

    def __call__(self, task, change_colors=True, rotate90=True, reflect_task=True):
        if change_colors:
            self._change_colors(task)
        if rotate90:
            self._rotate90(task)
        if reflect_task:
            self._reflect_task(task)
