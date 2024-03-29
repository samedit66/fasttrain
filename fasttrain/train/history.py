from collections import defaultdict
import warnings
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


def _add_file_ext(self, file_name: str, ext: str) -> str:
    if not ext.startswith("."):
        ext = f".{ext}"
    return file_name if file_name.endswith(ext) else f"{file_name}{ext}"


class History():
    r'''
    Dict-like object, used to store metrics' values during model training.
    Can be accesed by key (metric name) to get the metric's values over epochs, for example:

    ```python
    history = History()
    history.update({'accuracy': 0.96, 'loss': 1.241})
    history.update({'accuracy': 0.97, 'loss': 1.013})
    history.update({'accuracy': 0.98, 'loss': 0.958})
    # prints "[0.96, 0.97, 0.98]"
    print(history['accuracy'])
    ```

    To get the average value of a specified metric, you can use `average_of`:
    ```python
    # prints "1.0706"
    print(history.average_of('loss'))
    ```

    To get the average value of all metrics, you can use the property `average`:
    ```python
    # prints "{'accuracy': 0.97, 'loss': 1.0706}"
    print(history.average)
    ```

    To visualize a metric, you can call either `plot` or its alias `visualize`:
    ```python
    history.plot('accuracy')
    ```
    '''

    def __init__(self, initial_stats=None):
        initial_stats = {} if initial_stats is None else initial_stats
        self._stats_history = defaultdict(list, initial_stats)

    @classmethod
    def from_json(cls, file_name):
        with open(_add_file_ext(file_name, "json"), "r") as input_file:
            stats_history = json.load(input_file)
        return History(stats_history)
 
    def to_json(self, file_name, indent_level=4):
        with(open(_add_file_ext(file_name, "json"), "r")) as output_file:
            json.dump(self.history, output_file, indent=indent_level)

    def update(self, stats):
        for k, v in stats.items():
            if not isinstance(v, float):
                v = float(v)
            self._stats_history[k].append(v)

    def __iter__(self):
        return iter(self._stats_history)

    def __getitem__(self, key):
        if not key in self._stats_history:
            raise KeyError(key)
        return self._stats_history[key]

    def items(self):
        return self._stats_history.items()

    def keys(self):
        return self._stats_history.keys()

    @property
    def average(self):
        average_stats = {}
        for k, v in self._stats_history.items():
            average_stats[k] = sum(v) / len(v)
        return average_stats

    def average_of(self, metric_name):
        return self.average[metric_name]

    @classmethod
    def mean(cls, *histories):
        if not histories:
            raise ValueError("Cannot compute mean of no histories")
        
        stats = {}
        for key in histories[0].keys():
            mean_stat = np.mean([h[key] for h in histories], axis=0)
            stats[key] = mean_stat.tolist()
        
        return History(stats)

    def plot(self,
             what: str,
             show: bool = True,
             with_val: bool = False,
             figsize: tuple[int, int] = (5, 5),
             smooth: bool = True,
             grid: bool = True):
        
        stats = {what: self._stats_history[what]}
        epochs = np.arange(1, len(stats[what])+1)
        ticks = labels = epochs
        if with_val:
            val_what = f"val_{what}"
            if val_what in self._stats_history:
                stats[val_what] = self._stats_history[val_what]
            else:
                warnings.warn('"with_val" is set to True, but no "val_" key was found in history')
        graph_name = what.capitalize()

        if smooth:
            epochs_ = np.linspace(epochs.min(), epochs.max(), 1000)
            for (name, stat) in stats.items():
                stats[name] = PchipInterpolator(epochs, stat)(epochs_)
            epochs = epochs_

        fig, ax = plt.subplots(figsize=figsize, layout="tight")
        for (name, stat) in stats.items():
            ax.plot(epochs, stat, label=name)
        ax.set_xlabel("Epoch")
        ax.set_xticks(ticks=ticks, labels=labels)
        ax.set_ylabel(graph_name)
        ax.set_title(f"{graph_name} over epochs")
        ax.legend()

        if show:
            plt.grid(grid)
            plt.show()

        return (fig, ax)

    def visualize(self, *args, **kwargs):
        return self.plot(*args, **kwargs)