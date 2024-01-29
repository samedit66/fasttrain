from collections import defaultdict
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import tensor

import json


class _NumbersEncoder(json.JSONEncoder):

    def default(self, x):
        if isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
            return float(x)
        else:
            return super().default(self, x)


class History():
    
    def __init__(self, initial=None):
        if initial is not None:
            self._stats_history = defaultdict(list, initial)
        else:
            self._stats_history = defaultdict(list)

    @classmethod
    def from_json(cls, file_name):
        if not file_name.endswith(".json"):
            file_name = f"{file_name}.json"
        with open(file_name, "r") as input_file:
            stats_history = json.load(input_file)
        return History(stats_history) 
            
    def to_json(self, file_name):
        if not file_name.endswith(".json"):
            file_name = f"{file_name}.json"
        with open(file_name, "w") as output_file:
            json.dump(self.history, output_file, indent=4, cls=_NumbersEncoder)

    def update(self, stats):
        for k, v in stats.items():
            self._stats_history[k].append(v)

    @property
    def history(self):
        return self._stats_history

    def compute_average(self):
        average_stats = {}
        for k, v in self._stats_history.items():
            average_stats[k] = sum(v) / len(v)
        return average_stats
    
    def __getitem__(self, key):
        return self.history[key]

    def plot(self,
             what: str,
             show: bool = True,
             with_val: bool = False,
             figsize: tuple[int, int] = (5, 5)):
        
        stats = {what: self.history[what]}
        epochs = np.arange(1, len(stats[what])+1)
        if with_val:
            val_what = f"val_{what}"
            if val_what in self.history:
                stats[val_what] = self.history[val_what]
            else:
                warnings.warn('"with_val" is set to True, but no "val_" key was found in history')
        graph_name = what.capitalize()

        fig, ax = plt.subplots(figsize=figsize, layout="tight")
        for (name, stat) in stats.items():
            ax.plot(epochs, stat, label=name)
        ax.set_xlabel("Epoch")
        ax.set_xticks(ticks=epochs, labels=epochs)
        ax.set_ylabel(graph_name)
        ax.set_title(f"{graph_name} over epochs")
        ax.legend()

        if show:
            plt.show()

        return (fig, ax)