from __future__ import annotations
from collections.abc import Sequence, Mapping, Iterable
from typing import Any
import os
import platform
import subprocess
import re

import torch
from torch.utils.data import DataLoader


def get_cpu_name() -> str | None:
    if platform.system() == "Windows":
        name = subprocess.check_output(["wmic", "cpu", "get", "name"])
        return name.decode().strip().split('\n')[1]
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command ="sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*:", "", line, 1).strip()
    return None


def get_gpu_name() -> str | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name()


def can_be_used(device: str | torch.device) -> bool:
    try:
        _ = torch.tensor(1, device=device)
    except (AssertionError, RuntimeError):
        # AssertionError should be raised when the given device is unavailable.
        # RuntimeError should be raised when the given device name is incorrect.
        return False
    return True


def load_data_on_device(dl: DataLoader, device: str | torch.device) -> Iterable[Any]:
    '''
    Takes a `DataLoader` object and makes it yield batches transfered to specified device.
    Supported types, which the given `DataLoader` must return, are: `Sequence`, `Mapping` and `torch.Tensor`.
    Other types would cause throwing a `TypeError`.
    If `dl` yields `Sequence`, each element, that is a `torch.Tensor` ,will be transfered to `device`.
    If `dl` yields `torch.Tensor`, the tensor will be transfered to `device`. 
    If `dl` yields `Mapping`, each value, that is a `torch.Tensor`, of a pair will be transfered to `device`.

    :param dl: DataLoader object.
    :param device: Desired device to transfer batches to.
    :return: Generator, which yields transfered batches.
    '''
    for batch in dl:
        if isinstance(batch, Sequence):
            yield [
                elem.to(device)
                if isinstance(elem, torch.Tensor)
                else elem
                for elem in batch
                ]
        elif isinstance(batch, Mapping):
            yield {
                key: (elem.to(device) if isinstance(elem, torch.Tensor) else elem)
                for (key, elem) in batch.items()
                }
        elif isinstance(batch, torch.Tensor):
            yield batch.to(device)
        else:
            raise TypeError(f'Type "{type(batch)}" is not supported')
        