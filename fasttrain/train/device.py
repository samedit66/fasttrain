from __future__ import annotations
from collections.abc import Sequence, Mapping, Iterable
from typing import Any

import torch
from torch.utils.data import DataLoader


class Devices:

    @staticmethod
    def available_devices() -> Sequence[torch.device]:
        '''
        Returns all found devices (includes only cpu and cuda devices if available).
    
        :return: Available devices.
        '''
        found_devices = [torch.device('cpu')]
        if torch.cuda.is_available():
            found_devices.append(torch.device('cuda'))
            cuda_device_count = torch.cuda.device_count()
            found_devices.extend(torch.device(f'cuda:{i}') for i in range(cuda_device_count))

        return found_devices

    @staticmethod
    def appropriate_device() -> torch.device:
        '''
        The most suitable device for using.

        :return: Most suitable device for using.
        '''
        preferable = torch.device('cuda')
        if preferable in Devices.available_devices():
            return preferable
        else:
            return torch.device('cuda')

    @staticmethod
    def can_be_used(device: str | torch.device) -> bool:
        '''
        Checks if the given device can be used.

        :return: `True` if the given device can be used, `False` otherwise.
        '''
        try:
            _ = torch.tensor(1, device=device)
        except AssertionError:
            # This should fail when the given device is unavailable.
            return False
        except RuntimeError:
            # This should fail when the given device name is incorrect.
            return False
        return True

    @staticmethod
    def is_gpu_available() -> bool:
        '''
        Checks if GPU (CUDA) available.

        :return: `True` if GPU (cuda) available, `False` otherwise.
        '''
        return torch.cuda.is_available()


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