from collections.abc import Sequence, Mapping, Iterable
from typing import Any

import torch
from torch.utils.data import DataLoader


def available_devices() -> list[str]:
    '''
    Returns all available devices for training (includes only cpu and cuda devices!).
    :return: Avalilable devices for training.
    '''
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
        cuda_device_count = torch.cuda.device_count()
        devices.extend(f'cuda:{i}' for i in range(cuda_device_count))
    return devices


def auto_select_device(desired_device: str | torch.device = 'auto') -> torch.device:
    '''
    Selects device for training. Firstly, tries to find out `desired_device`, if fails,
    then tries to find cuda, and if fails again, returns `torch.device('cpu')`.

    :param desired_device: Desired device for training. Default to `"auto"`.
    :return: Selected device for training.
    '''
    found_devices = available_devices()
    default_device = torch.device('cpu')

    str_desired_device = str(desired_device)
    if str_desired_device in found_devices:
        return torch.device(desired_device)
    elif str_desired_device == 'auto':
        # Do not combine the condition above with this condition,
        # because it's unknown whether we will not search for other devices later!
        if 'cuda' in found_devices:
            return torch.device('cuda')
    else:
        try:
            _ = torch.tensor(1, device=torch.device(desired_device))
        except (RuntimeError, AssertionError):
            # This should fail when either the given device is incorrect, or it's not available.
            return default_device
        except:
            # Don't know when this exactly fails, but I want to distinguish between
            # this case and the above one. 
            return default_device
    
    return default_device


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