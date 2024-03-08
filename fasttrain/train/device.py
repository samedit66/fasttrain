import collections

import torch


def available_devices() -> list[str]:
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    return devices


def auto_select_device(desired_device: str = 'auto') -> str:
    found_devices = available_devices()

    if desired_device == 'auto':
        return 'cuda' if 'cuda' in found_devices else 'cpu'
    elif desired_device in found_devices:
        return desired_device
    
    return 'cpu'


def load_data_on_device(dl: torch.utils.data.DataLoader, device: str):
    for batch in dl:
        if isinstance(batch, collections.abc.Sequence):
            yield [
                elem.to(device)
                if isinstance(elem, torch.Tensor)
                else elem
                for elem in batch
                ]
        elif isinstance(batch, collections.abc.Mapping):
            yield {
                key: (elem.to(device) if isinstance(elem, torch.Tensor) else elem)
                for (key, elem) in batch.items()
                }
        elif isinstance(batch, torch.Tensor):
            yield batch.to(device)
        else:
            raise TypeError(f'Type "{type(batch)}" is not supported')