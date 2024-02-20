import collections.abc
import re

import torch


_COLORS = {
    'green': '\033[1;32m{text}\033[0m',
    'orange': '\033[38;5;208m{text}\033[0m',
    'purple': '\033[38;5;092m{text}\033[0m',
    }


def paint(text: str, color: str) -> str:
    assert color in _COLORS
    return _COLORS[color].format(text=text)


def format_metrics(metrics: dict[str, float],
                   metric_format: str = '{name}: {value:0.3f}',
                   sep: str = ', ',
                   with_color: bool = True,
                   ) -> str:
    if with_color:
        metric_format = re.sub(r'({value.*})',
                               paint(r'\1', 'purple'),
                               metric_format)
    return sep.join(metric_format.format(name=n, value=v) for (n, v) in metrics.items())


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


def add_file_ext(file_name: str, ext: str) -> str:
    if not ext.startswith("."):
        ext = f".{ext}"
    return file_name if file_name.endswith(ext) else f"{file_name}{ext}"