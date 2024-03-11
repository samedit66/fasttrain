import torch
import numpy as np

import random
import os


def seed_everything(seed: int) -> None:
    '''
    Sets a seed to every operation which includes randomness and disables
    non-deterministic behaviour of PyTorch.
    :param seed: Integer. The seed of all random operations.
    '''
    if not isinstance(seed, int):
        raise TypeError('Expect `seed` to be an integer')

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False