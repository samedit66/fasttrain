r'''
# What is fasttrain?

`fasttrain` is a lightweight framework for building training loops for neural nets as fast as possible.
It's designed to remove all boring details about making up training loops in [PyTorch](https://pytorch.org/),
so you don't have to concentrate on how to pretty print a loss or metrics or bother about how to calculate them right.
'''

from .train import Trainer
from .reproduce import seed_everything