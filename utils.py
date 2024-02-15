from typing import Any

import numpy as np
from torch.utils.data import Dataset, Subset


def under_sample(dataset: Dataset,
                 minority_label: Any,
                 ratio: float = 1.0
                 ) -> Subset:
    minority_indices = []
    rest_indices = []

    for idx, (_, label) in enumerate(dataset):
        if label == minority_label:
            minority_indices.append(idx)
        else:
            rest_indices.append(idx)

    np.random.shuffle(rest_indices)

    rest_count = int(len(minority_indices) * ratio)
    indices = minority_indices + rest_indices[:rest_count]
    return Subset(dataset, indices)