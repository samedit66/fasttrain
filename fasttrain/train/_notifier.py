from __future__ import annotations
from typing import (
    Any,
    Mapping,
    Sequence,
    TYPE_CHECKING,
    )

import torch

from ..callbacks import Callback
if TYPE_CHECKING:
    from .trainer import Trainer


class CallbackNotifier:
    
    def __init__(self) -> None:
        self._callbacks = []
        self._last_on_epoch_end_logs = {}

    def setup(self,
              callbacks: Sequence[Callback],
              model: torch.nn.Module,
              trainer: Trainer,
              training_args: Mapping[str, Any],
              ) -> None:
        self._callbacks.extend(callbacks)
        for cb in self._callbacks:
            cb.model = model
            cb.trainer = trainer
            cb.training_args = training_args

    def notify_on_train_begin(self) -> None:
        for cb in self._callbacks:
            cb.on_train_begin()

    def notify_on_train_end(self) -> None:
        for cb in self._callbacks:
            cb.on_train_end(self._last_on_epoch_end_logs)

    def notify_on_epoch_begin(self, epoch_num: int) -> None:
        for cb in self._callbacks:
            cb.on_epoch_begin(epoch_num)

    def notify_on_epoch_end(self,
                            epoch_num: int,
                            logs: Mapping[str, Any],
                            ) -> None:
        self._last_on_epoch_end_logs = logs
        for cb in self._callbacks:
            cb.on_epoch_end(epoch_num, logs)

    def notify_on_train_batch_begin(self, batch_num: int) -> None:
        for cb in self._callbacks:
            cb.on_train_batch_begin(batch_num)

    def notify_on_train_batch_end(self,
                                  batch_num: int,
                                  logs: Mapping[str, Any],
                                  ) -> None:
        for cb in self._callbacks:
            cb.on_train_batch_end(batch_num, logs)

    def notify_on_validation_begin(self) -> None:
        for cb in self._callbacks:
            cb.on_validation_begin()

    def notify_on_validation_end(self, logs: Mapping[str, Any]) -> None:
        for cb in self._callbacks:
            cb.on_validation_end(logs)

    def notify_on_validation_batch_begin(self, batch_num: int) -> None:
        for cb in self._callbacks:
            cb.on_validation_batch_begin(batch_num)

    def notify_on_validation_batch_end(self,
                                       batch_num: int,
                                       logs: Mapping[str, Any],
                                       ) -> None:
        for cb in self._callbacks:
            cb.on_validation_batch_end(batch_num, logs)
