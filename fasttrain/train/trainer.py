from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence, Mapping
from typing import Any
import logging

import torch
from tqdm.contrib.logging import logging_redirect_tqdm

from ..callbacks import (
    Callback,
    Tqdm    
    )
from .history import History
from .hardware import (
    load_data_on_device,
    can_be_used,
    appropriate_device,
    get_cpu_name,
    get_gpu_name,
    )
from .._utils.colors import success, fail, blue


class Trainer(ABC):
    '''
    Base class for all user defined trainers.
    To create a custom trainer, subclass `fasttrain.Trainer` and define/override
    `predict`, `compute_loss`, and `eval_metrics`.
    '''

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                 ) -> None:
        '''
        :param model: Model to train.
        :param optimizer: Optimizer for the model.
        '''
        self._model = model
        self._opt = optimizer
        self._lr_scheduler = lr_scheduler
        self._device = None
        self._is_training = False
        self._callbacks = []
        self._last_on_epoch_end_logs = {}
        self._logger = self._get_logger()

    def predict(self, input_batch: Any) -> Any:
        '''
        Called when the model predictions are needed.
        By default, unpacks input_batch into x and y, then calls `self.model` on x.

        :param input_batch: Input batch from the training/validating `DataLoader`.
        :return: Output batch.
        '''
        x, _ = input_batch
        output_batch = self.model(x)
        return output_batch

    @abstractmethod
    def compute_loss(self, input_batch: Any, output_batch: Any) -> torch.Tensor:
        '''
        Called when the loss value is needed.
        You need to define how the loss value is computed.
        This method must return a `torch.Tensor`.

        :param input_batch: Input batch from the training/validating `DataLoader`.
        :param output_batch: Batch returned by `predict(input_batch)`.
        :return: Loss value.
        '''

    def eval_metrics(self, input_batch: Any, output_batch: Any) -> Mapping[str, Any] | None:
        '''
        Evaluates metrics. Called when model predictions are made.
        If defined, the returned metrics are stored in the history of training.
        Metrics must be a dict or a mapping.

        :param input_batch: Input batch from the training/validating `DataLoader`.
        :param output_batch: Batch returned by `predict(input_batch)`.
        :return: Metrics.
        '''
        return None

    @property
    def model(self) -> torch.nn.Module:
        '''
        Returns the training model.

        :return: Training model.
        '''
        return self._model

    @property
    def is_training(self) -> bool:
        '''
        Returns a bool value whether the model is training now.

        :return: `True` if the model is training, `False` otherwise. 
        '''
        return self._is_training

    def log(self, message: str) -> None:
        '''
        Logs a message to stdout. Should be used to inform user about model training because
        ordinary `print` may break up the progress bar. Use it only inside a custom `Callback`.
        
        :param message: Message to log.
        '''
        with logging_redirect_tqdm():
            self._logger.info(message)

    def _get_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    def __log_cpu_and_gpu_info(self):
        cpu_name = get_cpu_name()
        if cpu_name is None:
            self.log(fail('Can not recognize CPU'))
        
        # Yup, there are actually 2 whitespaces instead of one -
        # otherwise the printing isn't pretty for me.
        cpu_msg = 'CPU  found: %s'
        if 'Intel' in cpu_name:
            cpu_msg = cpu_msg % blue(cpu_name)
        elif 'AMD' in cpu_name:
            cpu_msg = cpu_msg % fail(cpu_name)
        else:
            cpu_msg = cpu_msg % cpu_name
        self.log(cpu_msg)

        if torch.cuda.is_available():
            cuda_name = get_gpu_name()
            cuda_msg = f'CUDA found: {success(cuda_name)}'
            self.log(cuda_msg)
        else:
            self.log(fail('CUDA not found'))

    def __setup_device(self,
                       preferable_device: str | torch.device,
                       ) -> None:
        self.__log_cpu_and_gpu_info()

        if preferable_device != 'auto' and not can_be_used(preferable_device):
            device_name = str(preferable_device).upper()
            self.log(fail(f'Requested device {device_name} not available, default one will be used'))

        chosen_device = appropriate_device(preferable_device)
        device_name = str(chosen_device).upper()
        self.log(f'Using {device_name}')

        self._device = chosen_device

    def __setup_callbacks(self,
                          callbacks: Sequence[Callback],
                          training_args: Mapping[str, Any],
                          ) -> None:
        default_callbacks = [Tqdm()]
        all_callbacks = [*default_callbacks, *callbacks]
        for cb in all_callbacks:
            cb.trainer = self
            cb.model = self.model
            cb.training_args = training_args
        self._callbacks = all_callbacks
    
    def _stop_training(self) -> None:
        self._is_training = False

    def _notify_on_train_begin(self) -> None:
        for cb in self._callbacks:
            cb.on_train_begin()

    def _notify_on_train_end(self) -> None:
        for cb in self._callbacks:
            cb.on_train_end(self._last_on_epoch_end_logs)

    def _notify_on_epoch_begin(self, epoch_num: int) -> None:
        for cb in self._callbacks:
            cb.on_epoch_begin(epoch_num)

    def _notify_on_epoch_end(self, epoch_num: int, logs: Mapping) -> None:
        self._last_on_epoch_end_logs = logs
        for cb in self._callbacks:
            cb.on_epoch_end(epoch_num, logs)

    def _notify_on_train_batch_begin(self, batch_num: int) -> None:
        for cb in self._callbacks:
            cb.on_train_batch_begin(batch_num)

    def _notify_on_train_batch_end(self, batch_num: int, logs: Mapping) -> None:
        for cb in self._callbacks:
            cb.on_train_batch_end(batch_num, logs)

    def _notify_on_validation_begin(self) -> None:
        for cb in self._callbacks:
            cb.on_validation_begin()

    def _notify_on_validation_end(self, logs: Mapping) -> None:
        for cb in self._callbacks:
            cb.on_validation_end(logs)

    def _notify_on_validation_batch_begin(self, batch_num: int) -> None:
        for cb in self._callbacks:
            cb.on_validation_batch_begin(batch_num)

    def _notify_on_validation_batch_end(self, batch_num: int, logs: Mapping) -> None:
        for cb in self._callbacks:
            cb.on_validation_batch_end(batch_num, logs)

    def _compute_loss(self,
                      input_batch: Any,
                      training: bool,
                      ) -> tuple[Any, float]:
        output_batch = self.predict(input_batch)
        loss = self.compute_loss(input_batch, output_batch)

        if training:
            loss.backward()
            self._opt.step()
            self._opt.zero_grad()

        return (output_batch, loss.item())
    
    def _train(self, dl: torch.utils.data.DataLoader) -> Mapping:
        self.model.train()

        history = History()
        data_gen = load_data_on_device(dl, self._device)
        for (batch_num, input_batch) in enumerate(data_gen):
            self._notify_on_train_batch_begin(batch_num)
            if not self.is_training:
                break

            output_batch, loss_value = self._compute_loss(input_batch, training=True)
            metrics = self.eval_metrics(input_batch, output_batch) or {}
            metrics["loss"] = loss_value
            history.update(metrics)

            self._notify_on_train_batch_end(batch_num, history.average)
            if not self.is_training:
                break
        
        return history.average
    
    @torch.no_grad()
    def _validate(self, dl: torch.utils.data.DataLoader) -> Mapping:
        self.model.eval()

        history = History()
        data_gen = load_data_on_device(dl, self._device)
        for (batch_num, input_batch) in enumerate(data_gen):
            self._notify_on_validation_batch_begin(batch_num)
            if not self.is_training:
                break

            output_batch, loss_value = self._compute_loss(input_batch, training=False)
            metrics = self.eval_metrics(input_batch, output_batch) or {}
            metrics = {f'val_{k}': v for (k, v) in metrics.items()}
            metrics['val_loss'] = loss_value
            history.update(metrics)

            self._notify_on_validation_batch_end(batch_num, history.average)
            if not self.is_training:
                break

        return history.average

    def _training_loop(self,
                       train_dl: torch.utils.data.DataLoader,
                       val_dl: torch.utils.data.DataLoader,
                       num_epochs: int,
                       ) -> History:
        history = History()

        self._is_training = True
        self._notify_on_train_begin()
        current_epoch_num = 1

        while self.is_training and current_epoch_num <= num_epochs:
            self._notify_on_epoch_begin(current_epoch_num)
            if not self.is_training:
                break

            metrics = self._train(train_dl)
            if not self.is_training:
                break

            if val_dl is not None:
                metrics |= self._validate(val_dl)
                if not self.is_training:
                    break
            history.update(metrics)

            self._notify_on_epoch_end(current_epoch_num, metrics)
            if not self.is_training:
                break 

            current_epoch_num += 1

        self._stop_training()
        self._notify_on_train_end()

        return history

    def train(self,
              train_dl: torch.utils.data.DataLoader,
              num_epochs: int,
              val_dl: torch.utils.data.DataLoader | None = None,
              callbacks: Sequence[Callback] = (),
              device: str | torch.device = 'auto',
              ) -> History:
        '''
        Trains the model for a fixed number of epochs.

        :param train_dl: Data on which to train the model.
        :param num_epochs: Integer. Number of epochs to train the model.
        :param device: Defaults to `"auto"`. If `"auto"`, tries
            to automatically detect suitable device for training, preferrably, CUDA. 
        :param val_dl: Data on which to evaluate the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data. Defaults to `None`.
        :param callbacks: Callbacks to interact with the model and metrics during various stages of training.
            Tqdm callback, which prints the progress bar, is added automaticly.
        :return: History object. The history of training which includes validation metrics if `val_data` present.
        '''
        self.__setup_device(preferable_device=device)
        self._model = self._model.to(self._device)

        training_args = {
            'num_epochs': num_epochs,
            # TODO: Разобраться с IterableDataset при мультипроцессной загрузке данных
            'num_batches': len(train_dl),
            'val_num_batches': len(val_dl),
            }
        self.__setup_callbacks(callbacks, training_args)

        history = self._training_loop(train_dl, val_dl, num_epochs)
        return history
    