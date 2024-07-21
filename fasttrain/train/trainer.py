from __future__ import annotations
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Any
import logging

import torch
from tqdm.contrib.logging import logging_redirect_tqdm

from ..callbacks import (
    Callback,
    Tqdm    
    )
from .history import History
from ._utils import load_data_on_device
from ._notifier import CallbackNotifier


class UnavailableDeviceError(Exception):
    '''Raised when the given device is unavailable'''


class BaseTrainer(ABC):
    '''
    Base class for `Trainer`. Can be subclassed for creating a custom Trainer object,
    although subclassing `Trainer` is more preferrable (`BaseTrainer` doesn't support automatic device choosing,
    while `Trainer` does, same for `logger` parameter - `Trainer` creates a default one when no `logger` was provided).
    To create a custom trainer, subclass `BaseTrainer` and define/override `predict`, `compute_loss`, and `eval_metrics`.
    '''

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 logger: logging.Logger,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                 ) -> None:
        '''
        :param model: Model to train.
        :param optimizer: Optimizer for the model.
        :param logger: Logger to print metrics and other information about training.
        :param lr_scheduler: Learning rate scheduler. Defaults to `None`.
        '''
        self._model = model
        self._opt = optimizer
        self._lr_scheduler = lr_scheduler
        self._is_training = False
        self._logger = logger
        self._notifier = CallbackNotifier()

    def predict(self, input_batch: Any) -> Any:
        '''
        Called when the model predictions are needed.
        By default, unpacks input_batch into x and y, then calls `self._model` on x.

        :param input_batch: Input batch from the training/validating `DataLoader`.
        :return: Output batch.
        '''
        x, _ = input_batch
        output_batch = self._model(x)
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

    def eval_metrics(self, input_batch: Any, output_batch: Any) -> dict[str, Any] | None:
        '''
        Evaluates metrics. Called when model predictions are made.
        If defined, the returned metrics are stored in the history of training.
        Metrics must be a dict.

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

    def _stop_training(self) -> None:
        self._is_training = False

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

    def _train(self,
               dl: torch.utils.data.DataLoader,
               device: str | torch.device | None = None,
               ) -> dict[str, Any]:
        if device is not None:
            dl = load_data_on_device(dl, device)

        self.model.train()
        history = History()
        for (batch_num, input_batch) in enumerate(dl):
            self._notifier.notify_on_train_batch_begin(batch_num)
            if not self.is_training:
                break

            output_batch, loss_value = self._compute_loss(input_batch, training=True)
            metrics = self.eval_metrics(input_batch, output_batch) or {}
            metrics["loss"] = loss_value
            history.update(metrics)

            self._notifier.notify_on_train_batch_end(batch_num, history.average)
            if not self.is_training:
                break
        
        return history.average
    
    @torch.no_grad()
    def _validate(self,
                  dl: torch.utils.data.DataLoader,
                  device: str | torch.device,
                  ) -> dict[str, Any]:
        if device is not None:
            dl = load_data_on_device(dl, device)

        self.model.eval()
        history = History()
        for (batch_num, input_batch) in enumerate(dl):
            self._notifier.notify_on_validation_batch_begin(batch_num)
            if not self.is_training:
                break

            output_batch, loss_value = self._compute_loss(input_batch, training=False)
            metrics = self.eval_metrics(input_batch, output_batch) or {}
            metrics = {f'val_{k}': v for (k, v) in metrics.items()}
            metrics['val_loss'] = loss_value
            history.update(metrics)

            self._notifier.notify_on_validation_batch_end(batch_num, history.average)
            if not self.is_training:
                break

        return history.average

    def _training_loop(self,
                       train_dl: torch.utils.data.DataLoader,
                       val_dl: torch.utils.data.DataLoader,
                       num_epochs: int,
                       device: str | torch.device,
                       ) -> History:
        history = History()

        self._is_training = True
        self._notifier.notify_on_train_begin()
        current_epoch_num = 1

        while self.is_training and current_epoch_num <= num_epochs:
            self._notifier.notify_on_epoch_begin(current_epoch_num)
            if not self.is_training:
                break

            metrics = self._train(train_dl, device)
            if not self.is_training:
                break

            if val_dl is not None:
                metrics |= self._validate(val_dl, device)
                if not self.is_training:
                    break
            history.update(metrics)

            self._notifier.notify_on_epoch_end(current_epoch_num, metrics)
            if not self.is_training:
                break 

            current_epoch_num += 1
            
            if self._lr_scheduler is not None:
                self._lr_scheduler.step()

        self._stop_training()
        self._notifier.notify_on_train_end()

        return history

    def _setup_device(self,
                      device: str | torch.device,
                      ) -> torch.device | None:
        if device == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        elif device == 'none':
            device = None
        else:
            try:
                t = torch.tensor(1, device=device)
                device = t.device
            except (AssertionError, RuntimeError):
                # AssertionError should be raised when the given device is unavailable.
                # RuntimeError should be raised when the given device name is incorrect.
                raise UnavailableDeviceError(f'Requested device "{str(device)}" not available')
        return device

    def train(self,
              train_dl: torch.utils.data.DataLoader,
              num_epochs: int,
              device: str | torch.device | None = None,
              val_dl: torch.utils.data.DataLoader | None = None,
              callbacks: Sequence[Callback] = (),
              ) -> History:
        '''
        Trains the model for a fixed number of epochs.

        :param train_dl: Data on which to train the model.
        :param num_epochs: Integer. Number of epochs to train the model.
        :param device: Device for training. Defauls to `None`, which means no changing device for the model or dataloaders. 
        :param val_dl: Data on which to evaluate the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data. Defaults to `None`.
        :param callbacks: Callbacks to interact with the model and metrics during various stages of training.
        :return: History object. The history of training which includes validation metrics if `val_data` present.
        '''
        device = self._setup_device(device)
        if device is not None:
            self._model = self._model.to(device)

        training_args = {
            'num_epochs': num_epochs,
            # TODO: Разобраться с IterableDataset при мультипроцессной загрузке данных
            'num_batches': len(train_dl),
            'val_num_batches': len(val_dl),
            }
        
        self._notifier.setup(callbacks, self._model, self, training_args)
        
        history = self._training_loop(train_dl, val_dl, num_epochs, device)
        return history


class Trainer(BaseTrainer, metaclass=ABCMeta):
    '''
    Base class for all user-defined trainers.
    To create a custom trainer, subclass `fasttrain.Trainer` and define/override
    `predict`, `compute_loss`, and `eval_metrics`.
    '''

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 logger: logging.Logger | None = None,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                 ) -> None:
        logger = logger or self._get_default_logger()
        super().__init__(model=model,
                         optimizer=optimizer,
                         logger=logger,
                         lr_scheduler=lr_scheduler,
                         )

    def _get_default_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    def _select_device(self,
                       requested_device: str | torch.device,
                       ) -> torch.device | None:
        '''
        Selects a device for training based on the requested device.
        If `requested_device` is `'auto'`, tries to use CUDA if available, otherwise uses CPU.
        If `requested_device` is `'none'`, doesn't use any device at all and returns `None` (useful if you already set the device).
        Otherwise, tries to make up a tensor on the requested device, if fails, raises `UnavailableDeviceError`.
        
        :param requested_device: Desired device for training.
        :return: Chosen device for training or `None` if no device is selected.
        '''
        if requested_device == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        elif requested_device == 'none':
            device = None
        else:
            try:
                t = torch.tensor(1, device=requested_device)
                device = t.device
            except (AssertionError, RuntimeError):
                # AssertionError should be raised when the given device is unavailable.
                # RuntimeError should be raised when the given device name is incorrect.
                raise UnavailableDeviceError(f'Requested device "{str(requested_device)}" not available')
        return device

    def train(self,
              train_dl: torch.utils.data.DataLoader,
              num_epochs: int,
              device: str | torch.device = 'auto',
              val_dl: torch.utils.data.DataLoader | None = None,
              callbacks: Sequence[Callback] = (),
              ) -> History:
        '''
        Trains the model for a fixed number of epochs.

        :param train_dl: Data on which to train the model.
        :param num_epochs: Integer. Number of epochs to train the model.
        :param device: Defaults to `"auto"`. If `"auto"`, tries
            to automatically detect suitable device for training, preferrably, CUDA.
            If `'none'`, doesn't change device for training (useful if you already set the device).
        :param val_dl: Data on which to evaluate the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data. Defaults to `None`.
        :param callbacks: Callbacks to interact with the model and metrics during various stages of training.
            Tqdm callback, which prints the progress bar, is added automaticly.
        :return: History object. The history of training which includes validation metrics if `val_data` present.
        '''
        device = self._select_device(device)
        callbacks = list(callbacks) + [Tqdm()]
        return super().train(train_dl=train_dl,
                             num_epochs=num_epochs,
                             device=device,
                             val_dl=val_dl,
                             callbacks=callbacks,
                             )
    