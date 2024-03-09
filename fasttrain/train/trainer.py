from abc import ABC, abstractmethod
import collections.abc

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..callbacks import Callback
from .history import History
from .device import (
    load_data_on_device,
    auto_select_device,
    )


class Trainer(ABC):

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 ) -> None:
        self._model = model
        self._opt = optimizer
        self._device = None
        self._is_training = False
        self._callbacks = []
        self._last_on_epoch_end_logs = {}

    def predict(self, input_batch):
        '''
        This function is called every time when the model predictions are needed.
        By default it expects a batch which should be a tuple or a list with 2 elements -
        x-batch and y-batch. If your training data differs, you need to define a custom
        predict function.
        :param input_batch: batch that the DataLoader yields.
        :return: model output batch.
        '''

        if isinstance(input_batch, collections.abc.Sequence):
            (x_batch, _) = input_batch
            return self.model(x_batch)
        
        raise TypeError('Predefined predict failed, perhaps you need to define '
                        'you custom predict function'
                        )

    @abstractmethod
    def compute_loss(self, input_batch, output_batch) -> float:
        '''
        This function is called every time when the loss value is needed.
        You need to define how the loss value is computed. This method should 
        return a float value.
        :param input_batch: batch that the DataLoader yields.
        :param output_batch: model output batch.
        :return: loss value.
        '''
        ...

    @abstractmethod
    def eval_metrics(self, input_batch, output_batch) -> collections.abc.Mapping:
        '''
        This function is called every time when metrics' values are needed.
        You need to define how they are computed. This method should return
        a dict-like object that contains metrics.
        :param input_batch: batch that the DataLoader yields.
        :param output_batch: model output batch.
        :return: metrics.
        '''
        ...

    @property
    def model(self) -> torch.nn.Module:
        '''
        Returns training model.
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

    @is_training.setter
    def is_training(self, status: bool) -> None:
        if not isinstance(status, bool):
            raise TypeError('Expect a value of bool type')

        self._is_training = status

    def _on_train_begin(self, logs={}):
        for cb in self._callbacks:
            cb.on_train_begin(logs)

    def _on_train_end(self, logs=None):
        for cb in self._callbacks:
            cb.on_train_end(self._last_on_epoch_end_logs)

    def _on_epoch_begin(self, epoch_num, logs=None):
        for cb in self._callbacks:
            cb.on_epoch_begin(epoch_num, logs)

    def _on_epoch_end(self, epoch_num, logs=None):
        self._last_on_epoch_end_logs = logs
        for cb in self._callbacks:
            cb.on_epoch_end(epoch_num, logs)

    def _on_train_batch_begin(self, batch_num, logs=None):
        for cb in self._callbacks:
            cb.on_train_batch_begin(batch_num, logs)

    def _on_train_batch_end(self, batch_num, logs=None):
        for cb in self._callbacks:
            cb.on_train_batch_end(batch_num, logs)

    def _on_validation_begin(self, logs=None):
        for cb in self._callbacks:
            cb.on_validation_begin(logs)

    def _on_validation_end(self, logs=None):
        for cb in self._callbacks:
            cb.on_validation_end(logs)

    def _on_validation_batch_begin(self, batch_num, logs=None):
        for cb in self._callbacks:
            cb.on_validation_batch_begin(batch_num, logs)

    def _on_validation_batch_end(self, batch_num, logs=None):
        for cb in self._callbacks:
            cb.on_validation_batch_end(batch_num, logs)

    def log(self, message: str) -> None:
        if self.is_training:
            tqdm.write(message)
        else:
            print(message)

    def _setup_device(self, desired_device: str = 'auto'):
        found_device = auto_select_device(desired_device)
        if desired_device != 'auto' and found_device != desired_device:
            self.log(f'Desired device {desired_device} not available, using {found_device}')
        else:
            self.log(f'Using {found_device}')
        self._device = found_device

    def _get_data_loader(self,
                         data: torch.utils.data.Dataset | torch.utils.data.DataLoader,
                         batch_size: int,
                         shuffle: bool) -> torch.utils.data.DataLoader:
        if (data is None) or isinstance(data, torch.utils.data.DataLoader):
            return data
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    def _compute_loss(self,
                      input_batch,
                      training: bool
                      ):
        output_batch = self.predict(input_batch)
        loss = self.compute_loss(input_batch, output_batch)

        if training:
            loss.backward()
            self._opt.step()
            self._opt.zero_grad()

        return (output_batch, loss.item())
    
    def _train(self, dl: DataLoader) -> collections.abc.Mapping:
        self.model.train()

        history = History()
        data_gen = load_data_on_device(dl, self._device)
        for (batch_num, input_batch) in enumerate(data_gen):
            self._on_train_batch_begin(batch_num)
            output_batch, loss_value = self._compute_loss(input_batch, training=True)
            metrics = self.eval_metrics(input_batch, output_batch)
            metrics["loss"] = loss_value
            history.update(metrics)
            self._on_train_batch_end(batch_num, history.average)
        
        return history.average
    
    @torch.no_grad()
    def _validate(self, dl: DataLoader) -> collections.abc.Mapping:
        self.model.eval()

        history = History()
        data_gen = load_data_on_device(dl, self._device)
        for (batch_num, input_batch) in enumerate(data_gen):
            self._on_validation_batch_begin(batch_num)
            output_batch, loss_value = self._compute_loss(input_batch, training=False)
            metrics = self.eval_metrics(input_batch, output_batch)
            metrics = {f'val_{k}': v for (k, v) in metrics.items()}
            metrics['val_loss'] = loss_value
            history.update(metrics)
            self._on_validation_batch_end(batch_num, history.average)

        return history.average

    def train(self,
              train_data: torch.utils.data.Dataset | torch.utils.data.DataLoader,
              num_epochs: int,
              device: str = 'auto',
              val_data: torch.utils.data.Dataset | torch.utils.data.DataLoader | None = None,
              batch_size: int = 16,
              shuffle: bool = True,
              callbacks: collections.abc.Sequence[Callback] | None = None
              ):
        self._setup_device(device)
        self._model = self._model.to(self._device)

        train_dl = self._get_data_loader(train_data, batch_size, shuffle)
        val_dl = self._get_data_loader(val_data, batch_size, shuffle)

        if callbacks is not None:
            self._callbacks = callbacks

            training_params = {
                'num_epochs': num_epochs,
                # TODO: Разобраться с IterableDataset при мультипроцессной загрузке данных
                'num_batches': len(train_dl),
                }
            for cb in self._callbacks:
                cb.trainer = self
                cb.model = self._model
                cb.training_params = training_params

        history = History()
        self.is_training = True
        self._on_train_begin()
        current_epoch_num = 1
        while self.is_training and current_epoch_num <= num_epochs:
            self._on_epoch_begin(current_epoch_num)
            metrics = self._train(train_dl)
            if val_dl is not None:
                metrics |= self._validate(val_dl)
            self._on_epoch_end(current_epoch_num, metrics)
            history.update(metrics)
            current_epoch_num += 1
        self.is_training = False
        self._on_train_end()

        return history