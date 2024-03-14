from abc import ABC, ABCMeta, abstractmethod
import collections.abc

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..callbacks import (
    Callback,
    Tqdm    
    )
from .history import History
from .device import (
    load_data_on_device,
    auto_select_device,
    )


class Trainer(ABC):
    '''
    Base class for all user defined trainers. Usually, to make up a trainer,
    one should subclass `Trainer` and define `predict`, `compute_loss` and `eval_metrics`.
    Although you don't have to always define `predict` (see its docs).
    '''

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
        self._verbose = True
        self._in_notebook = None

    def predict(self, input_batch):
        '''
        This function is called every time when the model predictions are needed.
        By default it expects a batch which should be a tuple or a list with 2 elements -
        x-batch and y-batch. If your training data differs, you need to define a custom
        predict function.
        :param input_batch: Batch that the DataLoader yields.
        :return: Model output batch.
        '''
        if isinstance(input_batch, collections.abc.Sequence):
            (x_batch, _) = input_batch
            return self.model(x_batch)
        
        raise TypeError('Predefined predict failed, perhaps you need to define '
                        'your custom predict function'
                        )

    @abstractmethod
    def compute_loss(self, input_batch, output_batch) -> float:
        '''
        This function is called every time when the loss value is needed.
        You need to define how the loss value is computed. This method should 
        return a float value.
        :param input_batch: Batch that the DataLoader yields.
        :param output_batch: Model output batch.
        :return: Loss value.
        '''
        ...

    @abstractmethod
    def eval_metrics(self, input_batch, output_batch) -> collections.abc.Mapping:
        '''
        This function is called every time when metrics' values are needed.
        You need to define how they are computed. This method should return
        a dict-like object that contains metrics.
        :param input_batch: Batch that the DataLoader yields.
        :param output_batch: Model output batch.
        :return: Metrics.
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
        '''
        Sets the status of training. This function must be used only inside a `Callback` class to
        stop model training and to stop model training.
        :param status: Status of training. When `False` and the model was in training mode,
        training immediatly stops.
        '''
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

    def _log(self, message: str) -> None:
        '''
        Logs a message to stdout. Should be used to inform user about model training because
        ordinary `print` may break up the progress bar. Use it only inside a custom `Callback`.
        
        :param message: Message to log.
        '''
        if self.is_training:
            tqdm.write(message)
        else:
            print(message)

    def _is_in_notebook(self) -> bool:
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False      # Probably standard Python interpreter

    def _is_in_colab(self) -> bool:
        try:
            import google.colab
            return True
        except:
            return False

    def _setup_callbacks(self,
                         user_callbacks,
                         training_params: dict,
                         ) -> None:
        if user_callbacks is None:
            user_callbacks = []

        if self._verbose:
            if self._in_notebook is None:
                self._in_notebook = self._is_in_notebook() or self._is_in_colab()

            self._log(f'Running as a {"notebook" if self._in_notebook else "script"}')
            progress_bar = Tqdm(in_notebook=self._in_notebook)
            progress_bar.model = self.model
            progress_bar.trainer = self
            progress_bar.training_params = training_params
            self._callbacks.append(progress_bar)
        
        for user_callback in user_callbacks:
            if self._verbose and isinstance(user_callbacks, Tqdm):
                continue

            user_callback.model = self.model
            user_callback.trainer = self
            user_callback.training_params = training_params
            self._callbacks.append(user_callback)

    def _setup_device(self, desired_device: str = 'auto'):
        found_device = auto_select_device(desired_device)
        if desired_device != 'auto' and found_device != desired_device:
            self._log(f'Desired device {desired_device} not available, using {found_device}')
        else:
            self._log(f'Using {found_device}')
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

    def _training_loop(self,
                       train_dl: torch.utils.data.DataLoader,
                       val_dl: torch.utils.data.DataLoader,
                       num_epochs: int,
                       ) -> History:
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

    def train(self,
              train_data: torch.utils.data.Dataset | torch.utils.data.DataLoader,
              num_epochs: int,
              verbose: bool = True,
              device: str = 'auto',
              val_data: torch.utils.data.Dataset | torch.utils.data.DataLoader | None = None,
              batch_size: int = 16,
              shuffle: bool = True,
              callbacks: collections.abc.Sequence[Callback] | None = None,
              in_notebook: bool | None = None,
              ) -> History:
        '''
        Trains the model for a fixed number of epochs.

        :param train_data: A Dataset or DataLoader object. If it's a DataLoader,
        `batch_size` and `shuffle` are ignored. Otherwise, `train` makes up a DataLoader
            from the given Dataset object.
        :param num_epochs: Integer. Number of epochs to train the model.
        :param verbose: Verbosity mode. Default to `True`. If `False`, no progress bar
            appears and no messages are printed.
        :param device: `"auto"`, `"cpu"`, `"cuda"`. Default to `"auto"`. If `"auto"`, tries
            to automatically detect suitable device for training, preferrably, cuda. 
        :param val_data: Data on which to evaluate the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data. Can be either a Dataset or DataLoader object. If it's a DataLoader,
            `batch_size` and `shuffle` are ignored. Otherwise, `train` makes up a validation DataLoader
            from the given Dataset object.
        :param batch_size: Integer. Default to 16. Used when `train_data` or `val_data` aren't DataLoaders.
        :param shuffle: Boolean, whether to shuffle the training data before each epoch. Default to `True`.
            Used when `train_data` or `val_data` aren't DataLoaders.
        :param callbacks: Callbacks to interact with the model and metrics during various stages of training.
            The use of the progress bar callback is controlled by `verbose`, one don't need to add it explicity.
        :param in_notebook: Used to correctly display the progress bar. If `None`, tries to automatically detect
            whether running in a notebook or not. If `True`, forces to show a progress bar as it looks in a notebook
            (leads to a strange-looking progress bar when not in a notebook).
        :return: History object. The history of training which includes validation metrics if `val_data` present.
        '''
        self._setup_device(device)
        self._model = self._model.to(self._device)

        train_dl = self._get_data_loader(train_data, batch_size, shuffle)
        val_dl = self._get_data_loader(val_data, batch_size, shuffle)

        self._verbose = verbose
        self._in_notebook = in_notebook
        training_params = {
            'num_epochs': num_epochs,
            # TODO: Разобраться с IterableDataset при мультипроцессной загрузке данных
            'num_batches': len(train_dl),
            }
        self._setup_callbacks(callbacks, training_params)

        history = self._training_loop(train_dl, val_dl, num_epochs)
        return history
