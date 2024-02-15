from abc import ABC, abstractmethod
import collections.abc 

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.history import History
from _utils import (
    load_data_on_device,
    auto_select_device,
    format_metrics,
    paint,
    )


class Trainer(ABC):

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: str | torch.optim.Optimizer,
                 ) -> None:
        self.__model = model
        self.__opt = optimizer
        self.__device = None

    @abstractmethod
    def predict(self, input_batch): ...

    @abstractmethod
    def compute_loss(self, input_batch, output_batch): ...

    @abstractmethod
    def eval_metrics(self, input_batch, output_batch): ...

    def __log(self, message: str,) -> None:
        tqdm.write(message)

    def __setup_device(self, desired_device: str | None):
        found_device = auto_select_device(desired_device)
        if desired_device is not None and found_device != desired_device:
            self.__log('Desired device {desired_device} not available, using {found_device}')
        else:
            self.__log('Using {found_device}')
        self.__device = found_device

    def __get_data_loader(self,
                          data: collections.abc.Sequence | None,
                          batch_size: int,
                          shuffle: bool) -> torch.utils.data.DataLoader:
        if (data is None) or isinstance(data, torch.utils.data.DataLoader):
            return data
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    def __compute_loss(self,
                       input_batch,
                       training: bool
                       ):
        output_batch = self.predict(input_batch)
        loss = self.compute_loss(input_batch, output_batch)

        if training:
            loss.backward()
            self.__opt.step()
            self.__opt.zero_grad()

        return (output_batch, loss.item())
    
    def __one_epoch(self, dl: DataLoader, training: bool) -> dict:
        history = History()

        pbar = tqdm(load_data_on_device(dl, self.__device), unit="batch")
        for input_batch in pbar:
            output_batch, loss_value = self.__compute_loss(input_batch, training)
            metrics = self.eval_metrics(input_batch, output_batch)
            metrics["loss"] = loss_value
            history.update(metrics)
            pbar.set_description(f"  Current - {format_metrics(metrics)}")

        average = history.average
        self.__log(f"  Average - {format_metrics(average)}")

        return average
    
    def __train(self, dl: DataLoader) -> dict:
        self.__log(paint("Training...", "green"))
        self.model.train()
        metrics = self.__one_epoch(dl, training=True)
        return metrics
    
    @torch.no_grad()
    def __validate(self, dl: DataLoader) -> dict:
        self.__log(paint("Validating...", "orange"))
        self.model.eval()
        metrics = self.__one_epoch(dl, training=False)
        return metrics

    def train(self,
              train_data: collections.abc.Sequence | torch.utils.data.DataLoader,
              num_epochs: int,
              device: str | None = None,
              val_data: collections.abc.Sequence | torch.utils.data.DataLoader | None = None,
              batch_size: int = 32,
              shuffle: bool = True,
              patience: int | None = None
              ) -> History:
        self.__setup_device(device)
        self.__model = self.__model.to(self.__device)

        train_dl = self.__get_data_loader(train_data, batch_size, shuffle)
        val_dl = self.__get_data_loader(val_data, batch_size, shuffle)

        history = History()
        for epoch_num in range(1, num_epochs + 1):
            self.__log(f"Epoch {epoch_num}/{num_epochs}")

            t_metrics = self.__train(train_dl)
            history.update(t_metrics)

            if val_dl is not None:
                v_metrics = self.__validate(val_dl)
                history.update(v_metrics)

        return history
