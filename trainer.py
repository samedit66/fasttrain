from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from history import History


def _color_green(text):
    return f"\033[1;32m{text}\033[0m"


def _color_purple(text):
    return f"\033[38;5;092m{text}\033[0m"


def _color_orange(text):
    return f"\033[38;5;208m{text}\033[0m"


def _format_metrics(metrics):
    sep=", "
    metric_format = "{name}: " + _color_purple("{value:0.3f}")
    return sep.join(
        metric_format.format(name=k, value=v) for k, v in metrics.items()
    )


class Trainer(ABC):

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    @abstractmethod
    def predict(self, input_batch):
        pass

    @abstractmethod
    def compute_loss(self, input_batch, output_batch):
        pass

    @abstractmethod
    def evaluate_metrics(self, input_batch, output_batch):
        pass

    def train(self, train_data_loader, num_epochs, val_data_loader=None):
        history = History()

        for epoch_num in range(1, num_epochs+1):
            tqdm.write(f"Epoch {epoch_num}/{num_epochs}")

            tqdm.write(_color_green("Training..."))
            train_metrics = self._train_epoch(train_data_loader)
            history.update(train_metrics)

            if val_data_loader is not None:
                tqdm.write(_color_orange("Validating..."))
                val_metrics = self._validate(val_data_loader)
                history.update({f"val_{m}": v for m, v in val_metrics.items()})

            tqdm.write("")

        return history
    
    def _train_epoch(self, train_data_loader):
        self.model.train()

        metrics_history = History()
        progress_bar = tqdm(train_data_loader, unit="batch")
        for input_batch in progress_bar:
            output_batch = self.predict(input_batch)

            loss = self.compute_loss(input_batch, output_batch)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            metrics = self.evaluate_metrics(input_batch, output_batch)
            metrics["loss"] = loss.item()

            metrics_history.update(metrics)

            progress_bar.set_description(
                f"  Current - {_format_metrics(metrics)}"
            )

        average_metrics = metrics_history.average
        tqdm.write(f"  Average - {_format_metrics(average_metrics)}")

        return average_metrics

    @torch.no_grad()
    def _validate(self, val_data_loader):
        self.model.eval()

        metrics_history = History()
        progress_bar = tqdm(val_data_loader, unit="batch")
        for input_batch in progress_bar:
            output_batch = self.predict(input_batch)

            loss = self.compute_loss(input_batch, output_batch)
            metrics = self.evaluate_metrics(input_batch, output_batch)
            metrics["loss"] = loss.item()

            metrics_history.update(metrics)

            progress_bar.set_description(
                f"  Current - {_format_metrics(metrics)}"
            )

        average_metrics = metrics_history.average
        tqdm.write(f"  Average - {_format_metrics(average_metrics)}")

        return average_metrics