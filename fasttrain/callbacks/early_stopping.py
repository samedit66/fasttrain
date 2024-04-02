from collections.abc import Mapping

from . import Callback
from ._colors import paint


class EarlyStopping(Callback):
    '''
    Callback to prevent overfitting while training.
    Stops training when a monitored metrics has stopped improving.

    :param patience: Number of epochs with no improvement after which
        training will be stopped.
    :param monitor: Metric to monitor (defaults to `"val_loss"`).
    :param mode: One of `{"min", "max"}`. In `"min"` mode, training will
        stop when the quantity monitored has stopped descreasing; in `"max"`
        mode it will stop when the quantity monitored has stopped increasing.
    :param restore_best_weights: Whether to restore model weights from the epoch
        with the best value of monitored quantity (defaults to `False`).
    '''

    def __init__(self,
                 patience: int,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 restore_best_weights: bool = True,
                 ) -> None:
        super().__init__()
        assert isinstance(patience, int) and patience >= 1 
        self._patience = patience
        self._monitor = monitor
        self._mode = mode
        self._restore_best_weights = restore_best_weights
        self._last_metric_value = None
        self._remaining_patience = patience
        self._best_model_weights = None
        self._best_metric_value = None
        self._best_epoch = None
        self._stopped_epoch = None

    def _is_improvement(self, metric_value):
        if self._mode == 'min':
            return metric_value <= self._last_metric_value
        elif self._mode == 'max':
            return metric_value >= self._last_metric_value

    def _is_best(self, metric_value):
        if self._mode == 'min':
            return metric_value <= self._best_metric_value
        elif self._mode == 'max':
            return metric_value >= self._best_metric_value

    def on_train_end(self, logs: Mapping) -> None:
        if self._stopped_epoch is not None:
            self.trainer._log(
                f'Epoch: {self._stopped_epoch} - early stopping.'
            )
        if self._restore_best_weights and self._best_epoch is not None:
            self.trainer._log(
                f'Restoring model weights from the end of the best epoch: {self._best_epoch}.'
            )
            if self._best_model_weights:
                self.model.load_state_dict(self._best_model_weights)
            else:
                self.trainer._log('No model weights were saved... '
                                 f'Maybe you specified wrong mode? Mode specidifed: "{self._mode}"')

    def on_epoch_end(self, epoch_num: int, logs: Mapping) -> None:
        if self._remaining_patience == 0:
            self._stopped_epoch = epoch_num
            self.trainer.is_training = False
            return

        if self._last_metric_value is None and self._best_metric_value is None:
            self._last_metric_value = logs[self._monitor]
            self._best_metric_value = logs[self._monitor]
            self._best_epoch = epoch_num
            return

        metric_value = logs[self._monitor]
        if self._is_improvement(metric_value):
            self._remaining_patience = self._patience

            if self._is_best(metric_value):
                self._best_metric_value = logs[self._monitor]
                self._best_model_weights = self.model.state_dict()
                self._best_epoch = epoch_num
        else:
            self.trainer._log(paint(f'Watch out, quality of {self._monitor} is decreasing!', 'orange'))
            self._remaining_patience -= 1

        self._last_metric_value = logs[self._monitor]