import torch

from . import Callback


class Checkpoint(Callback):
    '''
    Callback to save the model parameters (`state_dict`) at some frequency.

    This callback is used to save model parameters at some interval,
    so they can be loaded later to continue the training from the state saved.

    :param file_path: str or `PathLike`, path to save the model file.
        `file_path` can contain named formatting options, which will
        be filled with calue of `epoch_num` and keys in `logs`.
        For example: if `file_path` is `"{epoch_num:02d}-{val_loss:.2f}.pth"`, then
        the model checkpoints will be saved with the epoch number and the
        validation loss in the filename.
    :param monitor: Metric to monitor (defaults to `"val_loss"`). When `None`,
        the model parameters will be saved at every epoch.
    :param save_best_only: if `True`, it only saves the one "best" model. 
    :param mode: One of `{"min", "max"}`. If `save_best_only=True`, the
        decision to overwrite the current save file is made based on either
        the maximization or the minimization of the monitored quantity.
        For `val_acc`, this should be `"max"`, for `val_loss` this should be
        `"min"`, etc.
    :param threshold: Floating point initial "best" value of the
        metric to be monitored. Only applies if `save_best_value=True`.
    '''

    def __init__(self,
                 file_path: str,
                 monitor: str | None = 'val_loss',
                 save_best_only: bool = True,
                 mode: str = 'min',
                 threshold: float | None = None,
                 ) -> None:
        self._file_path = file_path
        self._monitor = monitor
        self._save_best_only = save_best_only
        self._mode = mode
        self._threshold = threshold
        self._best_model_weights = None
        self._best_metric_value = None
        self._best_epoch = None

    def _is_best(self, metric_value):
        if self._mode == 'min':
            is_better = metric_value <= self._best_metric_value
            if self._threshold:
                is_better = is_better and (metric_value <= self._threshold)
            return is_better
        elif self._mode == 'max':
            is_better = metric_value >= self._best_metric_value
            if self._threshold:
                is_better = is_better and (metric_value >= self._threshold)
            return is_better

    def _save_model(self):
        if self._monitor is None:
            file_path = self._file_path
        else:
            file_path = self._file_path.format(epoch_num=self._best_epoch, **self._best_metrics)
        torch.save(self.model.state_dict(), file_path)

    def on_train_end(self, logs={}):
        if self._best_model_weights is not None:
            self.trainer._log(f'Saving best model weights from the end of the best epoch: {self._best_epoch}.')
            self._save_model()

    def on_epoch_end(self, epoch_num, logs={}):
        if (self._monitor is not None) and (logs.get(self._monitor) is None):
            raise ValueError(f'Expected metric to monitor "{self._monitor}" not found')

        if self._monitor is None:
            self._save_model()
            return

        if self._best_metric_value is None:
            self._best_model_weights = self.model.state_dict()
            self._best_metric_value = logs[self._monitor]
            self._best_epoch = epoch_num
            self._best_metrics = logs
            return
        
        metric_value = logs[self._monitor]
        if self._save_best_only:
            if self._is_best(metric_value):
                self._best_model_weights = self.model.state_dict()
                self._best_metric_value = metric_value
                self._best_epoch = epoch_num
                self._best_metrics = logs
            else:
                # TODO: Печатать сообщение о том, что метрика перестала увеличиваться?
                pass
        else: 
            self._save_model(epoch_num=self._best_epoch,
                             name=self._monitor,
                             value=self._best_metric_value,
                             )