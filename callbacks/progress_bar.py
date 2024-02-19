from sys import stderr

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from callbacks import Callback
from train._utils import format_metrics


class Tqdm(Callback):
    '''
    Callback which prints a pretty-looking progress bar for training.
    Output includes training metrics and, if present, validation metrics.

    :param outer_desc: Description of the main training progress bar.
    :param inner_desc_initial: Initial format for epoch (defaults to `"Epoch {epoch_num}"`).
    :param inner_desc_update: Format after metrics collected (defaults to `"Epoch: {epoch_num} - {metrics}"`).
    :param metric_format: Format for each metric name/value pait (defaults to `"{name}: {value:0.3f}"`).
    :param sep: Separator between metrics (defaults to `","`)
    :param leave_inner: `True` to leave inner bars (defaults to `False`).
    :param leave_outer: `True` to leave outer bars (defaults to `True`).
    :param show_inner: `False` to hide inner bars (defaults to `True`).
    :param show_outer: `False` to hide outer bars (defaults to `True`).
    :param output_file: Output file (defaults to `sys.stderr`).
    :param initial: Initial counter state (defaults to `0`).
    :param colab: Set to `True` when in Google Colab (defaults to `False`), if `False` when
        in Google Colab the progress bar may be strange looking.
    '''

    def __init__(self,
                 outer_desc: str = 'Training',
                 inner_desc_initial: str = 'Epoch: {epoch_num}',
                 inner_desc_update: str = 'Epoch: {epoch_num} - {metrics}',
                 metric_format: str = "{name}: {value:0.3f}",
                 sep: str = ', ',
                 leave_inner: bool = True,
                 leave_outer: bool = True,
                 show_inner: bool = True,
                 show_outer: bool = True,
                 output_file = stderr,
                 initial: int = 0,
                 colab: bool = False,
                 ) -> None:
        self._outer_desc = outer_desc
        self._inner_desc_initial = inner_desc_initial
        self._inner_desc_update = inner_desc_update
        self._metric_format = metric_format
        self._sep = sep
        self._leave_inner = leave_inner
        self._leave_outer = leave_outer
        self._show_inner = show_inner
        self._show_outer = show_outer
        self._output_file = output_file
        self._initial = initial
        self._colab = colab
        self._current_epoch_num = 0
        self._tqdm_outer = None
        self._tqdm_inner = None
        self._inner_count = 0
        self._inner_total = 0

    def format_metrics(self, logs):
        return format_metrics(logs,
                              metric_format=self._metric_format,
                              sep=self._sep,
                              with_color=(not self._colab))

    def _tqdm(self, desc, total, leave, initial=0):
        tqdm_ = tqdm_notebook if self._colab else tqdm

        return tqdm_(desc=desc,
                    total=total,
                    leave=leave,
                    file=self._output_file,
                    initial=initial,
                    )
    
    def _build_tqdm_outer(self, desc, total):
        return self._tqdm(desc=desc,
                          total=total,
                          leave=self._leave_outer,
                          initial=self._initial,
                          )
    
    def _build_tqdm_inner(self, desc, total):
        return self._tqdm(desc=desc,
                          total=total,
                          leave=self._leave_inner,
                          )

    def on_train_begin(self, logs={}):
        if self._show_outer:
            num_epochs = self.training_params['num_epochs']
            self._tqdm_outer = self._build_tqdm_outer(desc=self._outer_desc, total=num_epochs)

    def on_train_end(self, logs={}):
        if self._show_outer:
            self._tqdm_outer.close()

    def on_epoch_begin(self, epoch_num, logs={}):
        self._current_epoch_num = epoch_num
        desc = self._inner_desc_initial.format(epoch_num=epoch_num)
        self._inner_total = self.training_params['num_batches']
        if self._show_inner:
            self._tqdm_inner = self._build_tqdm_inner(desc=desc, total=self._inner_total)
        self._inner_count = 0

    def on_epoch_end(self, epoch_num, logs={}):
        metrics = self.format_metrics(logs)
        desc = self._inner_desc_update.format(epoch_num=epoch_num, metrics=metrics)
        if self._show_inner:
            self._tqdm_inner.desc = desc
            # Set miniters and mininterval to 0 so last update displays
            self._tqdm_inner.miniters = 0
            self._tqdm_inner.mininterval = 0
            self._tqdm_inner.update(self._inner_total - self._tqdm_inner.n)
            self._tqdm_inner.close()
        if self._show_outer:
            self._tqdm_outer.update(1)

    def on_train_batch_end(self, batch_num, logs={}):
        update = 1
        self._inner_count += update
        if self._inner_count < self._inner_total:
            metrics = self.format_metrics(logs)
            desc = self._inner_desc_update.format(epoch_num=self._current_epoch_num, metrics=metrics)
            if self._show_inner:
                self._tqdm_inner.desc = desc
                self._tqdm_inner.update(update)