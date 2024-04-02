from abc import ABC
from collections.abc import Mapping


class Callback(ABC):
    '''
    Base class used to build new callbacks.

    Callbacks can be passed to `Trainer()` constructor in order to
    hook into the various stages of the model training and validation.

    To create a custom callback, subclass `fasttrain.callbacks.Callback` and
    override the method associated with the stage of interest.

    Callbacks have access to the trainer, the model and training arguments.
    '''

    def __init__(self) -> None:
        self._trainer = None
        self._model = None
        self._training_args = None

    @property
    def trainer(self):
        '''
        Returns the trainer associated with the callback.

        :return: The `Trainer` associated with the callback.
        '''
        return self._trainer
    
    @trainer.setter
    def trainer(self, new_trainer):
        if self._trainer is not None:
            raise RuntimeError('Trainer can be set only once')
        self._trainer = new_trainer

    @property
    def model(self):
        '''
        Returns the model associated with the callback.

        :return: The model associated with the callback.
        '''
        return self._model
    
    @model.setter
    def model(self, new_model) -> None:
        if self._model is not None:
            raise RuntimeError('Model can be set only once')
        self._model = new_model

    @property
    def training_args(self) -> Mapping:
        '''
        Returns training arguments. Currently, they include the number of epochs
        and the batch size. Example: `self.training_args['epochs_num'], self.training_args['batch_size']`.

        :return: Training arguments.
        '''
        return self._training_args
    
    @training_args.setter
    def training_args(self, new_training_args: Mapping) -> Mapping:
        if self._training_args is not None:
            raise RuntimeError('Training arguments can be set only once')
        self._training_args = new_training_args

    def on_train_begin(self) -> None:
        '''
        Called at the beginning of training.
        '''

    def on_train_end(self, logs: Mapping) -> None:
        '''
        Called at the end of training.

        :param logs: Mapping. Currently the output of the last call to `on_epoch_end()`
            is passed to this argument, but that may change in the future.
        '''

    def on_epoch_begin(self, epoch_num: int) -> None:
        '''
        Called at the beginning of an epoch.

        :param epoch_num: Current epoch number.
        '''

    def on_epoch_end(self, epoch_num: int, logs: Mapping) -> None:
        '''
        Called at the end of an epoch.

        :param epoch_num: Current epoch number.
        :param logs: Mapping, metric results for this training epoch, and for the
            validation epoch if validation is performed. Validation result
            keys are prefixed with `val_`. Example: `{'loss': 0.2, 'accuracy': 0.7, 'val_loss': 0.21, 'val_accuracy': 0.74}`.
        '''

    def on_train_batch_begin(self, batch_num: int) -> None:
        '''
        Called at the beginning of a training batch.

        :param batch_num: Current batch number.
        '''

    def on_train_batch_end(self, batch_num: int, logs: Mapping) -> None:
        '''
        Called at the end of a training batch.

        :param batch_num: Current batch number.
        :param logs: Mapping. Aggregated metric results up until this batch.
        '''

    def on_validation_begin(self) -> None:
        '''
        Called at the beginning of validation.
        '''

    def on_validation_end(self, logs: Mapping) -> None:
        '''
        Called at the end of validation.

        :param logs: Mapping. Currently the output of the last call to
            `on_validation_batch_end()` is passed to this argument
            but that may change in the future.
        '''

    def on_validation_batch_begin(self, batch_num: int) -> None:
        '''
        Called at the beginning of a validation batch.

        :param batch_num: Current batch number.
        '''

    def on_validation_batch_end(self, batch_num: int, logs: Mapping) -> None:
        '''
        Called at the end of a validation batch.

        :param batch_num: Current batch number.
        :param logs: Mapping. Aggregated metric results up until this batch. 
        '''