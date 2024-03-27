from abc import ABC
import collections.abc

import torch


class Callback(ABC):
    '''
    Base class used to build new callbacks.

    Callbacks can be passed to `Trainer()` constructor in order to
    hook into the various stages of the model training and validation.

    To create a custom callback, subclass `fasttrain.callbacks.Callback` and
    override the method associated with the stage of interest.
    '''

    def __init__(self) -> None:
        self._training_params = None

    @property
    def training_params(self) -> collections.abc.Mapping:
        '''
        Training parameters. Mapping. Includes the number or epochs and
        the number of training batches.
        Example: `self.training_params['num_epochs'], self.training_params['num_batches']`.
        '''
        if self._training_params is None:
            raise RuntimeError("`training_params` are not initialized")
        return self._training_params
    
    @training_params.setter
    def training_params(self, training_params: collections.abc.Mapping) -> None:
        '''
        Sets up the training parameters. Does it only once.
        Raises exceptions either when trying to set the training params twice
        or when `training_params` are not a mapping.

        :param training_params: Mapping.
        :raises RuntimeError: When trying to set the training params twice or more.
        :raises TypeError: When given `training_params` are not a `Mapping`.
        '''
        if self._training_params is not None:
            raise RuntimeError("`training_params` can't be changed")
        if not isinstance(training_params, collections.abc.Mapping):
            raise TypeError("`training_params` must be a mapping")
        self._training_params = training_params

    def on_train_begin(self,
                       trainer,
                       model: torch.nn.Module,
                       ) -> None:
        '''
        Called at the beginning of training.
        
        :param trainer: Trainer.
        :param model: Training model.
        '''

    def on_train_end(self,
                     trainer,
                     model: torch.nn.Module,
                     logs: collections.abc.Mapping,
                     ) -> None:
        '''
        Called at the end of training.

        :param trainer: Trainer.
        :param model: Training model.
        :param logs: Mapping. Currently the output of the last call to `on_epoch_end()`
            is passed to this argument, but that may change in the future.
        '''

    def on_epoch_begin(self,
                       trainer,
                       model: torch.nn.Module,
                       epoch_num: int,
                       ) -> None:
        '''
        Called at the beginning of an epoch.

        :param trainer: Trainer.
        :param model: Training model.
        :param epoch_num: Current epoch number.
        '''

    def on_epoch_end(self,
                     trainer,
                     model: torch.nn.Module,
                     epoch_num: int,
                     logs: collections.abc.Mapping,
                     ) -> None:
        '''
        Called at the end of an epoch.

        :param trainer: Trainer.
        :param model: Training model.
        :param epoch_num: Current epoch number.
        :param logs: Mapping, metric results for this training epoch, and for the
            validation epoch if validation is performed. Validation result
            keys are prefixed with `val_`. Example: `{'loss': 0.2, 'accuracy': 0.7, 'val_loss': 0.21, 'val_accuracy': 0.74}`.
        '''

    def on_train_batch_begin(self,
                             trainer,
                             model: torch.nn.Module,
                             batch_num: int,
                             ) -> None:
        '''
        Called at the beginning of a training batch.

        :param trainer: Trainer.
        :param model: Training model.
        :param batch_num: Current batch number.
        '''

    def on_train_batch_end(self,
                           trainer,
                           model: torch.nn.Module,
                           batch_num: int,
                           logs: collections.abc.Mapping,
                           ) -> None:
        '''
        Called at the end of a training batch.

        :param trainer: Trainer.
        :param model: Training model.
        :param batch_num: Current batch number.
        :param logs: Mapping. Aggregated metric results up until this batch.
        '''

    def on_validation_begin(self,
                            trainer,
                            model: torch.nn.Module,
                            ) -> None:
        '''
        Called at the beginning of validation.

        :param trainer: Trainer.
        :param model: Training model.
        '''

    def on_validation_end(self,
                          trainer,
                          model: torch.nn.Module,
                          logs: collections.abc.Mapping,
                          ) -> None:
        '''
        Called at the end of validation.

        :param trainer: Trainer.
        :param model: Training model.
        :param logs: Mapping. Currently the output of the last call to
            `on_validation_batch_end()` is passed to this argument
            but that may change in the future.
        '''

    def on_validation_batch_begin(self,
                                  trainer,
                                  model: torch.nn.Module,
                                  batch_num: int,
                                  ) -> None:
        '''
        Called at the beginning of a validation batch.

        :param trainer: Trainer.
        :param model: Training model.
        :param batch_num: Current batch number.
        '''

    def on_validation_batch_end(self,
                                trainer,
                                model: torch.nn.Module,
                                batch_num: int,
                                logs: collections.abc.Mapping,
                                ) -> None:
        '''
        Called at the end of a validation batch.

        :param trainer: Trainer.
        :param model: Training model.
        :param batch_num: Current batch number.
        :param logs: Mapping. Aggregated metric results up until this batch. 
        '''