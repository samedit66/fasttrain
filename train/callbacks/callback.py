from abc import ABC

import collections.abc


class Callback(ABC):
    '''
    Base class used to build new callbacks.

    Callbacks can be passed to `Trainer.train()` method in order to
    hook into the various stages of the model training and validation.

    To create a custom callback, subclass `fasttrain.callbacks.Callback` and
    override the method associated with the stage of interest.

    To access the trainer, one should use `self.trainer`.
    To access the training model, one should use `self.trainer.model`.
    '''

    def __init__(self) -> None:
        self._trainer = None
        self._model = None
        self._training_params = None

    @property
    def trainer(self):
        return self._trainer
    
    @trainer.setter
    def trainer(self, new_trainer):
        self._trainer = new_trainer

    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, new_model):
        self._model = new_model

    @property
    def training_params(self):
        return self._training_params
    
    @training_params.setter
    def training_params(self, new_training_params):
        self._training_params = new_training_params

    def on_train_begin(self,
                       logs: collections.abc.Mapping | None = None
                       ) -> None:
        '''
        Called at the beginning of training.
        
        :param logs: Mapping. Currently no data is passed to this argument,
            but that may change in the future.
        '''

    def on_train_end(self,
                     logs: collections.abc.Mapping | None = None
                     ) -> None:
        '''
        Called at the end of training.

        :param logs: Mapping. Currently the output of the last call to `on_epoch_end()`
            is passed to this argument, but that may change in the future.
        '''

    def on_epoch_begin(self,
                       epoch_num: int,
                       logs: collections.abc.Mapping | None = None
                       ) -> None:
        '''
        Called at the beginning of an epoch.

        :param epoch_num: Current epoch number
        :param logs: Mapping. Currently no data is passed to this argument,
            but that may change in the future.
        '''

    def on_epoch_end(self,
                     epoch_num: int,
                     logs: collections.abc.Mapping | None = None
                     ) -> None:
        '''
        Called at the end of an epoch.

        :param epoch_num: Current epoch number
        :param logs: Mapping, metric results for this training epoch, and for the
            validation epoch if validation is performed. Validation result
            keys are prefixed with `val_`. Example: `{'loss': 0.2, 'accuracy': 0.7}`.
        '''

    def on_train_batch_begin(self,
                             batch_num: int,
                             logs: collections.abc.Mapping | None = None
                             ) -> None:
        '''
        Called at the beginning of a training batch.

        :param batch_num: Current batch number
        :param logs: Mapping. Currently no data is passed to this argument,
            but that may change in the future.
        '''

    def on_train_batch_end(self,
                           batch_num: int,
                           logs: collections.abc.Mapping | None = None
                           ) -> None:
        '''
        Called at the end of a training batch.

        :param batch_num: Current batch number
        :param logs: Mapping. Aggregated metric results up until this batch.
        '''

    def on_validation_begin(self,
                            logs: collections.abc.Mapping | None = None
                            ) -> None:
        '''
        Called at the beginning of validation.

        :param logs: Mapping. Currently no data is passed to this argument,
            but that may change in the future.
        '''

    def on_validation_end(self,
                          logs: collections.abc.Mapping | None = None
                          ) -> None:
        '''
        Called at the end of validation.

        :param logs: Mapping. Currently the output of the last call to
            `on_validation_batch_end()` is passed to this argument
            but that may change in the future.
        '''

    def on_validation_batch_begin(self,
                                  batch_num: int,
                                  logs: collections.abc.Mapping | None = None
                                  ) -> None:
        '''
        Called at the beginning of a validation batch.

        :param batch_num: Current batch number
        :param logs: Mapping. Currently no data is passed to this argument,
            but that may change in the future.
        '''

    def on_validation_batch_end(self,
                                batch_num: int,
                                logs: collections.abc.Mapping | None = None
                                ) -> None:
        '''
        Called at the end of a validation batch.

        :param batch_num: Current batch number
        :param logs: Mapping. Aggregated metric results up until this batch. 
        '''