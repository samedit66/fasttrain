from abc import ABC

from train.trainer import Trainer


class Callback(ABC):
    
    def on_train_begin(self,
                       trainer: Trainer,
                       logs: dict | None = None
                       ) -> None:
        ...

    def on_train_end(self,
                     trainer: Trainer,
                     logs: dict | None = None
                     ) -> None:
        ...

    def on_epoch_begin(self,
                       epoch_idx: int,
                       trainer: Trainer,
                       logs: dict | None = None
                       ) -> None:
        ...

    def on_epoch_end(self,
                     epoch_idx: int,
                     trainer: Trainer,
                     logs: dict | None = None
                     ) -> None:
        ...

    def on_train_batch_begin(self,
                             batch_idx: int,
                             trainer: Trainer,
                             logs: dict | None = None
                             ) -> None:
        ...

    def on_train_batch_end(self,
                           batch_idx: int,
                           trainer: Trainer,
                           logs: dict | None = None
                           ) -> None:
        ...

    def on_validation_begin(self,
                            trainer: Trainer,
                            logs: dict | None = None
                            ) -> None:
        ...

    def on_validation_end(self,
                          trainer: Trainer,
                          logs: dict | None = None
                          ) -> None:
        ...

    def on_validation_batch_begin(self,
                                  batch_idx: int,
                                  trainer: Trainer,
                                  logs: dict | None = None
                                  ) -> None:
           ...

    def on_validation_batch_end(self,
                                batch_idx: int,
                                trainer: Trainer,
                                logs: dict | None = None
                                ) -> None:
        ...