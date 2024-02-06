from typing import Sequence

from torch import Tensor, stack
from torch.nn import (
    ModuleDict,
    Module,
    ReLU,
    Sequential,
    Linear,
    Dropout,
)


class MLP(Module):
    """
    Реализация многойслойного персептрона (далее - MLP).
    Поддерживает многоклассовую классификацию (multiclass) и
    классификацию по нескольким меткам (multilabel).

    Параметры:
    ----------
        input_size (int): длина входного тензора

        output_size (int, optional): длина выходного тензора, по умолчанию - 1

        is_multilabel (bool, optional): признак, что MLP реализует multilabel классификацию,
                                        по умолчанию - False

        hidden_layers (Sequence[int], optional): список количества нейронов в каждом скрытом слое,
                                                 по умолчанию - [] (скрытых нейронов нет)

        hidden_activation (Module, optional): функция активации в скрытых слоях, по умолчанию - ReLU

        dropout_rate (float, optional): коэффициент прореживания (дропаута),
                                        по умочанию - None (прореживание не применяется)

        output_activation (Module, optional): выходная функция активации,
                                              по умолчанию - не применяется (линейная функция активации)
    """

    def __init__(self,
                 input_size: int,
                 output_size: int = 1,
                 is_multilabel: bool = False,
                 hidden_layers: Sequence[int] = [],
                 hidden_activation: Module | None = ReLU(),
                 dropout_rate: float | None = None,
                 output_activation: Module | None = None) -> None:
        super().__init__()

        # TODO: проверка валидности входных параметров
        self._is_multilabel = is_multilabel
        self._mlp = self._build_mlp(input_size,
                                    output_size,
                                    is_multilabel,
                                    hidden_layers,
                                    hidden_activation,
                                    dropout_rate,
                                    output_activation)

    def _build_block(self,
                     input_size: int,
                     output_size: int,
                     dropout_rate: float | None = None,
                     activation: Module | None = None) -> Module:
        layers = []

        layers.append(Linear(input_size, output_size))
        if activation is not None:
            layers.append(activation)
        if dropout_rate is not None:
            layers.append(Dropout(dropout_rate))

        return layers

    def _build_perceptron(self,
                          input_size: int,
                          output_size: int,
                          hidden_layers: Sequence[int] = [],
                          dropout_rate: float | None = None,
                          hidden_activation: Module | None = None,
                          output_activation: Module | None = None) -> Module:
        layers = []

        sizes = [input_size, *hidden_layers]
        for i in range(len(sizes)-1):
            in_features, out_features = sizes[i:i+2]
            block = self._build_block(in_features,
                                     out_features,
                                     dropout_rate,
                                     hidden_activation)
            layers.extend(block)

        end_block = self._build_block(sizes[-1],
                                      output_size,
                                      None,
                                      output_activation)
        layers.extend(end_block)

        return Sequential(*layers)

    def _build_mlp(self,
                   input_size: int,
                   output_size: int,
                   is_multilabel: bool,
                   hidden_layers: Sequence[int] = [],
                   hidden_activation: Module | None = ReLU(),
                   dropout_rate: float | None = None,
                   output_activation: Module | None = None) -> Module:

        if is_multilabel:
            perceptrons = ModuleDict()

            labels = output_size
            for label_num in range(labels):
                perceptrons.update(
                    {f"Perceptron #{label_num}":
                        self._build_perceptron(input_size,
                                               1,
                                               hidden_layers,
                                               dropout_rate,
                                               hidden_activation,
                                               output_activation)}
                )

            return perceptrons

        return self._build_perceptron(input_size,
                                      output_size,
                                      hidden_layers,
                                      dropout_rate,
                                      hidden_activation,
                                      output_activation)

    def forward(self, input_batch: Tensor) -> Tensor:
        if self._is_multilabel:
            batch_size = input_batch.shape[0]
            preds = [perceptron(input_batch) for perceptron in self._mlp.values()]
            return stack(preds, dim=1).view(batch_size, -1)

        return self._mlp(input_batch)