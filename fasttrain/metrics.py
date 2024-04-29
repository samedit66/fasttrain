import torch
import torch.nn.functional as F


def _check_type(y_pred: torch.Tensor, y_true: torch.Tensor) -> str:
    '''
    Tries to find out the type of classification by the shapes of
    predicted labels/classes and expected ones.
    `'_scores'` ending types indicate that thresholding needs to be applied.
    '''
    match (y_pred.shape, y_true.shape):
        case [n], [m] if n == m:
            if torch.equal(y_pred, y_pred**2) and torch.equal(y_true, y_true**2):
                return 'binary'
            elif torch.equal(y_true, y_true**2):
                return 'binary_scores'
            return 'multiclass'
        case [n, m], [k] if n == k:
            return 'multiclass_scores'
        case [n, m], [k, t] if n == k and m == t:
            if torch.equal(y_pred, y_pred**2) and torch.equal(y_true, y_true**2):
                return 'multilabel'
            return 'multilabel_scores'
        case _:
            raise ValueError("Can't guess the type of classification "
                             f'from y_pred (shape={y_pred.shape}) and '
                             f'y_true (shape={y_true.shape})'
                             )


def accuracy(y_pred: torch.Tensor,
             y_true: torch.Tensor,
             threshold: float = 0.5,
             average: str | None = "macro",
             multilabel: str = "hamming",
             ) -> torch.Tensor:
    type_ = _check_type(y_pred, y_true)
    
    if type_ in {'binary_scores', 'multilabel_scores'}:
        y_pred = (y_pred > threshold).float()
    elif type_ == 'multiclass_scores':
        y_pred = torch.argmax(y_pred, dim=1)

    if type_ in {'binary', 'binary_scores', 'multiclass', 'multiclass_scores'}:
        correct = (y_pred == y_true).view(-1)
    elif type_ in {'multilabel', 'multilabel_scores'}:
        if multilabel == "hamming":
            y_pred = torch.transpose(y_pred, 0, 1)
            y_true = torch.transpose(y_true, 0, 1)
            correct = y_pred == y_true 
        elif multilabel == "exact":
            correct = torch.all(y_pred == y_true, dim=-1)

    total = correct.shape[-1]
    accuracy_ = torch.sum(correct, dim=-1) / total
    if average == "macro":
        return torch.mean(accuracy_)
    return accuracy_


def precision(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        task: str = "binary",
        threshold: float | None = None,
        average: str | None = None
        ):

    assert task in {"binary", "multiclass", "multilabel"}, \
        'task must be "binary", "multiclass" or "multilabel"'

    assert threshold is None or 0 < threshold < 1, \
        "threshold must be either None or in (0; 1)"

    assert average is None or average in {"macro", "micro"}, \
        'average must be None, "macro" or "micro"'

    if task in {"binary", "multilabel"} and threshold is not None:
        y_pred = y_pred > threshold

    if task == "binary":
        pass # предварительной обработки не требуется
    elif task == "multiclass":
        if not torch.all(torch.logical_or(y_pred == 0, y_pred == 1)):
            y_pred = torch.argmax(y_pred, dim=1)

        y_pred = F.one_hot(y_pred)
        y_true = F.one_hot(y_true)

        y_pred = torch.transpose(y_pred, 0, 1)
        y_true = torch.transpose(y_true, 0, 1)
    elif task == "multilabel":
        y_pred = torch.transpose(y_pred, 0, 1)
        y_true = torch.transpose(y_true, 0, 1)

    tp = torch.sum(torch.logical_and(y_pred == y_true, y_true == 1), dim=-1)
    fp = torch.sum(torch.logical_and(y_pred == 1, y_true == 0), dim=-1)

    if average == "micro":
        sum_tp = torch.sum(tp)
        sum_fp = torch.sum(fp)
        micro_precision = torch.nan_to_num(sum_tp / (sum_tp + sum_fp), nan=0)
        return micro_precision
    elif average == "macro":
        precision_ = torch.nan_to_num(tp / (tp + fp), nan=0)
        macro_precision = torch.mean(precision_)
        return macro_precision
    elif average is None:
        precision_ = torch.nan_to_num(tp / (tp + fp), nan=0)
        return precision_
    

def recall(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        task: str = "binary",
        threshold: float | None = None,
        average: str | None = None
        ):

    assert task in {"binary", "multiclass", "multilabel"}, \
        'task must be "binary", "multiclass" or "multilabel"'

    assert threshold is None or 0 < threshold < 1, \
        "threshold must be either None or in (0; 1)"

    assert average is None or average in {"macro", "micro"}, \
        'average must be None, "macro" or "micro"'

    if task in {"binary", "multilabel"} and threshold is not None:
        y_pred = y_pred > threshold

    if task == "binary":
        pass # предварительной обработки не требуется
    elif task == "multiclass":
        if not torch.all(torch.logical_or(y_pred == 0, y_pred == 1)):
            y_pred = torch.argmax(y_pred, dim=1)

        y_pred = F.one_hot(y_pred)
        y_true = F.one_hot(y_true)

        y_pred = torch.transpose(y_pred, 0, 1)
        y_true = torch.transpose(y_true, 0, 1)
    elif task == "multilabel":
        y_pred = torch.transpose(y_pred, 0, 1)
        y_true = torch.transpose(y_true, 0, 1)

    tp = torch.sum(torch.logical_and(y_pred == y_true, y_true == 1), dim=-1)
    fn = torch.sum(torch.logical_and(y_pred == 0, y_true == 1), dim=-1)

    if average == "micro":
        sum_tp = torch.sum(tp)
        sum_fn = torch.sum(fn)
        micro_recall = torch.nan_to_num(sum_tp / (sum_tp + sum_fn), nan=0)
        return micro_recall
    elif average == "macro":
        recall_ = torch.nan_to_num(tp / (tp + fn), nan=0)
        macro_recall = torch.mean(recall_)
        return macro_recall
    elif average is None:
        recall_ = torch.nan_to_num(tp / (tp + fn), nan=0)
        return recall_
    

def fbeta(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        beta: float,
        task: str = "binary",
        threshold: float | None = None,
        average: str | None = None
        ):

    assert 0 <= beta <= 1, \
        "beta must be in [0; 1]"

    assert task in {"binary", "multiclass", "multilabel"}, \
        'task must be "binary", "multiclass" or "multilabel"'

    assert (threshold is None) or (0 < threshold < 1), \
        "threshold must be either None, dict or in (0; 1)"

    assert (average is None) or (average in {"macro", "micro"}), \
        'average must be None, "macro" or "micro"'


    precision_ = precision(y_pred, y_true, task, threshold, average)
    recall_ = recall(y_pred, y_true, task, threshold, average)
    fbeta_ = torch.nan_to_num((1 + beta**2) * precision_ * recall_ / (beta * precision_ + recall_), nan=0)

    if average == "macro":
        return torch.mean(fbeta_)
    elif average == "micro":
        raise NotImplementedError

    return fbeta_


def f1(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        task: str = "binary",
        threshold: float | None = None,
        average: str | None = None
        ):
    return fbeta(y_pred, y_true, beta=1.0, task=task, threshold=threshold, average=average)