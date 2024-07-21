import torch
import torch.nn.functional as F


def _check_type(y_pred: torch.Tensor, y_true: torch.Tensor) -> str:
    '''
    Tries to find out the type of classification by the shapes of
    predicted labels/classes and expected ones.
    `'_scores'` ending types indicate that "thresholding" or "argmaxing"
    needs to be applied.
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


def _preprocess_scores(type_: str, y_pred: torch.Tensor, threshold: float) -> torch.Tensor:
    if type_ in {'binary_scores', 'multilabel_scores'}:
        y_pred = (y_pred > threshold).float()
    elif type_ == 'multiclass_scores':
        y_pred = torch.argmax(y_pred, dim=-1)
    return y_pred


def accuracy(y_pred: torch.Tensor,
             y_true: torch.Tensor,
             threshold: float = 0.5,
             average: str = 'macro',
             multilabel: str = 'hamming',
             ) -> torch.Tensor:
    '''
    Accuracy metric. Supported types of computations are:
    1) Multiclass.
        Expects `y_pred` to be either classes indexes (0, 1, 2, ...) or logits,
        `y_true` to be classes indexes.
    2) Binary.
        Special case of multiclass when classes are 0 and 1.
        Like multiclass, expects `y_pred` to be either classes indexes or logits
        (when logits given, thresholding is applied).
    3) Multilabel.
        Expects `y_pred` to be either labels (0 or 1) or logits (same requirements as for binary).
    The type of accuracy score is determined on given values.
        
    :param y_pred: Predicted values, might be either indexes (classes, labels) or logits.
    :param y_true: Ground truth values.
    :param threshold: Threshold. Applied to binary or multilabel classification.
        Defaults to `0.5`.
    :param average: Type of averaging to done after computation (`'macro'`, `'none'`).
        Defaults to `'macro'`. `'macro'` means taking average of computated accuracies.
        When `'none'` no averaging is applied.
    :param multilabel: Type of multilabel classification. Might be `'hamming'` or `'exact'`.
        Defaults to `'hamming'`.
    :return: Computed accuracy.
    '''
    type_ = _check_type(y_pred, y_true)
    
    y_pred = _preprocess_scores(type_, y_pred, threshold)

    if type_ in {'binary', 'binary_scores', 'multiclass', 'multiclass_scores'}:
        correct = (y_pred == y_true).view(-1)
    elif type_ in {'multilabel', 'multilabel_scores'}:
        if multilabel == 'hamming':
            y_pred = y_pred.transpose(0, 1)
            y_true = y_true.transpose(0, 1)
            correct = y_pred == y_true 
        elif multilabel == 'exact':
            correct = torch.all(y_pred == y_true, dim=-1)

    total = correct.shape[-1]
    accuracy_ = torch.sum(correct, dim=-1) / total
    if average == 'macro':
        return torch.mean(accuracy_)
    return accuracy_


def _precision_recall_prepare(type_: str,
                              y_pred: torch.Tensor,
                              y_true: torch.Tensor,
                              threshold: float,
                              num_classes: int,
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    y_pred = _preprocess_scores(type_, y_pred, threshold)

    if type_ in {'multiclass', 'multiclass_scores'}:
        # Here we try to guess the number of classes.
        # The assumption is simple: we get the max class index and add one to it.
        # As long as a batch of either y_pred or y_true has all classes present,
        # it will guess right. However, when the batch size is too small,
        # this may fail because the lack of classes.
        # So, when possible, specify the number of classes directly. 
        if num_classes == -1:
            num_classes = int(max(torch.max(y_pred), torch.max(y_true))) + 1
        y_pred = F.one_hot(y_pred, num_classes=num_classes).transpose(0, 1)
        y_true = F.one_hot(y_true, num_classes=num_classes).transpose(0, 1)
    elif type_ in {'multilabel', 'multilabel_scores'}:
        y_pred = y_pred.transpose(0, 1)
        y_true = y_true.transpose(0, 1)

    return y_pred, y_true


def confusion_matrix(y_pred: torch.Tensor,
                     y_true: torch.Tensor,
                     threshold: float = 0.5,
                     num_classes: int = -1,
                     ):
    type_ = _check_type(y_pred, y_true)

    y_pred, y_true = _precision_recall_prepare(type_=type_,
                                               y_pred=y_pred,
                                               y_true=y_true,
                                               threshold=threshold,
                                               num_classes=num_classes,
                                               )

    tp = torch.sum(torch.logical_and(y_pred == 1, y_true == 1), dim=-1)
    fp = torch.sum(torch.logical_and(y_pred == 1, y_true == 0), dim=-1)
    tn = torch.sum(torch.logical_and(y_pred == 0, y_true == 0), dim=-1)
    fn = torch.sum(torch.logical_and(y_pred == 0, y_true == 1), dim=-1)

    if type_ in {'multiclass', 'multiclass_scores', 'binary', 'binary_scores'}:
        ...

    return tp, fp, tn, fn


def precision(y_pred: torch.Tensor,
              y_true: torch.Tensor,
              threshold: float = 0.5,
              num_classes: int = -1,
              average: str = 'macro',
              ) -> torch.Tensor:
    type_ = _check_type(y_pred, y_true)

    y_pred, y_true = _precision_recall_prepare(type_=type_,
                                               y_pred=y_pred,
                                               y_true=y_true,
                                               threshold=threshold,
                                               num_classes=num_classes,
                                               )

    tp, fp, tn, fn = confusion_matrix(y_pred, y_true)

    if average == 'micro':
        sum_tp = torch.sum(tp)
        sum_fp = torch.sum(fp)
        micro_precision = torch.nan_to_num(sum_tp / (sum_tp + sum_fp), nan=0)
        return micro_precision
    elif average == 'macro':
        precision_ = torch.nan_to_num(tp / (tp + fp), nan=0)
        macro_precision = torch.mean(precision_)
        return macro_precision
    elif average == 'none':
        precision_ = torch.nan_to_num(tp / (tp + fp), nan=0)
        return precision_
    

def recall(y_pred: torch.Tensor,
           y_true: torch.Tensor,
           threshold: float = 0.5,
           num_classes: int = -1,
           average: str = 'macro',
           ) -> torch.Tensor:
    type_ = _check_type(y_pred, y_true)

    y_pred, y_true = _precision_recall_prepare(type_=type_,
                                               y_pred=y_pred,
                                               y_true=y_true,
                                               threshold=threshold,
                                               num_classes=num_classes,
                                               )

    tp = torch.sum(torch.logical_and(y_pred == y_true, y_true == 1), dim=-1)
    fn = torch.sum(torch.logical_and(y_pred == 0, y_true == 1), dim=-1)

    if average == 'micro':
        sum_tp = torch.sum(tp)
        sum_fn = torch.sum(fn)
        micro_recall = torch.nan_to_num(sum_tp / (sum_tp + sum_fn), nan=0)
        return micro_recall
    elif average == 'macro':
        recall_ = torch.nan_to_num(tp / (tp + fn), nan=0)
        macro_recall = torch.mean(recall_)
        return macro_recall
    elif average == 'none':
        recall_ = torch.nan_to_num(tp / (tp + fn), nan=0)
        return recall_
    

def fbeta(y_pred: torch.Tensor,
          y_true: torch.Tensor,
          beta: float,
          threshold: float = 0.5,
          num_classes: int = -1,
          average: str = 'macro',
          ) -> torch.Tensor:
    type_ = _check_type(y_pred, y_true)

    if type_ in {'binary_scores', 'multilabel_scores'}:
        y_pred = (y_pred > threshold).float()
    elif type_ == 'multiclass_scores':
        y_pred = torch.argmax(y_pred, dim=1)

    if type_ in {'multiclass', 'multiclass_scores'}:
        y_pred = F.one_hot(y_pred).transpose(0, 1)
        y_true = F.one_hot(y_true).transpose(0, 1) 
    elif type_ in {'multilabel', 'multilabel_scores'}:
        y_pred = y_pred.transpose(0, 1)
        y_true = y_true.transpose(0, 1)

    precision_ = precision(y_pred,
                           y_true,
                           threshold=threshold,
                           num_classes=num_classes,
                           average=average,
                           )
    recall_ = recall(y_pred,
                     y_true,
                     threshold=threshold,
                     num_classes=num_classes,
                     average=average,
                     )
    fbeta_ = torch.nan_to_num((1 + beta**2) * precision_ * recall_ / (beta * precision_ + recall_), nan=0)

    if average == 'macro':
        return torch.mean(fbeta_)
    elif average == 'micro':
        tp = torch.sum(torch.logical_and(y_pred == y_true, y_true == 1), dim=-1)
        fp = torch.sum(torch.logical_and(y_pred == 1, y_true == 0), dim=-1)
        fn = torch.sum(torch.logical_and(y_pred == 0, y_true == 1), dim=-1)
        return torch.nan_to_num(tp / (tp + 0.5*(fp + fn)), 0)
    elif average == 'none':
        return fbeta_
    

def f1(y_pred: torch.Tensor,
       y_true: torch.Tensor,
       threshold: float = 0.5,
       num_classes: int = -1,
       average: str = 'macro',
       ) -> torch.Tensor:
    return fbeta(y_pred, y_true, beta=1.0, threshold=threshold, num_classes=num_classes, average=average)
