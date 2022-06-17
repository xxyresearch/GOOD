r"""A metric function module that is consist of a Metric class which incorporate many score and loss functions.
"""

from math import sqrt

import torch
from sklearn.metrics import roc_auc_score as sk_roc_auc, mean_squared_error, \
    accuracy_score, average_precision_score, mean_absolute_error, f1_score
from torch.nn.functional import cross_entropy, l1_loss, binary_cross_entropy_with_logits





class Metric(object):


    def __init__(self):
        self.task2loss = {
            'Binary classification': binary_cross_entropy_with_logits,
            'Multi-label classification': self.cross_entropy_with_logit,
            'Regression': l1_loss
        }
        self.score_name2score = {
            'RMSE': self.rmse,
            'MAE': mean_absolute_error,
            'Average Precision': self.ap,
            'F1': self.f1,
            'ROC-AUC': self.roc_auc_score,
            'Accuracy': self.acc,
        }
        self.loss_func = self.cross_entropy_with_logit
        self.score_func = self.roc_auc_score
        self.dataset_task = ''
        self.score_name = ''

        self.lower_better = -1

        self.best_stat = {'score': None, 'loss': float('inf')}
        self.id_best_stat = {'score': None, 'loss': float('inf')}

    def set_loss_func(self, task_name):
        self.dataset_task = task_name
        self.loss_func = self.task2loss.get(task_name)
        assert self.loss_func is not None

    def set_score_func(self, metric_name):
        self.score_func = self.score_name2score.get(metric_name)
        assert self.score_func is not None
        self.score_name = metric_name.upper()
        if self.score_name in ['RMSE', 'MAE']:
            self.lower_better = 1
        else:
            self.lower_better = -1

    def f1(self, y_true, y_pred):
        true = torch.tensor(y_true)
        pred_label = torch.tensor(y_pred)
        pred_label = pred_label.round() if self.dataset_task == "Binary classification" else torch.argmax(pred_label,
                                                                                                            dim=1)
        return f1_score(true, pred_label, average='micro')

    def ap(self, y_true, y_pred):
        return average_precision_score(torch.tensor(y_true).long(), torch.tensor(y_pred))

    def roc_auc_score(self, y_true, y_pred):
        return sk_roc_auc(torch.tensor(y_true).long(), torch.tensor(y_pred), multi_class='ovo')

    def reg_absolute_error(self, y_true, y_pred):
        return mean_absolute_error(torch.tensor(y_true), torch.tensor(y_pred))

    def acc(self, y_true, y_pred):
        true = torch.tensor(y_true)
        pred_label = torch.tensor(y_pred)
        pred_label = pred_label.round() if self.dataset_task == "Binary classification" else torch.argmax(pred_label,
                                                                                                            dim=1)
        return accuracy_score(true, pred_label)

    def rmse(self, y_true, y_pred):
        return sqrt(mean_squared_error(y_true, y_pred))

    def cross_entropy_with_logit(self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
        return cross_entropy(y_pred, y_true.long(), **kwargs)

