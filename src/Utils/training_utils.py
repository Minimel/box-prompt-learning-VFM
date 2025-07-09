"""
Author: Mélanie Gaillochet
"""
import numpy as np
import math
import torch
from monai import metrics as monai_metrics

from torch.optim import lr_scheduler

from Utils.utils import to_onehot

############################# METRICS #############################
dice_metric = monai_metrics.DiceMetric(include_background=False, reduction='none', get_not_nans=False, ignore_empty=False)
assd_metric = monai_metrics.SurfaceDistanceMetric(include_background=False, symmetric=True, distance_metric='euclidean', reduction='none', get_not_nans=False)

metrics = {
    'dice': dice_metric,
    'assd': assd_metric
}


def convert_inf_to_max_distance(assd, shape):
    """
    Replace an infinite ASSD value with the maximum Euclidean distance for the image.

    Parameters:
    - assd: float, the computed ASSD value (may be np.inf)
    - shape: tuple, the image dimensions (H, W)

    Returns:
    - float: the original ASSD if it's finite, or the maximum Euclidean distance if ASSD is infinite.
    """
    _, _, H, W = shape
    max_distance = np.sqrt(H**2 + W**2)
    return max_distance if np.isinf(assd) else assd


def convert_nan_to_max_distance(hd, shape):
    """
    Replace an nan Hausdorff distance value with the maximum Euclidean distance for the image.

    Parameters:
    - assd: float, the computed ASSD value (may be np.inf)
    - shape: tuple, the image dimensions (H, W)

    Returns:
    - float: the original HD if it's finite, or the maximum Euclidean distance if HD is Nan.
    """
    _, _, H, W = shape
    max_distance = np.sqrt(H**2 + W**2)
    return max_distance if math.isnan(hd) else hd


def mean_excluding_nan(x):
    if x.isnull().any():
        return None
    else:
        return x.mean()

def compute_seg_metrics(pred, y, out_channels=2):
    """
    Computing typical segmentation metrics: dice, iou and hausdorff95, both total and per class
    args:
        pred: (B, C, H, W) or (B, H, W) tensor
        y: (B, C, H, W) or (B, H, W) tensor
    """
    onehot_pred = to_onehot(pred.squeeze(1), out_channels)
    onehot_target = to_onehot(y.squeeze(1), out_channels)
    _dice = dice_metric(onehot_pred, onehot_target)
    dice = torch.mean(_dice)

    assd = torch.mean(assd_metric(onehot_pred, onehot_target))

    metrics = {
        'dice': 100*dice,
        'assd': assd
        }
    return metrics


def mean_dice_per_channel(predictions, onehot_target, eps=1e-9,
                          global_dice=False,
                          reduction='mean'):
    """
    We compute the dice, averaged for each channel
    :pa
    """

    dice_dic = {}
    for c in range(1, onehot_target.shape[1]):
        # We select only the predictions and target for the given class
        _selected_idx = torch.tensor([c])
        selected_idx = _selected_idx.to(predictions.get_device())
        pred = torch.index_select(predictions, 1, selected_idx)
        tg = torch.index_select(onehot_target, 1, selected_idx)

        # For each channel, we compute the mean dice
        dice = compute_dice(pred, tg, eps=eps, global_dice=global_dice,
                            reduction=reduction)
        dice_dic['dice_{}'.format(c)] = dice

    return dice_dic


def compute_dice(pred, tg, eps=1e-9, global_dice=False, reduction='mean'):
    """
    We compute the dice for a 3d image
    :param pred: normalized tensor (3d) [BS, x, y, z]
    :param target: tensor (3d) [BS, x, y, z]
    :param eps:
    :param normalize_fct:
    :param weighted:
    :return:
    """
    # if we compute the global dice then we will sum over the batch dim,
    # otherwise no
    if global_dice:
        dim = list(range(0, len(pred.shape)))
    else:
        dim = list(range(1, len(pred.shape)))

    intersect = torch.sum(pred * tg, dim=dim)
    union = pred.sum(dim=dim) + tg.sum(dim=dim)
    dice = (2. * intersect + eps) / (union + eps)

    if reduction == 'mean':
        # We average over the number of samples in the batch
        dice = dice.mean()

    return dice


"""from https://github.com/jizongFox/deepclustering2/blob/master/deepclustering2/schedulers/warmup_scheduler.py"""
class GradualWarmupScheduler(lr_scheduler._LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.0:
            raise ValueError("multiplier should be greater than 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [
            base_lr
            * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
            for base_lr in self.base_lrs
        ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = (
            epoch if epoch != 0 else 1
        )  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != lr_scheduler.ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
