"""
Author: Mélanie Gaillochet
"""
from typing import List, cast, Dict, Union, cast
import torch
import torch.nn as nn
from torch import Tensor

from Utils.utils import to_onehot, get_nested_value
from Utils.training_utils import mean_dice_per_channel


############ Penalty functions ############
class LogBarrierPenalty(nn.Module):
    def __init__(self, **kwargs): #t: float = 5, epoch_multiplier: Union[int, float] = 1.1):
        super().__init__()
        t: float = kwargs.get('t', 5.0)
        multiplier: Union[int, float] = get_nested_value(kwargs, 'scheduler', 'multiplier', default=1.1)
        update_frequency: Union[int, float] = get_nested_value(kwargs, 'scheduler', 'update_frequency', default=1)
        self.register_buffer('t', torch.as_tensor(t))
        self.register_buffer('multiplier', torch.as_tensor(multiplier))
        self.register_buffer('update_frequency', torch.as_tensor(update_frequency))
        self.register_buffer('ceil', -1 / self.t ** 2)
        self.register_buffer('b', -torch.log(1 / (self.t**2)) / self.t + 1 / self.t)
        
    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return torch.where(
            cast(torch.Tensor, z <= self.ceil),
            - torch.log(-z) / self.t,
            self.t * z + self.b,
        )
        
    def step(self):
        self.t *= self.multiplier
        self.ceil[...] = -1 / self.t ** 2
        self.b[...] = -torch.log(1 / (self.t**2)) / self.t + 1 / self.t
        
        return self.t
        

class ReLUPenalty(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return nn.ReLU(inplace=False)(z)


class LeakyReLUPenalty(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return nn.LeakyReLU(inplace=False)(z)


penalty_zoo = {'log_barrier': LogBarrierPenalty,
               'relu': ReLUPenalty,
               'leaky_relu': LeakyReLUPenalty
               }

############ Losses ############
class DiceLoss(nn.Module):
    """
    This loss is based on the mean dice computed over all channels
    """

    def __init__(self, **kwargs):
        super().__init__()
        # print('Using {} normalization function'.format(normalize_fct))
        self.reduction = kwargs.get('reduction', 'mean')
        self.global_dice = False

    def forward(self, prob, onehot_target, eps=1e-9):

        dice_dic = mean_dice_per_channel(prob, onehot_target,
                                         global_dice=self.global_dice,
                                         reduction=self.reduction, eps=eps)
        mean_dice = sum(dice_dic.values()) / len(dice_dic)

        loss = 1 - mean_dice

        return loss, None
    
    
class WBCE_Dice(nn.Module):
    def __init__(self, **kwargs):
        """
        A combination of Weight Binary Cross Entropy and Binary Dice Loss
        """
        super().__init__()
        self.target_str: str = kwargs["target_str"] # ie: 'label', which refers to segmentation mask
        self.idc: List[int] = kwargs["idc"] #ie: 1 
        self.alpha_CE: float = kwargs.get('alpha_CE', 0.5)
        self.reduction: str = kwargs.get('reduction')

        assert 0 <= self.alpha_CE <= 1, '`alpha` should in [0,1]'
        
        self.dice_fct = DiceLoss(**{'reduction': self.reduction})

    def __call__(self, probs: Tensor, batch: Dict[str, Tensor], eps=1e-8, **kwargs) -> Tensor: 
        target = batch[self.target_str].squeeze(1).to(torch.float)
        onehot_target = to_onehot(target, kwargs['out_channels'])
        
        assert torch.min(probs) >= 0 and torch.max(probs) <= 1
        
        fg_probs = probs[:, self.idc, :, :].squeeze(1)
        
        dice_loss, _ = self.dice_fct(probs, onehot_target)
        _wce_loss = - (target* torch.log(fg_probs + eps) + (1 - target) * torch.log(1 - fg_probs + eps))
        wce_loss = torch.mean(_wce_loss)
    
        loss = self.alpha_CE * wce_loss + (1 - self.alpha_CE) * dice_loss
        
        return loss, None


class BinaryCrossEntropy_OuterBoundingBox(nn.Module):
    """
    Code adapted from https://github.com/LIVIAETS/boxes_tightness_prior/blob/master/losses.py
    """
    def __init__(self, **kwargs):
        super().__init__()
        # Self.idc is used to filter out some classes of the prediction mask
        self.target_str: str = kwargs["target_str"] # ie: 'weak_label', which refers to bounding box
        self.idc: List[int] = kwargs["idc"] #ie: 0 to compute BCE on region outside bounding box
        self.nd: str = kwargs["nd"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, batch: Dict[str, Tensor], eps=1e-10, **kwargs) -> Tensor:
        target = batch[self.target_str]
        assert torch.min(probs) >= 0 and torch.max(probs) <= 1

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        # We create a mask to only consider the target region (true mask, bounding box, etc.), with the idc classes as positive
        mask: Tensor = torch.zeros(target.shape)
        for i in self.idc:
            mask[target == i] = 1
        mask = cast(Tensor, mask).to(target.device)

        # We compute log_p on all values in mask (for self.idc=0, we know region in mask should be background)
        loss = -torch.sum(log_p)
        loss /= mask.sum() + eps

        return loss, None
    
    
class BoxSizePrior(nn.Module):
    """
    Code adapted from https://github.com/LIVIAETS/boxes_tightness_prior/blob/master/losses.py
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.target_str: str = kwargs["target_str"] # ie: 'weak_label', which refers to bounding box
        self.idc: List[int] = kwargs["idc"] #ie: 1 for foreground
        self.C: int = len(self.idc)
        self.thres_prob: List[float] = kwargs["thres_prob"] #ie: 0 or 0.5 (to only consider big enough probabilities)

        # Selecting which penalty to apply (log_barrier or relu)
        penalty = kwargs.get('penalty_type', 'log_barrier')
        self.penalty = penalty_zoo[penalty](**kwargs)

    def __call__(self, probs: Tensor, batch: Dict[str, Tensor],  **kwargs) -> Tensor:
        bounds = batch['bounds']
        assert self.target_str == 'weak_label'
        box_sizes = torch.sum(batch[self.target_str], dim=(-1, -2)).squeeze(1)
        
        assert torch.min(probs) >= 0 and torch.max(probs) <= 1

        # b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        b: int
        b, _, *im_shape = probs.shape
        _, _, k, two = bounds.shape  # scalar or vector
        assert two == 2

        relu = nn.ReLU(inplace=False)
        # Adding threshold on probabilities (to only consider big enough probabilities)
        restricted_probs = relu(probs[:, self.idc, ...] - self.thres_prob).type(torch.float64)
        value = torch.sum(restricted_probs, dim=(-1, -2))[..., None]
        lower_b = bounds[:, [i - 1 for i in self.idc], :, 0]
        upper_b = bounds[:, [i - 1 for i in self.idc], :, 1]

        assert value.shape == (b, self.C, k), value.shape
        assert lower_b.shape == upper_b.shape == (b, self.C, k), lower_b.shape

        upper_z: Tensor = cast(Tensor, (value - upper_b).type(torch.float64)).flatten()
        lower_z: Tensor = cast(Tensor, (lower_b - value).type(torch.float64)).flatten()

        upper_penalty: Tensor = self.penalty(upper_z)
        lower_penalty: Tensor = self.penalty(lower_z)

        _loss: Tensor = upper_penalty + lower_penalty

        loss: Tensor = torch.mean(_loss / box_sizes) 
        assert loss.requires_grad == probs.requires_grad  # Handle the case for validation

        return loss, None
        
    def scheduler_step(self):
        if hasattr(self.penalty, 'step'):
            return self.penalty.step()
        else:
            pass


class ConsistencyLoss(nn.Module):
    """ We compute the consistency loss (MSE loss) between the output of transformed tensors and transformed output tensors

    Returns:
        MSE loss beteen D(T) and T(D)
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.target_str: str = kwargs["target_str"] # ie: 'probs_transformed', which refers to the prediction of the model on the transformed input
        self.pred_name: str = kwargs["pred_name"] # ie: 'transformed_probs', which refers to the transformed prediction of the model
        self.idc: List[int] = kwargs["idc"] #ie: 1 to compute MSE on foreground probabilities
        self.loss_type: str = kwargs.get('loss_type', 'MSE')
        
    def forward(self, probs: Tensor, batch: Dict[str, Tensor], eps=1e-8, **kwargs) -> Tensor:
        target = batch[self.target_str]
        pred = batch[self.pred_name]
        
        if self.idc != []:
            pred = pred[:, self.idc, ...]
            target = target[:, self.idc, ...]
        
        if self.loss_type == 'MSE':
            self.loss_fct = nn.MSELoss()
            loss = self.loss_fct(pred, target)**2
        elif self.loss_type == 'L2':
            _loss = torch.sqrt(torch.sum((pred - target) ** 2, dim=[1, 2, 3]))
            loss = torch.mean(_loss)

        return loss, None