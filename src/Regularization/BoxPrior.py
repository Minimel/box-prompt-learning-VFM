"""
Code adapted from official repository of 'Bounding boxes for weakly supervised segmentation: Global constraints get close to full supervision,' Kervadec et al. (MIDL 2020)
https://github.com/LIVIAETS/boxes_tightness_prior/blob/master/bounds.py
"""

from typing import Any, Callable, List, Tuple
import torch
from torch import Tensor

from Utils.data_utils import BoxCoords, binary2boxcoords, boxcoords2masks_bounds


class BoxPriorBounds():
    """
    This class returns all individual bands of width w that make up the box mask
    
    Args:
        w (int): width of the segments we want to divide the box into (horizontal and vertical)
        idc (List of ints): indices of the classes to consider  
    
    Returns:
        res (List of Tuples): List of tuples of tensors, each tuple containing the masks and the bounds of the object size
    """
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"] 
        self.class_mask = [i - 1 for i in self.idc] # Removing 1 because class 0 is the background and has no bounding box
        self.w: int = kwargs['w'] 

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, box_targets: Tensor) -> List[Tuple[Tensor, Tensor]]:
        K, W, H = box_targets.shape # K is the number of classes, W is the width and H is the height of the image
        assert torch.all(torch.isin(box_targets, torch.tensor([0, 1])))

        # Because computing the boxes on the background class, then discarding it, would destroy the memory
        boxes_per_class: List[List[BoxCoords]]
        boxes_per_class = [binary2boxcoords(box_targets[k]) if k in self.class_mask else [] for k in range(K)]

        try:
            [[masks, bounds]] = [boxcoords2masks_bounds(boxes, (W, H), self.w) for boxes in boxes_per_class]
        except Exception as e:
            print(f"Error in {self.__class__.__name__}: {str(e)}")

        return [[masks, bounds]]


class BoxBounds():
    """
    This class returns the bounds of the object size, given the bounding box and [eps1, eps2]
    
    Args:   
        margins (Tuple of floats): the margins to apply to the bounding box, each valued between 0 and 1 (ie: [0.5, 0.9])
        
    Returns:
        bounds (Tensor): the bounds of the object size as [box_size*eps1, box_size*eps2]
    
    """
    def __init__(self, **kwargs):
        self.margins: Tensor = torch.Tensor(kwargs['margins'])
        assert len(self.margins) == 2
        assert self.margins[0] <= self.margins[1]

    def __call__(self, weak_target: Tensor, filename: str) -> Tensor:
        c = len(weak_target)
        box_sizes: Tensor = torch.sum(weak_target, dim=(-1, -2))[..., None]

        _bounds: Tensor = box_sizes * self.margins

        bounds = _bounds[:, None, :]
        assert bounds.shape == (c, 1, 2)

        return bounds
    

box_prior_zoo = {'BoxPriorBounds': BoxPriorBounds,
                 'BoxBounds': BoxBounds}