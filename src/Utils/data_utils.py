"""
Author: Mélanie Gaillochet
"""
import numpy as np

from collections import namedtuple
from typing import Iterable, List, Set, Tuple, cast

import torch
import numpy as np
from torch import Tensor
from skimage import measure


def get_bounding_box(ground_truth_map, perturbation_bounds=[5, 20]):
    '''
    This function creates varying bounding box coordinates based on the segmentation contours as prompt for the SAM model
    The padding is random int values between 5 and 20 pixels
    '''

    if len(np.unique(ground_truth_map)) > 1:

        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(*perturbation_bounds))
        x_max = min(W, x_max + np.random.randint(*perturbation_bounds))
        y_min = max(0, y_min - np.random.randint(*perturbation_bounds))
        y_max = min(H, y_max + np.random.randint(*perturbation_bounds))
        
        bbox = [x_min, y_min, x_max, y_max]

        return bbox
    else:
        return [0, 0, 256, 256] # if there is no mask in the array, set bbox to image size


def create_bbox_mask(x_min, y_min, x_max, y_max, mask_size):
    """
    Creates a binary mask with a bounding box.

    Args:
    x_min, y_min, x_max, y_max (int): Coordinates of the bounding box.
    mask_size (tuple): Size of the output mask (height, width).

    Returns:
    numpy.ndarray: A binary mask with the bounding box.
    """
    # Create an empty mask with the given size
    mask = np.zeros(mask_size, dtype=np.uint8)

    # Set the pixels within the bounding box to 1
    mask[y_min:y_max+1, x_min:x_max+1] = 1

    return mask


BoxCoords = namedtuple("BoxCoords", ["x", "y", "width", "height"])


def binary2boxcoords(seg: Tensor) -> List[BoxCoords]:
    """
    Code for bounding box tightness prior utils 
    from official repository of 'Bounding boxes for weakly supervised segmentation: Global constraints get close to full supervision,' Kervadec et al. (MIDL 2020)
    Taken from https://github.com/LIVIAETS/boxes_tightness_prior/blob/master/utils.py

    Converts (0-1) bounding box mask to box prompt coordinates (x, y, box_height, box_width)
    """
    blobs, n_blob = measure.label(seg.cpu().numpy(), background=0, return_num=True)

    assert set(np.unique(blobs)) <= set(range(0, n_blob + 1)), np.unique(blobs)

    class_coords: List[BoxCoords] = []
    for b in range(1, n_blob + 1):
        blob_mask = (blobs == b)
        coords = np.argwhere(blob_mask)
        x1, y1 = coords.min(axis=0)
        x2, y2 = coords.max(axis=0)
        class_coords.append(BoxCoords(x1, y1, x2 - x1, y2 - y1))

    assert len(class_coords) == n_blob

    return class_coords


def boxcoords2masks_bounds(boxes: List[BoxCoords], shape: Tuple[int, int], w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Code for bounding box tightness prior utils 
    from official repository of 'Bounding boxes for weakly supervised segmentation: Global constraints get close to full supervision,' Kervadec et al. (MIDL 2020)
    Taken from https://github.com/LIVIAETS/boxes_tightness_prior/blob/master/utils.py

    Divides the box mask into individual bands of width w.
    Returns 2 tensors: one of shape [#bands, H, W], containing the value 1 for every vertical and horizontal band, and another containing the associated width of every band
    #bands = height of bounding box / d + width of bounding_box / d
    """
    masks_list, bounds_list = [], []

    box: BoxCoords
    for box in boxes:
        for i in range(box.width // w):
            mask = torch.zeros(shape, dtype=torch.float32)
            mask[box.x + i * w:box.x + (i + 1) * w, box.y:box.y + box.height + 1] = 1
            masks_list.append(mask)
            bounds_list.append(w)

        if box.width % w:
            mask = torch.zeros(shape, dtype=torch.float32)
            mask[box.x + box.width - (box.width % w):box.x + box.width + 1, box.y:box.y + box.height + 1] = 1
            masks_list.append(mask)
            bounds_list.append(box.width % w + 1)   # +1 because the width does not include the first pixel

        for j in range(box.height // w):
            mask = torch.zeros(shape, dtype=torch.float32)
            mask[box.x:box.x + box.width + 1, box.y + j * w:box.y + (j + 1) * w] = 1
            masks_list.append(mask)
            bounds_list.append(w)

        if box.height % w:
            mask = torch.zeros(shape, dtype=torch.float32)
            mask[box.x:box.x + box.width + 1, box.y + box.height - (box.height % w):box.y + box.height + 1] = 1
            masks_list.append(mask)
            bounds_list.append(box.height % w + 1)   # +1 because the width does not include the first pixel

    # Stack all masks and convert bounds list to tensor
    if masks_list:
        masks = torch.stack(masks_list, dim=0)
    else:
        masks = torch.zeros((0, *shape), dtype=torch.float32)

    bounds = torch.tensor(bounds_list, dtype=torch.float32) if bounds_list else torch.zeros((0,), dtype=torch.float32)

    # We fill up the tensor to make it same size as the other tensors
    # Create a zero tensor of size [256 - # bands in mask, 512, 512]
    max_n_channels = masks.shape[1] // w + masks.shape[2] // w
    zero_padding = torch.zeros(max_n_channels - masks.shape[0], masks.shape[1], masks.shape[2])

    # Concatenate the original tensor with the zero tensor
    masks = torch.cat((masks, zero_padding), 0)
    bounds = torch.cat((bounds, torch.zeros(max_n_channels - bounds.shape[0])))

    return masks, bounds