"""
Author: Mélanie Gaillochet
"""
import re
import torch


def get_nested_value(dictionary, *keys, default=None):
    """
    We retrieve a value nested within a dictionary using the **kwargs approach
    """
    for key in keys:
        if dictionary is not None:
            dictionary = dictionary.get(key, {})
    return dictionary if dictionary else default

def _atoi(text):
    """
    We return the string as type int if it represents a number (or the string itself otherwise)
    """
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    We return the list of string and digits as different entries,
    following the order in which they appear in the string
    """
    return [_atoi(c) for c in re.split('(\\d+)', text)]


def to_onehot(input, n_classes):
    """
    We do a one hot encoding of each label in 3D.
    (ie: instead of having a dimension of size 1 with values 0-k,
    we have 3 axes, all with values 0 or 1)
    :param input: tensor
    :param n_classes:
    :return:
    """
    assert torch.is_tensor(input)

    # We get (bs, l, h, w, n_channels), where n_channels is now > 1
    one_hot = torch.nn.functional.one_hot(input.to(torch.int64), n_classes)

    # We permute axes to put # channels as 2nd dim
    if len(one_hot.shape) == 5:
        # (BS, H, W, L, n_channels) --> (BS, n_channels, H, W, L)
        one_hot = one_hot.permute(0, 4, 1, 2, 3)
    elif len(one_hot.shape) == 4:
        # (BS, H, W, n_channels) --> (BS, n_channels, H, W)
        one_hot = one_hot.permute(0, 3, 1, 2)
    elif len(one_hot.shape) == 3:
        # (H, W, n_channels) --> (n_channels, H, W)
        one_hot = one_hot.permute(2, 1, 0)
    return one_hot
        

def find_matching_key(data, key_substring, default=None):
    """
    Search for keys in the dictionary that contain the specified substring.

    Args:
    data (dict): The dictionary in which to search for keys.
    key_substring (str): The substring to look for in the keys of the dictionary.

    Returns:
    Any: The value corresponding to the first key that contains the substring, or {} if no such key exists.
    """
    for key, value in data.items():
        if key_substring in key:
            return value
    return default
