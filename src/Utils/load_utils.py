"""
Author: Mélanie Gaillochet
"""
import os
import sys
import json
import yaml
import numpy as np
from flatten_dict import flatten
from flatten_dict import unflatten
import h5py
import nibabel as nib
import pytorch_lightning as pl

sys.path.append(".") 
from Configs.config import config_folder


def _read_json_file(file_path):
    """
    We are reading the json file and returning a dictionary
    :param file_path:
    :return:
    """
    # We parse the configurations from the config json file provided
    with open(file_path, 'r') as file:
        output_dict = json.load(file)
    return output_dict


def _read_yaml_file(file_path):
    """
    We are reading the yaml file and returning a dictionary
    :param file_path:
    :return:
    """
    # We parse the configurations from the config json file provided
    with open(file_path, 'r') as file:
        output_dict = yaml.safe_load(file)
    return output_dict


def get_dict_from_config(config_filename):
    """
    Get the config file (json or yaml) as a dictionary
    :param config_filename: name of config file (located in config folder)
    :return: dictionary
    """
    config_filepath = os.path.join(config_folder, config_filename)
    
    if config_filepath.endswith('.json'):
        config_dict = _read_json_file(config_filepath)
    elif config_filepath.endswith('.yaml'):
        config_dict = _read_yaml_file(config_filepath)
    else:
        config_dict = {}

    return config_dict


def update_config_from_args(config, args, prefix):
    """
    We update the given config with the values given by the args
    Args:
        config (list): config that we would like to update
        args (parser arguments): input arguments whose values starting with given prefix we would like to use
                                Must be in the form <prefix>/<config_var_name-separated-by-/-if-leveled>/ (ie: ssl/sched/step_size) 
                                ! Must also not contain 'config' in the name !
        prefix (str): all parser arguments starting with the prefix + '/' will be updated.

    Returns:
        config: updated_config
    """
    # We extract the names of variables to update
    var_to_update_list = [name for name in vars(args) if prefix +'__' in name]# and 'config' not in name)]
    
    updated_config = flatten(config)  # We convert dictionary to list of tuples (tuples incorporating level information)
    for name in var_to_update_list:
        new_val = getattr(args, name)
        if new_val is not None:   # if the values given is not null, we will update the dictionary
            variable = name.replace(prefix + '__', '', 1)  # We remove the prefix
            level_tuple = tuple(variable.split('__'))   # We create a tuple with all sublevels of config
            updated_config[level_tuple] = new_val
    updated_config = unflatten(updated_config)  # We convert back to a dictionary
    
    return updated_config


def save_hdf5(data, img_idx, dest_file):
    """
    We are saving an hdf5 object
    :param data:
    :param filename:
    :return:
    """
    with h5py.File(dest_file, "a", libver='latest', swmr=True) as hf:
        hf.swmr_mode = True
        hf.create_dataset(name=str(img_idx), data=data, shape=data.shape, dtype=data.dtype)


def create_unexisting_folder(dir_path):
    """
    We create a folder with the given path.
    If the folder already exists, we add '_1', '_2', ... to it
    :param dir_path:
    """
    i = 0
    created = False
    path = dir_path
    while not created:
        try:
            os.makedirs(path)
            created = True
        except OSError or FileExistsError:
            i += 1
            path = dir_path + '_' + str(i)
            # print(path)
    return path


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def save_to_logger(logger, type, data, name, epoch=None):
    """
    We save data to the given logger

    Args:
        logger (tensorboard logger, comet ml logger, etc.): logger of pytorch lightning trainer
        type (str): 'metric' (to save scalar), 'list'
        data (any): what we want to save
        name (str): name to use when saving data
        epoch (int): if we want to assign the data to a given epoch
    """
    if type == 'metric':
        if isinstance(logger, pl.loggers.CometLogger):
            # Saving on comet_ml
            logger.experiment.log_metric(name, data)
            
    elif type == 'list':
        if isinstance(logger, pl.loggers.CometLogger):
            # Saving on comet_ml
            logger.experiment.log_other(name, data)
        else:
            # Saving on TensorBoardLogger as scalar, with epoch as indice in list
            for i in range(len(data)):
                logger.experiment.add_scalar(name, data[i], i)
                
    elif type == 'hyperparameter':
        if isinstance(logger, pl.loggers.CometLogger):
            # Saving on comet_ml
            logger.experiment.log_parameter(name, data)
        else:
            # Saving on TensorBoardLogger
            logger.log_hyperparams({name: data})
