"""
Author: Mélanie Gaillochet
"""
import os
import time
from datetime import datetime
import numpy as np
import json
import time
import argparse
from argparse import ArgumentParser
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from Data.datamodule import SAMDataModule
from segment_anything import sam_model_registry

from Models.SAM_WithPromptGenerator import SAMPromptLearning_Ours
from Utils.load_utils import get_dict_from_config, update_config_from_args, NpEncoder, save_to_logger
from Utils.utils import find_matching_key


print("PyTorch version:", torch.__version__)
print("PyTorch Lightning version:", pl.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())


def train_model(model_config, module_config, train_config, data_config, data_dir, logger_config,
                checkpoint_path=None, gpu_devices=1, seed=42):

    seed_everything(seed, workers=True)

    if logger_config['name'] == 'comet':
            # We set comet_ml logger
        logger = CometLogger(
        api_key=logger_config['api_key'],
        workspace=logger_config['workspace'],
        experiment_name=logger_config['experiment_name'],
        )
    else:
        logger = True  # Default logger (TensorBoard)
              
    # We create the datamodule  
    kwargs = {'prompt': train_config['prompt'] if 'prompt' in train_config else None,
              'data_shape': data_config['data_shape'],
              'class_to_segment': data_config['class_to_segment'],
              'box_prior_args': find_matching_key(train_config["loss"], "TightBoxPrior", default={}).get('kwargs', None),
              'bounds_args_list': [{**_config["other_kwargs"], 'C': model_config["out_channels"]} for _, _config in train_config["loss"].items() if (_config["other_kwargs"] is not None and "bounds_name" in _config["other_kwargs"] and _config["other_kwargs"]["bounds_name"] is not None)],
              'compute_sam_embeddings': data_config['compute_sam_embeddings'],
              'model_image_size': model_config.get('image_size', None),
              'sam_checkpoint': model_config.get('sam_checkpoint', None)
            }
    
    data_module = SAMDataModule(data_dir=data_dir,
                                dataset_name=data_config["dataset_name"],
                                batch_size=train_config["batch_size"],
                                val_batch_size=train_config["batch_size"],
                                num_workers=train_config["num_workers"],
                                train_indices=train_config["train_indices"],
                                val_indices=train_config["val_indices"],
                                dataset_kwargs=kwargs)
                
    # We create model (importing the appropriate class from model_config['model_class'])
    num_indices = 'all' if train_config["train_indices"]==[] else len(train_config["train_indices"])
    os.makedirs(os.path.join(checkpoint_path, '{}labeled'.format(num_indices)), exist_ok=True)
    model_cls = globals().get(model_config['model_class'])
    full_model = model_cls(num_devices=1,
                           model_config=model_config,
                           module_config=module_config,
                           train_config=train_config,
                           seed=seed,
                           **{"checkpoint_path": os.path.join(checkpoint_path, '{}labeled'.format(num_indices))}
                    )          

    # We can remove SAM's image encoder (if the image embeddings are already computed)
    if not data_config['compute_sam_embeddings']:
        del full_model.sam.image_encoder

    # Get Total number of parameters and total number of trainable parameters
    total_params = sum(p.numel() for p in full_model.parameters())
    trainable_params = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
    print(f"\n Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}\n")

    # We set-up the trainer
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(
            deterministic=True,
            max_epochs=train_config["num_epochs"],
            precision="16-mixed",
            devices=gpu_devices,
            accelerator='gpu',
            sync_batchnorm=True,
            log_every_n_steps=5,
            check_val_every_n_epoch=5,
            callbacks=[
                ModelCheckpoint(dirpath=os.path.join(checkpoint_path, '{}labeled'.format(num_indices)),
                                save_last=True),
                lr_monitor
            ],
            logger=logger,
            num_sanity_val_steps=0)
    save_to_logger(trainer.logger, 'hyperparameter', checkpoint_path, 'checkpoint_path')
    save_to_logger(trainer.logger, 'hyperparameter', total_params, 'total_params')
    save_to_logger(trainer.logger, 'hyperparameter', trainable_params, 'trainable_params')

    # We train our model
    print("\n #### Training model ####")
    train_start_time = time.time()
    trainer.fit(full_model, data_module)
    
    # We keep track of runtime
    train_time = time.time() - train_start_time
    save_to_logger(trainer.logger, 'hyperparameter', np.round(train_time/60, 2), 'train time (min)')

    # We evaluate our trained model on the test set
    print("\n #### Evaluating on test set ####")
    
     # We put back the original image encoder if model is SAM or its variants
    if hasattr(full_model, 'sam'):
        model_args = argparse.Namespace(**model_config)
        temp_model = sam_model_registry[model_config["model_name"]](model_args.sam_checkpoint)
        full_model.sam.image_encoder = temp_model.image_encoder
        full_model.sam.prompt_encoder = temp_model.prompt_encoder
        print('Loaded original image encoder')
    
    data_module.setup(stage = 'test')    
    metric_dict = trainer.test(full_model, datamodule=data_module)
    
    # We save the metrics
    results_path = os.path.join(checkpoint_path, 'metrics.json')
    with open(results_path, 'w') as file:
        json.dump(metric_dict, file, indent=4, cls=NpEncoder)
        
    return trainer, full_model, data_module, train_time


if __name__ == "__main__":
    parser = ArgumentParser()
    # These are the paths to the data and output folder
    parser.add_argument('--data_dir', default='/home/AR32500/net/data', type=str, help='Directory for data')
    parser.add_argument('--models_dir', default='/home/AR32500/net/models', type=str, help='Path to model checkpoints')
    parser.add_argument('--output_dir', default='/home/AR32500/AR32500/output', type=str, help='Directory for output run')

    # These are config files located in src/Config
    parser.add_argument('--data_config',  type=str, 
                        default='data_config/ACDC_256.yaml'
                        #default='data_config/CAMUS_512.yaml'
                        #default='data_config/HC_640.yaml'
                        )
    parser.add_argument('--model_config', type=str, 
                        default='model_config/ours_samh_config.yaml'
                        )
    parser.add_argument('--module_config', type=str, default='model_config/module_hardnet_config.yaml')
    parser.add_argument('--train_config', type=str, default='train_config/train_config_200_100_00001.yaml')
    parser.add_argument('--logger_config', type=str, default='logger_config.yaml')
    parser.add_argument('--prompt_config', type=str, 
                        default='prompt_config/box_tight.yaml',
                        )
    parser.add_argument('--loss_config', type=str, nargs='+',
                        help='type of loss to appply (ie: CE, entropy_minimization, bounding_box_prior)',
                        default=[
                            'loss_config/WBCE_Dice/wbcedice_gtpromptedpred.yaml',
                            'loss_config/BoxSizePrior/base_70-90_mult11_freq5_W001.yaml',
                            'loss_config/BinaryCrossEntropy_OuterBoundingBox/W0001.yaml',
                            'loss_config/Consistency/L2_weak_W0001.yaml'
                            ])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu_idx', default=[0], type=int, nargs='+', help='otherwise, gpu index, if we want to use a specific gpu')

    # Training hyper-parameters that we should change according to the dataset
    # Arguments of data input and output
    parser.add_argument('--data__compute_sam_embeddings', help='whether to use compute embeddings during training (if not, will use precomputed embeddings)',
                        action="store_true", default=False)
    parser.add_argument('--data__class_to_segment', type=int, help='class values to segment',
                        default=1)

    parser.add_argument('--train__train_indices', type=int, nargs='+', help='indices of training data for Segmentation task',
                        default=[])
    parser.add_argument('--train__val_indices', help='indices of val data for Segmentation task', 
                        default=[])
    parser.add_argument('--train__clip_gradient_norm_value', type=float, help='value to clip gradient norm (0 = No clipping)',
                        default=1.0)
    args = parser.parse_args()
    
    # We set the gpu devices (either a specific gpu or a given number of available gpus)
    gpu_devices = args.gpu_idx
    print('gpu_devices {}'.format(gpu_devices))

    # We extract the configs from the file names
    train_config = get_dict_from_config(args.train_config)
    data_config = get_dict_from_config(args.data_config)
    model_config = get_dict_from_config(args.model_config)
    module_config = get_dict_from_config(args.module_config)
    logger_config = get_dict_from_config(args.logger_config)
    train_config["loss"] = {}
    
    # We add the loss configs to the train config. If two losses have the same type, we will add a subscript
    for _file_config in args.loss_config:
        cur_config = get_dict_from_config(_file_config)
        loss_name = cur_config["type"]
        # Check if the loss_name is already in the dictionary
        original_loss_name = loss_name
        count = 1
        while loss_name in train_config["loss"]:
            # Append a number to the loss_name if it already exists
            loss_name = f"{original_loss_name}{count}"
            count += 1
        # Add the (possibly renamed) loss_name to the train_config
        train_config["loss"][loss_name] = cur_config
            
    # We update the model and logger config files with the command-line arguments
    data_config = update_config_from_args(data_config, args, 'data')
    train_config = update_config_from_args(train_config, args, 'train')
    model_config['sam_checkpoint'] = os.path.join(args.models_dir, model_config.get('sam_checkpoint', ''))  # We set the path to the SAM checkpoint

    # If we are at inference, we can add the config on the prompts to be used
    if args.prompt_config != '':
        prompt_config = get_dict_from_config(args.prompt_config)
        train_config = {**train_config, **{'prompt': prompt_config}}
    
    # We create the experiment name
    experiment_name = data_config['dataset_name'] + '__class' + str(data_config['class_to_segment'])
    experiment_name += '__indices' + '-'.join(map(str, train_config['train_indices']))[:60] if len(train_config['train_indices']) > 0 else ''
    logger_config['experiment_name'] = experiment_name  # Useful if we use comet logger

    # We create a checkpoint path
    start_time = datetime.today()
    log_id = '{}_{}h{}min'.format(start_time.date(), start_time.hour, start_time.minute)
    checkpoint_path = os.path.join(args.output_dir, log_id, experiment_name + '_seed{}'.format(args.seed))
    print('\n Checkpoint_path is: {}\n'.format(checkpoint_path))
    
    trainer, full_model, AL_data_module, train_time = train_model(model_config, module_config, train_config, data_config, args.data_dir, logger_config,
                checkpoint_path, gpu_devices, args.seed)
