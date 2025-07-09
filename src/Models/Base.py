"""
Author: Mélanie Gaillochet
"""
import re
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, LambdaLR

import pytorch_lightning as pl

from Utils.training_utils import (dice_metric, assd_metric)
from Utils.utils import to_onehot
from Utils.load_utils import save_to_logger
from Utils.data_utils import get_bounding_box, create_bbox_mask


class BaseModel(pl.LightningModule):
    def __init__(
            self, 
            num_devices: int = 1,
            model_config: dict = {},
            train_config: dict = {},
            val_plot_slice_interval: int = 1,
            seed = 0,
            **kwargs
    ):
        super().__init__()
        torch.use_deterministic_algorithms(True)  # For reproducibility
        self.save_hyperparameters()
        self.num_devices = num_devices
        self.train_config = train_config
        self.per_device_batch_size = train_config["batch_size"] # int
        self.optimizer_config =  train_config["optimizer"] # dict
        self.sched_config = train_config["sched"] # dict
        self.loss_config = train_config["loss"] # dict

        self.in_channels = model_config["in_channels"]
        self.out_channels = model_config["out_channels"]

        self.activation_fct = nn.Sigmoid() if self.out_channels == 2 else nn.Softmax(dim=1)
        
        # We define the model losses from the provided loss list
        self.all_loss_names = defaultdict(list)
        self.all_loss_fct = defaultdict(list)
        self.all_loss_weights = defaultdict(list)
        self.all_loss_start_epoch = defaultdict(list)
        self.all_loss_target = defaultdict(list)

        for i, loss_name in enumerate(self.loss_config.keys()):
            print(f">> {i}th list of losses: {loss_name} - {self.loss_config[loss_name]}")
            loss_params = self.loss_config[loss_name]["kwargs"] if self.loss_config[loss_name]["kwargs"] is not None else {}
            pred_name = loss_params.get('pred_name', 'probs') if loss_params is not None else 'probs'
            target_name = loss_params.get('target_str')
            fn = self.loss_config[loss_name]["other_kwargs"]["fn"] if self.loss_config[loss_name]["other_kwargs"] is not None else None
            loss_name = re.sub(r'\d+$', '', loss_name) # We remove integers at the end of the string (added if several losses with the same type). Note that the losses should not apply to the same pred_name
            loss_class = getattr(__import__('losses'), loss_name) #
            self.all_loss_names[pred_name].append(loss_name)
            self.all_loss_fct[pred_name].append(loss_class(**loss_params, fn=fn))
            self.all_loss_weights[pred_name].append(self.loss_config[loss_name]["weight"])
            self.all_loss_start_epoch[pred_name].append(self.loss_config[loss_name]["start_epoch"])
            self.all_loss_target[pred_name].append(target_name)

        # Initialize empty lists for active losses and weights
        self.loss_names = defaultdict(list)
        self.loss_fct = defaultdict(list)
        self.loss_weights = defaultdict(list)
        self.loss_kwargs = {'out_channels': self.out_channels}

        # For 3D validation and test (Note that train_volume_list and test_volume_list are loaded by the datamodule)
        self.val_data_list, self.test_data_list = [], []
        self.val_target_list, self.test_target_list = [], []
        self.val_logits_list, self.test_logits_list = [], []
        self.val_dice_list, self.test_dice_list = [], []
        self.val_per_class_dice_list, self.test_per_class_dice_list = [], []
        self.val_slice, self.test_slice = 0, 0

        # Plot params
        self.log_metric_freq = 5 
        self.log_img_freq = 50  # Must be multiple of self.log_metric_freq
        self.val_plot_slice_interval = val_plot_slice_interval
        self.plot_type = 'contour' if self.out_channels == 2 else 'image'
        
        # To save best and last val loss and metrics
        self.best_val_metric = float(0)
        self.best_epoch = 0
        self.validation_losses = []
        self.val_outputs = []
        self.checkpoint_path = kwargs.get("checkpoint_path")
                 
    def forward(self, x):
        raise NotImplementedError

    def _training_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def _validation_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def _test_step(self, batch, batch_idx):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        # If logger is comet_ml
        if isinstance(self.logger, pl.loggers.CometLogger):
            with self.logger.experiment.train():
                return  self._training_step(batch, batch_idx)
        # If logger is the default TensorBoardLogger or another logger
        else:
            return self._training_step(batch, batch_idx)
        
    def validation_step(self, batch, batch_idx):
        # If logger is comet_ml
        if isinstance(self.logger, pl.loggers.CometLogger):
            with self.logger.experiment.validate():
                return  self._validation_step(batch, batch_idx)
        # If logger is the default TensorBoardLogger or another logger
        else:
            return self._validation_step(batch, batch_idx)
        
    def test_step(self, batch, batch_idx):
        # If logger is comet_ml
        if isinstance(self.logger, pl.loggers.CometLogger):
            with self.logger.experiment.test():
                return self._test_step(batch, batch_idx)
        # If logger is the default TensorBoardLogger or another logger
        else:
            return self._test_step(batch, batch_idx)

    def on_train_epoch_start(self):
        """
        PyTorch Lightning does not natively support learning rate warmup. 
        Therefore, you need to manually adjust the learning rate during the first few epochs (warmup phase).
        """
        # We adjust the learning rate
        cur_epoch = self.current_epoch
        if 'GradualWarmup' in self.sched_config:
            if cur_epoch < self.sched_config['GradualWarmup']["warmup_steps"]:
                lr_scale = min(1., float(cur_epoch + 1) / self.sched_config['GradualWarmup']["warmup_steps"])
                for pg in self.optimizers().param_groups:
                    pg['lr'] = lr_scale * self.optimizer_config["lr"]

        # We adjust the loss list
        self.loss_fct = defaultdict(list)
        self.loss_weights = defaultdict(list)
                
        # Update active losses and weights based on the current epoch
        for pred_name in self.all_loss_names.keys():
            for loss_name, loss_fn, start_epoch, weight in zip(self.all_loss_names[pred_name], self.all_loss_fct[pred_name], 
                                                    self.all_loss_start_epoch[pred_name], self.all_loss_weights[pred_name]):
                if cur_epoch >= start_epoch:
                    # We activate losses based on the current epoch
                    self.loss_names[pred_name].append(loss_name)
                    self.loss_fct[pred_name].append(loss_fn)
                    self.loss_weights[pred_name].append(weight)

    def on_train_epoch_end(self):
        # Update active losses and weights based on the current epoch
        for pred_name in self.all_loss_names.keys():
            for loss_name, loss_fn, in zip(self.all_loss_names[pred_name], self.all_loss_fct[pred_name]):
                # We update the parameters of the loss functions
                if hasattr(loss_fn, 'scheduler_step'):
                    if hasattr(loss_fn.penalty, 'update_frequency') and self.current_epoch % loss_fn.penalty.update_frequency == 0:
                        updated_param = loss_fn.scheduler_step()
                        save_to_logger(self.logger, 'metric', updated_param, loss_name + '_updated_param')

    def _configure_optimizers(self, params):
        # We set the optimizer
        if self.optimizer_config["type"] == 'SGD':        
            optimizer = optim.SGD(params, lr=self.optimizer_config["lr"],
                                momentum=self.optimizer_config["momentum"], 
                                weight_decay=self.optimizer_config["weight_decay"])
        elif self.optimizer_config["type"] == 'Adam': 
            optimizer = optim.Adam(params, lr=self.optimizer_config["lr"],
                              weight_decay=self.optimizer_config["weight_decay"])
            
        # We set the scheduler
        if "MultiStepLR" in self.sched_config:
            scheduler = MultiStepLR(optimizer, milestones=self.sched_config["MultiStepLR"]["milestones"], 
                                                 gamma=self.sched_config["MultiStepLR"]["gamma"])
        elif "CosineAnnealingLR" in self.sched_config:
            scheduler = CosineAnnealingLR(optimizer, T_max=self.sched_config["CosineAnnealingLR"]["max_epoch"])
        elif "PolynomialDecay" in self.sched_config:
            lambda_poly = lambda iter_num: (1.0 - iter_num / self.sched_config["PolynomialDecay"]["max_epoch"]) ** 0.9
            scheduler = LambdaLR(optimizer, lr_lambda=lambda_poly)
        
        scheduler = {
            "scheduler": scheduler,
            "interval": self.sched_config["update_interval"],
            "frequency": self.sched_config["update_freq"],
        }

        return [optimizer], [scheduler]
    
    def _compute_seg_metrics(self, pred, y):
        """
        Computing typical segmentation metrics: dice, iou and hausdorff95, both total and per class
        args:
            pred: (B, C, H, W) or (B, H, W) tensor
            y: (B, C, H, W) or (B, H, W) tensor
        """
        onehot_pred = to_onehot(pred.squeeze(1), self.out_channels)
        onehot_target = to_onehot(y.squeeze(1), self.out_channels)
        _dice = dice_metric(onehot_pred, onehot_target)
        dice = torch.mean(_dice)

        assd = torch.mean(assd_metric(onehot_pred, onehot_target))

        pred_bb_coords = np.stack([get_bounding_box(_pred[0, :, :].detach().cpu().numpy(), perturbation_bounds=[0, 1]) for _pred in pred ])
        pred_bb = torch.tensor(np.stack([create_bbox_mask(*coords, pred.shape[-2:]) for coords in pred_bb_coords]))
        y_bb_coords = np.stack([get_bounding_box(_y[0, :, :].detach().cpu().numpy(), perturbation_bounds=[0, 1]) for _y in y ])
        y_bb = torch.tensor(np.stack([create_bbox_mask(*coords, y.shape[-2:]) for coords in y_bb_coords]))
        onehot_pred_bb = to_onehot(pred_bb, self.out_channels)
        onehot_target_bb = to_onehot(y_bb, self.out_channels)
        
        # Bounding box-based metrics
        _dice_bb = dice_metric(onehot_pred_bb, onehot_target_bb)
        dice_bb = torch.mean(_dice_bb)
        
        sum_tg_bb = torch.sum(onehot_target_bb[:, 1, :, :], dim=(-1, -2)).detach().cpu()
        sum_pred = torch.sum(onehot_pred[:, 1, :, :], dim=(-1, -2)).detach().cpu()
        box_proportion = torch.mean(sum_pred / sum_tg_bb)

        metrics = {'dice': 100*dice,
                   'assd': assd,
                   'dice_bounding_box': 100*dice_bb,
                   'box_proportion': 100*box_proportion
                   }
        
        return metrics



