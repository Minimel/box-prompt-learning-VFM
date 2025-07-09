"""
Author: Mélanie Gaillochet
"""
import os
import sys
import numpy as np
import argparse
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import Resize, InterpolationMode
from scipy.ndimage import label as scipy_label
from skimage.measure import find_contours
from monai.transforms import Compose, RandAffined, RandFlipd, RandRotate90d, RandSpatialCropd, RandRotated, CenterSpatialCropd, ToTensord, CenterSpatialCrop

sys.path.append(".") 
from segment_anything import sam_model_registry

from Models.Base import BaseModel
from Models.PromptGenerator import promptmodule_zoo
from Utils.plot_utils import plot_data_pred_volume


class SAMPromptLearning_Base(BaseModel):
    def __init__(self, 
                 num_devices: int = 1,
                 model_config: dict = {},
                 module_config: dict = {},
                 train_config: dict = {},
                 val_plot_slice_interval: int = 1,
                 seed = 0, 
                 **kwargs) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.
        """
        super().__init__(num_devices, model_config, train_config,
                            val_plot_slice_interval, seed, **kwargs)
        
        sam_args = argparse.Namespace(**model_config)
        print('sam_args: {}'.format(sam_args))
        
        self.sam = sam_model_registry[model_config["model_name"]](sam_args.sam_checkpoint)
        self.sam_checkpoint = sam_args.sam_checkpoint
        
        # SAM only works for 2 classes, so activation will be sigmoid
        self.activation_fct = nn.Sigmoid()
        self.pred_threshold = 0.5

        # We iterate through the named children of sam and unfreeze the specified blocks
        for name, module in self.sam.named_children():
            # We freeze SAM module
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
        
        # We create the prompt embedding module
        module_name = module_config['type']
        self.module_input: str = module_config['args']['input']   # 'image_orig_size'
        self.module_outputs: list[str] = module_config['args']['output']   # ['sparse_embeddings', 'dense_embeddings']
        self.module = promptmodule_zoo[module_name](**module_config['args'])
        self.clip_gradient_norm_value = train_config['clip_gradient_norm_value']
        
        # Other variables to define
        self.sam_multimask = False

        self.plot_type = 'image_contour'
        self.log_metric_freq = 1
        self.log_img_freq = 0  # Must be multiple of self.log_metric_freq

    def get_input_dict(self, imgs, original_sz, img_sz):
        batched_input = []
        for i, img in enumerate(imgs):
            input_size = tuple([int(x) for x in img_sz])
            original_size = tuple([int(x) for x in original_sz])
            singel_input = {
                'image': img,
                'original_size': original_size,
                'image_size': input_size,
                'point_coords': None,
                'point_labels': None,
            }
            batched_input.append(singel_input)
        return batched_input

    def sam_forward(self, image_embeddings, image_positional_embeddings, sparse_embeddings, dense_embeddings, batched_input):
        low_res_pred_masks, iou_predictions = self.sam.mask_decoder(
        image_embeddings = image_embeddings, # (B, 256, 64, 64)
        image_pe = image_positional_embeddings[0:1],  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings.squeeze(1), # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=self.sam_multimask
        )

        if self.sam_multimask:
            max_values, max_indexs = torch.max(iou_predictions, dim=1)
            max_values = max_values.unsqueeze(1)
            iou_predictions = max_values
            low_res = []
            for i, idx in enumerate(max_indexs):
                low_res.append(low_res_pred_masks[i:i+1, idx])
            low_res_pred_masks = torch.stack(low_res, 0)

        pred_masks = F.interpolate(low_res_pred_masks,(batched_input['label'].shape[-2], batched_input['label'].shape[-1]), mode="bilinear", align_corners=False)
 
        return pred_masks, low_res_pred_masks, iou_predictions

    def configure_optimizers(self):
        params = list(self.sam.parameters()) + list(self.module.parameters())
        return self._configure_optimizers(params)

    def on_before_optimizer_step(self, optimizer):
        # Clip gradients
        if self.clip_gradient_norm_value != 0:
            params = list(self.sam.parameters()) + list(self.module.parameters())
            torch.nn.utils.clip_grad_norm_(params, 
                                           max_norm=self.clip_gradient_norm_value)


class SAMPromptLearning_Ours(SAMPromptLearning_Base):
    def __init__(self, 
                 num_devices: int = 1,
                 model_config: dict = {},
                 module_config: dict = {},
                 train_config: dict = {},
                 val_plot_slice_interval: int = 1,
                 seed = 0,
                **kwargs) -> None:
        """
        Compared to SAMPromptLearning_Base, this model allows for teh consistency loss computation.
        """
        super().__init__(num_devices, model_config, module_config, train_config,
                            val_plot_slice_interval, seed, **kwargs)
        self.eval_metric = 'dice_bounding_box'
        
        if 'ConsistencyLoss' in self.loss_config.keys():
            if 'augmentation_strength' in self.loss_config['ConsistencyLoss']["kwargs"]:
                self.augmentation_strength = self.loss_config['ConsistencyLoss']["kwargs"]['augmentation_strength']
            else:
                self.augmentation_strength = 'weak'

        # If one loss requires transformed data (~pred or target), we apply transforms
        self.list_all_loss_targets = list(itertools.chain.from_iterable([self.all_loss_target[key] for key in self.all_loss_target.keys()]))
        if any('transformed' in key for key in self.all_loss_target.keys()) or any('transformed' in key for key in self.list_all_loss_targets):
            if self.augmentation_strength == 'weak':
                keys = ['data', 'label', 'image_embeddings', 'dense_embeddings', 'reshaped_probs', 'identity']
                mode = ['bilinear', 'nearest', 'bilinear', 'bilinear', 'bilinear', 'nearest']
                self.transforms = Compose([
                    RandFlipd(keys=keys, spatial_axis=[0, 1], prob=0.75),  # Flip along axes 0 and 1
                    RandRotate90d(keys=keys, spatial_axes=[0, 1], prob=0.75),  # Rotate by 90 degrees
                    RandRotated(keys=keys, mode=mode, range_x=0.25, prob=0.75, padding_mode='zeros')  # Rotate by a random angle
                ])
            elif self.augmentation_strength == 'stronger':
                keys = ['data', 'label', 'image_embeddings', 'dense_embeddings', 'reshaped_probs', 'identity']
                mode = ['bilinear', 'nearest', 'bilinear', 'bilinear', 'bilinear', 'nearest']
                self.transforms = Compose([
                    RandAffined(keys=keys, prob=1, rotate_range=0.25, translate_range=0.1, scale_range=0.1, padding_mode='zeros', mode=mode),
                    CenterSpatialCropd(keys=['image_embeddings', 'dense_embeddings'], roi_size=[64, 64])
                ])
                
        self.log_img_freq = 25  # Must be multiple of self.log_metric_freq

    def norm_batch(self, x):
        bs = x.shape[0]
        Isize = x.shape[-1]
        min_value = x.view(bs, -1).min(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
        max_value = x.view(bs, -1).max(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
        x = (x - min_value) / (max_value - min_value + 1e-6)
        return x

    def forward(self, batched_input):   
        
        try:
            image_embeddings = batched_input['image_embeddings']
        except KeyError:
            with torch.no_grad():       
                if 'sam_preprocessing' not in self.train_config.keys():
                    input_images = batched_input['data']     
                else:
                    imgs, original_sz = batched_input['data'], batched_input['original_sizes'][0]
                    img_sz = torch.tensor(batched_input['data'].shape[-2:])
                    orig_imgs = imgs.to(self.sam.device)
                    sam_batched_input = self.get_input_dict(orig_imgs, original_sz, img_sz)
                    input_images = torch.stack([self.sam.preprocess(x['image']) for x in sam_batched_input], dim=0)
                image_embeddings = self.sam.image_encoder(input_images)
        
        # We get the embeddings from the prompt module
        if self.module_input == 'image_orig_size':
            size_small = batched_input['label'].shape[2:]
            input_small = F.interpolate(batched_input['data'], size_small, mode='bilinear', align_corners=True)
            sparse_embeddings, dense_embeddings, _ = self.module(input_small)

        # If the prompt module does not output sparse or dense embeddings, we use SAM's default embeddings
        if 'sparse_embeddings' not in self.module_outputs or 'dense_embeddings' not in self.module_outputs:
            sparse_embeddings_none, dense_embeddings_none = self.sam.prompt_encoder(points=None, boxes=None, masks=None)
            if 'sparse_embeddings' not in self.module_outputs:
                sparse_embeddings = sparse_embeddings_none
            if 'dense_embeddings' not in self.module_outputs:
                dense_embeddings = dense_embeddings_none

        # We get the image positional embeddings
        with torch.no_grad():
            image_positional_embeddings = self.sam.prompt_encoder.get_dense_pe()

        pred_masks, low_res_pred_masks, iou_predictions = self.sam_forward(image_embeddings, image_positional_embeddings, 
                                                                           sparse_embeddings, dense_embeddings, batched_input)

        return image_embeddings, pred_masks, low_res_pred_masks, iou_predictions, image_positional_embeddings, \
                sparse_embeddings, dense_embeddings

    def _transform_embeddings(self, x, y, image_embeddings, dense_embeddings, low_res_logits):
        """
        We transform the embeddings of the input image to see the effect of the prompt module on the
        Args:
            x (tensor): input image. Shape (BS, 3, H, W)
            y (tensor): target mask. Shape (BS, 1, H, W)
            image_embeddings (tensor): image embeddings given by SAM. Shape (BS, 256, 64, 64)
            dense_embeddings (tensor): dense prompt embedding. Shape (BS, 256, 64, 64)
            low_res_logits (tensor): output logits of SAM. Shape (BS, 1, 256, 256)
        """
        with torch.no_grad():
            reshaped_logits = F.interpolate(low_res_logits,(image_embeddings.shape[-2], image_embeddings.shape[-1]), mode="bilinear", align_corners=False)
            reshaped_out_probs = self.norm_batch(reshaped_logits).type(torch.float)
            reshaped_probs = reshaped_out_probs.repeat(1, 2, 1, 1)
            reshaped_probs[:, 0, :, :] = 1 - reshaped_probs[:, 1, :, :]
            identity_mask = torch.ones(reshaped_probs.shape).to(self.device)  # shape (BS, 2, H, W)
            
            # Initialize list to hold transformed data
            transformed_data = {'data': [], 'label': [], 'image_embeddings': [], 'dense_embeddings': [], 'reshaped_probs': [], 'identity': []}
            
            # Loop over each sample in the batch
            for i in range(image_embeddings.shape[0]):
                # Extract each sample's data
                sample = {
                    'data': x[i, :, :, :],
                    'label': y[i, :, :, :], 
                    'image_embeddings': image_embeddings[i, :, :, :],
                    'dense_embeddings': dense_embeddings[i, :, :, :],
                    'reshaped_probs': reshaped_probs[i, :, :, :],
                    'identity': identity_mask[i, :, :, :]
                }
                
                # Apply the transforms
                transformed_sample = self.transforms(sample)
                
                # Collect transformed data
                transformed_data['data'].append(transformed_sample['data'])
                transformed_data['label'].append(transformed_sample['label'])
                transformed_data['image_embeddings'].append(transformed_sample['image_embeddings'])
                transformed_data['dense_embeddings'].append(transformed_sample['dense_embeddings'])
                transformed_data['reshaped_probs'].append(transformed_sample['reshaped_probs'])
                transformed_data['identity'].append(transformed_sample['identity'])

            # Recombine the transformed data into full batch tensors
            transf_data = torch.stack(transformed_data['data'], dim=0)
            transf_label = torch.stack(transformed_data['label'], dim=0)
            transf_image_embeddings = torch.stack(transformed_data['image_embeddings'], dim=0)
            transf_dense_embeddings = torch.stack(transformed_data['dense_embeddings'], dim=0)
            transf_probs = torch.stack(transformed_data['reshaped_probs'], dim=0)
            transf_identity_masks = torch.stack(transformed_data['identity'], dim=0)
            
        return transf_data, transf_label, transf_image_embeddings, transf_dense_embeddings, transf_probs, transf_identity_masks
                
    def _training_step(self, batch, batch_idx):
        """
        Batch should have keys:
            'data': (BS, 1, 3, H_target, W_target)
            'seg_mask': (BS, n_channels, 1, H_target, W_target)
            'boxes': (BS, n_channels, H_target, W_target)
            'point_coords': (BS, n_channels, 1, 2)
            'point_labels': (BS, n_channels, 1)
        """        
        torch.use_deterministic_algorithms(True, warn_only=True)  # Because cumsum_cuda_kernel does not have a deterministic implementation...
        self.sam.eval()  # We set SAM to eval mode to avoid dropout and batchnorm issues
        
        x, y, img_idx = batch['data'], batch['label'], batch['idx'] 
        image_embeddings, logits, low_res_logits, iou_predictions, image_positional_embeddings, sparse_embeddings, dense_embeddings = self.forward(batch)
        batch['image_embeddings'] = image_embeddings
        batch['image_positional_embeddings'] = image_positional_embeddings
        
        # We compute the probability map
        #out_probs = self.activation_fct(logits).type(torch.float)
        out_probs = self.norm_batch(logits).type(torch.float)
        probs = out_probs.repeat(1, 2, 1, 1)
        probs[:, 0, :, :] = 1 - probs[:, 1, :, :]
            
        # We compute the output with the GT bounding box 
        with torch.no_grad():
            pseudo_sparse_embeddings, pseudo_dense_embeddings = self.sam.prompt_encoder(points=None, boxes=batch['input_boxes'], masks=None)
            low_res_pseudo_logits, _ = self.sam.mask_decoder(
                image_embeddings=image_embeddings,  # (B, 256, 64, 64)
                image_pe=image_positional_embeddings[0:1], # self.sam.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=pseudo_sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=pseudo_dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False)
            low_res_pseudo_probs = torch.sigmoid(low_res_pseudo_logits)  # (1, 1, 256, 256)
            pseudo_probs = F.interpolate(
                low_res_pseudo_probs,
                size=(batch['label'].shape[-2], batch['label'].shape[-1]),
                mode="bilinear",
                align_corners=False)  # (1, 1, gt.shape)
            batch['gt_prompted_pred_masks'] = (pseudo_probs > 0.5).float()
            
        # We get SAM's output with transformed image embeddings and dense prompt embeddings
        if any('transformed' in key for key in self.all_loss_target.keys()) or any('transformed' in key for key in self.list_all_loss_targets): 
            transf_data, _transf_label, transf_image_embeddings, transf_dense_embeddings, \
                 _transf_probs, _transf_identity_masks =  self._transform_embeddings(x, y, image_embeddings, dense_embeddings, low_res_logits)
            transf_label = CenterSpatialCrop(roi_size=y.shape[-3:])(_transf_label).to(float)
            transf_probs = CenterSpatialCrop(roi_size=image_embeddings.shape[-3:])(_transf_probs)  
            transf_identity_masks = CenterSpatialCrop(roi_size=image_embeddings.shape[-3:])(_transf_identity_masks)
                      
            batch_transf = {'data': transf_data, 'image_embeddings': transf_image_embeddings, 
                            'label': transf_label, 'image_positional_embeddings': batch['image_positional_embeddings']}
            _, _, low_res_logits_transf, _, _, _, dense_embeddings_transf = self.forward(batch_transf)
        
            reshaped_logits_transf =  F.interpolate(low_res_logits_transf,(image_embeddings.shape[-2], image_embeddings.shape[-1]), 
                                                    mode="bilinear", align_corners=False)
            _probs_transf = self.norm_batch(reshaped_logits_transf).type(torch.float)
            probs_transf = _probs_transf.repeat(1, 2, 1, 1)
            probs_transf[:, 0, :, :] = 1 - probs_transf[:, 1, :, :]
            # We apply a mask to the transformed embeddings to discard padding differences
            batch['probs_transformed'] = probs_transf * transf_identity_masks
            batch['transformed_probs'] = transf_probs * transf_identity_masks
            reshaped_transf_identity_mask = transf_identity_masks[:, :1, :, :].repeat(1, dense_embeddings_transf.shape[1], 1, 1)
            batch['dense_embeddings_transformed'] = dense_embeddings_transf * reshaped_transf_identity_mask
            batch['transformed_dense_embeddings'] = transf_dense_embeddings * reshaped_transf_identity_mask
            
        # We compute the loss
        losses = []
        losses_dict = {}
        for key in self.loss_names.keys():
            for name, func, w in zip(self.loss_names[key], self.loss_fct[key], self.loss_weights[key]):
                _loss, kwargs = func(probs, batch, **self.loss_kwargs)
                losses.append(w * _loss)
                losses_dict['train_' + name] = _loss 
        loss = sum(losses)

        if self.current_epoch % self.log_metric_freq == 0:
            i = 0
            for pred_name in sorted(self.all_loss_names.keys(), key=lambda x: (x not in ['probs', 'prompt_mask'], x)):
                for _name, _w in zip(self.loss_names[pred_name], self.loss_weights[pred_name]): 
                    _loss = losses[i]
                    self.log('train/' + pred_name + '_' + _name, _loss / _w)
                    i += 1
            
        pred_masks = (out_probs > self.pred_threshold).float()
        metrics = self._compute_seg_metrics(pred_masks, y)

        if self.current_epoch % self.log_metric_freq == 0:
            self.log('train/loss', loss)
            for cur_metric in metrics.keys():
                self.log('train/{}'.format(cur_metric), metrics[cur_metric])
            
        return loss
    
    def _validation_step(self, batch, batch_idx):
        x, y, img_idx = batch['data'], batch['label'], batch['idx'] 

        image_embeddings, logits, low_res_logits, iou_predictions, image_positional_embeddings, sparse_embeddings, dense_embeddings = self.forward(batch)
        batch['image_embeddings'] = image_embeddings
        batch['image_positional_embeddings'] = image_positional_embeddings

        # out_probs = self.activation_fct(logits).type(torch.float)
        out_probs = self.norm_batch(logits).type(torch.float)
        probs = out_probs.repeat(1, 2, 1, 1)
        probs[:, 0, :, :] = 1 - probs[:, 1, :, :]

        # We compute the output with the GT bounding box 
        with torch.no_grad():
            pseudo_sparse_embeddings, pseudo_dense_embeddings = self.sam.prompt_encoder(points=None, boxes=batch['input_boxes'], masks=None)
            low_res_pseudo_logits, _ = self.sam.mask_decoder(
                image_embeddings=image_embeddings,  # (B, 256, 64, 64)
                image_pe=image_positional_embeddings[0:1], # self.sam.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=pseudo_sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=pseudo_dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False)
            low_res_pseudo_probs = torch.sigmoid(low_res_pseudo_logits)  # (1, 1, 256, 256)
            pseudo_probs = F.interpolate(
                low_res_pseudo_probs,
                size=(batch['label'].shape[-2], batch['label'].shape[-1]),
                mode="bilinear",
                align_corners=False)  # (1, 1, gt.shape)
            batch['gt_prompted_pred_masks'] = (pseudo_probs > 0.5).float()

        # We get SAM's output with transformed image embeddings and dense prompt embeddings
        if any('transformed' in key for key in self.all_loss_target.keys()) or any('transformed' in key for key in self.list_all_loss_targets): 
            transf_data, _transf_label, transf_image_embeddings, transf_dense_embeddings, \
                _transf_probs, _transf_identity_masks =  self._transform_embeddings(x, y, image_embeddings, dense_embeddings, low_res_logits)
            transf_label = CenterSpatialCrop(roi_size=y.shape[-3:])(_transf_label).to(float)
            transf_probs = CenterSpatialCrop(roi_size=image_embeddings.shape[-3:])(_transf_probs)  
            transf_identity_masks = CenterSpatialCrop(roi_size=image_embeddings.shape[-3:])(_transf_identity_masks)
            
            batch_transf = {'data': transf_data, 'image_embeddings': transf_image_embeddings, 'label': transf_label, 
                            'image_positional_embeddings': batch['image_positional_embeddings']}
            _, _, low_res_logits_transf, _, _, _, dense_embeddings_transf = self.forward(batch_transf)
        
            reshaped_logits_transf =  F.interpolate(low_res_logits_transf,(image_embeddings.shape[-2], image_embeddings.shape[-1]), 
                                                    mode="bilinear", align_corners=False)
            _probs_transf = self.norm_batch(reshaped_logits_transf).type(torch.float)
            probs_transf = _probs_transf.repeat(1, 2, 1, 1)
            probs_transf[:, 0, :, :] = 1 - probs_transf[:, 1, :, :]
            # We apply a mask to the transformed embeddings to discard padding differences
            batch['probs_transformed'] = probs_transf * transf_identity_masks
            batch['transformed_probs'] = transf_probs * transf_identity_masks
            reshaped_transf_identity_mask = transf_identity_masks[:, :1, :, :].repeat(1, dense_embeddings_transf.shape[1], 1, 1)
            batch['dense_embeddings_transformed'] = dense_embeddings_transf * reshaped_transf_identity_mask
            batch['transformed_dense_embeddings'] = transf_dense_embeddings * reshaped_transf_identity_mask
            
        # We compute the loss
        losses = []
        losses_dict = {}
        for key in self.loss_names.keys():
            for name, func, w in zip(self.loss_names[key], self.loss_fct[key], self.loss_weights[key]):
                _loss, kwargs = func(probs, batch, **self.loss_kwargs)
                losses.append(w * _loss)
                losses_dict['val_' + name] = _loss 
        loss = sum(losses)

        if self.current_epoch % self.log_metric_freq == 0:
            i = 0
            for pred_name in sorted(self.all_loss_names.keys(), key=lambda x: (x not in ['probs', 'prompt_mask'], x)):
                for _name, _w in zip(self.loss_names[pred_name], self.loss_weights[pred_name]): 
                    _loss = losses[i]
                    self.log('val/' + pred_name + '_' + _name, _loss / _w)
                    i += 1

        pred_masks = (out_probs > self.pred_threshold).float()
        metrics = self._compute_seg_metrics(pred_masks, y)

        if self.current_epoch % self.log_metric_freq == 0:
            self.log('val/loss', loss)
            for cur_metric in metrics.keys():
                self.log('val/{}'.format(cur_metric), metrics[cur_metric])

        self.val_outputs.append({'val_loss': loss, 'individual_losses': losses_dict, **metrics})

        return loss
    
    def _test_step(self, batch, batch_idx):
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        x, y, img_idx = batch['data'], batch['label'], batch['idx'].item()
        _, logits, _, _, _, _, _ = self.forward(batch)
        
        out_probs = self.norm_batch(logits).type(torch.float)
        probs = out_probs.repeat(1, 2, 1, 1)
        probs[:, 0, :, :] = 1 - probs[:, 1, :, :]
        pred_masks = (out_probs > self.pred_threshold).float()
        metrics = self._compute_seg_metrics(pred_masks, y)

        for cur_metric in metrics.keys():
            self.log('test/{}'.format(cur_metric), metrics[cur_metric])
    
        return metrics

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack([x['val_loss'] for x in self.val_outputs]).mean()
        avg_val_metrics = {k: torch.stack([x[k] for x in self.val_outputs]).mean() for k in self.val_outputs[0] if k not in ['val_loss', 'individual_losses']}
        avg_val_metric = avg_val_metrics[self.eval_metric]
        all_loss_names = list(itertools.chain.from_iterable([self.all_loss_names[key] for key in self.all_loss_target.keys()]))
        individual_losses = {name: torch.stack([x['individual_losses']['val_'+name] for x in self.val_outputs]).mean() for name in all_loss_names}

        if avg_val_metric > self.best_val_metric:
            self.best_val_metric = avg_val_metric
            self.best_epoch = self.current_epoch
            self.save_best_val_metric_and_losses(avg_val_metric, avg_val_metrics, individual_losses, self.best_epoch)
        
        self.save_last_val_losses_and_metrics(avg_val_loss, avg_val_metrics, individual_losses, self.current_epoch)

        # Clear val_outputs for the next epoch
        self.val_outputs.clear()

    def save_best_val_metric_and_losses(self, val_metric, val_metrics, individual_losses, epoch):
        with open(os.path.join(self.checkpoint_path, 'best_val_metrics_losses.txt'), 'w') as f:
            f.write(f'Best Validation Metric: {val_metric.item()}\n')
            f.write(f'Best Epoch: {epoch}\n')
            for name, loss in individual_losses.items():
                f.write(f'Val {name} Loss: {loss.item()}\n')
            for metric, value in val_metrics.items():
                f.write(f'Val {metric}: {value.item()}\n')

    def save_last_val_losses_and_metrics(self, val_loss, val_metrics, individual_losses, epoch):
        with open(os.path.join(self.checkpoint_path, 'last_val_metrics_losses.txt'), 'w') as f:
            f.write(f'Epoch: {epoch}\n')
            f.write(f'Validation Loss: {val_loss.item()}\n')
            for name, loss in individual_losses.items():
                f.write(f'Val {name} Loss: {loss.item()}\n')
            for metric, value in val_metrics.items():
                f.write(f'Val {metric}: {value.item()}\n')
            f.write('\n')