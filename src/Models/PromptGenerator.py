"""
Author: Mélanie Gaillochet
"""
import torch.nn as nn
from torch.nn import functional as F

from Models.modules_promptgenerator import HarDNet, SmallDecoder


class ModelEmb(nn.Module):
    """ Modified from 'AutoSAM: Adapting SAM to Medical Images by Overloading the Prompt Encoder'
    https://github.com/talshaharabany/AutoSAM/blob/main/models/model_single.py
    """
    def __init__(self, **kwargs):
        super(ModelEmb, self).__init__()
        self.hardnet_depth_wise = bool(int(kwargs.get('depth_wise')))
        self.hardnet_arch = int(kwargs.get('order'))
        self.in_channels = kwargs.get('in_channels', 3)
        
        if self.in_channels == 256:
            self.conv1 = nn.Conv2d(256, 3, kernel_size=1, stride=1, padding=0)
        
        self.backbone = HarDNet(in_channels=3, 
                                depth_wise=self.hardnet_depth_wise, arch=self.hardnet_arch, args=kwargs)
        d, f = self.backbone.full_features, self.backbone.features # full_features: [96, 192, 320, 720, 1280]
        self.decoder = SmallDecoder(d, out=256)
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        if self.in_channels == 256:
            x = self.conv1(x)
            
        z = self.backbone(x) # len(z) = 5, (BS, 96, H/2, W/2), (BS, 192, H/4, W/4), (BS, 320, H/8, W/8), (BS, 720, H/16, W/16), (BS, 1280, H/32, W/32)
        dense_embeddings = self.decoder(z) # (BS, 256, H/4, W/4)
        dense_embeddings = F.interpolate(dense_embeddings, (64, 64), mode='bilinear', align_corners=True)
        
        sparse_embeddings = None

        return sparse_embeddings, dense_embeddings, None


promptmodule_zoo = {
            'module_hardnet': ModelEmb
            }
