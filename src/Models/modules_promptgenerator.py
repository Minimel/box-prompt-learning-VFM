"""
Codea adapted from official repository of 'AutoSAM: Adapting SAM to Medical Images by Overloading the Prompt Encoder'
https://github.com/talshaharabany/AutoSAM/blob/main/models/hardnet.py
https://github.com/talshaharabany/AutoSAM/blob/main/models/base.py
https://github.com/talshaharabany/AutoSAM/blob/main/models/model_single.py
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptationMismatch(Exception): pass


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.data.size(0), -1)


class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, dropout=0.1, bias=False):
        super().__init__()
        self.add_module('layer1', ConvLayer(in_channels, out_channels, kernel))
        self.add_module('layer2', DWConvLayer(out_channels, out_channels, stride=stride))

    def forward(self, x):
        return super().forward(x)


class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__()
        out_ch = out_channels

        groups = in_channels
        kernel = 3
        # print(kernel, 'x', kernel, 'x', out_channels, 'x', out_channels, 'DepthWise')

        self.add_module('dwconv', nn.Conv2d(groups, groups, kernel_size=3,
                                            stride=stride, padding=1, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(groups))

    def forward(self, x):
        return super().forward(x)


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, bias=False):
        super().__init__()
        out_ch = out_channels
        groups = 1
        # print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)
        self.add_module('conv', nn.Conv2d(in_channels, out_ch, kernel_size=kernel,
                                          stride=stride, padding=kernel // 2, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(out_ch))
        self.add_module('relu', nn.ReLU6(True))

    def forward(self, x):
        return super().forward(x)


class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0  # if upsample else in_channels
        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out
            if dwconv:
                layers_.append(CombConvLayer(inch, outch))
            else:
                layers_.append(ConvLayer(inch, outch))

            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        # print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]

        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)

        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or \
                    (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out


class HarDNet(nn.Module):
    def __init__(self, in_channels=3, depth_wise=False, arch=85, pretrained=True, weight_path='', out=1, args=None):
        super().__init__()
        first_ch = [32, 64]
        second_kernel = 3
        max_pool = True
        grmul = 1.7
        drop_rate = 0.1

        # HarDNet68
        ch_list = [128, 256, 320, 640, 1024]
        gr = [14, 16, 20, 40, 160]
        n_layers = [8, 16, 16, 16, 4]
        downSamp = [1, 0, 1, 1, 0]

        if arch == 85:
            # HarDNet85
            first_ch = [48, 96]
            ch_list = [192, 256, 320, 480, 720, 1280]
            gr = [24, 24, 28, 36, 48, 256]
            n_layers = [8, 16, 16, 16, 16, 4]
            downSamp = [1, 0, 1, 0, 1, 0]
            drop_rate = 0.2
        elif arch == 39:
            # HarDNet39
            first_ch = [24, 48]
            ch_list = [96, 320, 640, 1024]
            grmul = 1.6
            gr = [16, 20, 64, 160]
            n_layers = [4, 16, 8, 4]
            downSamp = [1, 1, 1, 0]

        if depth_wise:
            second_kernel = 1
            max_pool = False
            drop_rate = 0.05

        blks = len(n_layers)
        self.base = nn.ModuleList([])

        # First Layer: Standard Conv3x3, Stride=2
        self.base.append(
            ConvLayer(in_channels, out_channels=first_ch[0], kernel=3,
                      stride=2, bias=False))

        # Second Layer
        self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=second_kernel))

        # Maxpooling or DWConv3x3 downsampling
        if max_pool:
            self.base.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.base.append(DWConvLayer(first_ch[1], first_ch[1], stride=2))

        # Build all HarDNet blocks
        ch = first_ch[1]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise)
            ch = blk.get_out_ch()
            self.base.append(blk)

            if i == blks - 1 and arch == 85:
                self.base.append(nn.Dropout(0.1))

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            if downSamp[i] == 1:
                if max_pool:
                    self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    self.base.append(DWConvLayer(ch, ch, stride=2))

        ch = ch_list[blks - 1]
        self.base.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Dropout(drop_rate),
                nn.Linear(ch, 1000)))

        # print(self.base)

        if pretrained:
            if hasattr(torch, 'hub'):

                if arch == 68 and not depth_wise:
                    checkpoint = 'https://ping-chao.com/hardnet/hardnet68-5d684880.pth'
                elif arch == 85 and not depth_wise:
                    checkpoint = 'https://ping-chao.com/hardnet/hardnet85-a28faa00.pth'
                elif arch == 68 and depth_wise:
                    checkpoint = 'https://ping-chao.com/hardnet/hardnet68ds-632474d2.pth'
                else:
                    checkpoint = 'https://ping-chao.com/hardnet/hardnet39ds-0e6c6fa9.pth'

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False, map_location=device))

            else:
                postfix = 'ds' if depth_wise else ''
                weight_file = '%shardnet%d%s.pth' % (weight_path, arch, postfix)
                if not os.path.isfile(weight_file):
                    print(weight_file, 'is not found')
                    exit(0)
                weights = torch.load(weight_file)
                self.load_state_dict(weights)

            postfix = 'DS' if depth_wise else ''
            print('ImageNet pretrained weights for HarDNet%d%s is loaded' % (arch, postfix))
            if arch == 39:
                self.features = 640
                # self.base = self.base[0:14]
                self.base = self.base[0:11]
            elif arch == 68:
                self.features = 1024
                self.base = self.base[0:16]
            elif arch == 85:
                self.features = 1280
                self.base = self.base[0:19]
            if arch == 39:
                self.full_features = [48, 96, 320, 640, 1024]
                self.list = [1, 4, 7, 10, 13]
            elif arch == 68:
                self.full_features = [64, 128, 320, 640, 1024]
                self.list = [1, 4, 9, 12, 15]
            elif arch == 85:
                self.full_features = [96, 192, 320, 720, 1280]
                self.list = [1, 4, 9, 14, 18]

    def forward(self, x):
        for inx, layer in enumerate(self.base):
            x = layer(x)
            if inx == self.list[0]:
                x2 = x
                if inx == len(self.base) - 1:
                    return x2
            elif inx == self.list[1]:
                x4 = x
                if inx == len(self.base) - 1:
                    return x2, x4
            elif inx == self.list[2]:
                x8 = x
                if inx == len(self.base) - 1:
                    return x2, x4, x8
            elif inx == self.list[3]:
                x16 = x
                if inx == len(self.base) - 1:
                    return x2, x4, x8, x16
            elif inx == self.list[4]:
                x32 = x
                if inx == len(self.base) - 1:
                    return x2, x4, x8, x16, x32
                

class SmallDecoder(nn.Module):
    def __init__(self, full_features, out):
        super(SmallDecoder, self).__init__()
        self.up1 = UpBlockSkip(full_features[3] + full_features[2], full_features[2],
                               func='relu', drop=0)
        self.up2 = UpBlockSkip(full_features[2] + full_features[1], full_features[1],
                               func='relu', drop=0)
        self.final = CNNBlock(full_features[1], out, kernel_size=3, drop=0)

    def forward(self, x):
        z = self.up1(x[3], x[2])
        z = self.up2(z, x[1])
        out = F.tanh(self.final(z))
        # out = self.final(z)
        return out


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop=0):
        super(CNNBlock, self).__init__()
        P = int((kernel_size-1)/2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv1_drop = nn.Dropout2d(drop)
        self.conv2_drop = nn.Dropout2d(drop)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.BN2 = nn.BatchNorm2d(out_channels)

    def forward(self, x_in, inx=-1):
        x = self.conv1_drop(self.conv1(x_in))
        x = F.relu(self.BN1(x))
        x_out = self.conv2(x)
        return x_out


class UpBlockSkip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, func=None, drop=0):
        super(UpBlockSkip, self).__init__()
        P = int((kernel_size - 1) / 2)
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv1_drop = nn.Dropout2d(drop)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv2_drop = nn.Dropout2d(drop)
        self.BN = nn.BatchNorm2d(out_channels)
        self.func = func

    def forward(self, x_in, x_up):
        x = self.Upsample(x_in)
        x_cat = torch.cat((x, x_up), 1)
        x1 = self.conv2_drop(self.conv2(self.conv1_drop(self.conv1(x_cat))))
        if self.func == 'tanh':
            return F.tanh(self.BN(x1))
        elif self.func == 'relu':
            return F.leaky_relu(self.BN(x1))
        elif self.func == 'sigmoid':
            return F.sigmoid(self.BN(x1))
        else:
            return x1


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop=0, func=None):
        super(UpBlock, self).__init__()
        d = drop
        P = int((kernel_size - 1) / 2)
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv1_drop = nn.Dropout2d(d)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv2_drop = nn.Dropout2d(d)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.BN2 = nn.BatchNorm2d(out_channels)
        self.func = func

    def forward(self, x_in):
        x = self.Upsample(x_in)
        x = self.conv1_drop(self.conv1(x))
        x = F.relu(self.BN1(x))
        x = self.conv2_drop(self.conv2(x))
        if self.func == 'None':
            return x
        elif self.func == 'tanh':
            return F.tanh(self.BN2(x))
        elif self.func == 'relu':
            return F.relu(self.BN2(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop=0):
        super(DownBlock, self).__init__()
        P = int((kernel_size -1 ) /2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=P)
        self.pool = nn.MaxPool2d((2, 2))
        self.conv1_drop = nn.Dropout2d(drop)
        self.conv2_drop = nn.Dropout2d(drop)
        self.BN = nn.BatchNorm2d(out_channels)

    def forward(self, x_in):
        x1 = self.conv2_drop(self.conv2(self.conv1_drop(self.conv1(x_in))))
        x1_pool = F.relu(self.BN(self.pool(x1)))
        return x1, x1_pool


class Encoder(nn.Module):
    def __init__(self, AEdim, drop=0):
        super(Encoder, self).__init__()
        self.full_features = [AEdim, AEdim*2, AEdim*4, AEdim*8, AEdim*8]
        self.down1 = DownBlock(3, AEdim, drop=drop)
        self.down2 = DownBlock(AEdim, AEdim*2, drop=drop)
        self.down3 = DownBlock(AEdim*2, AEdim*4, drop=drop)
        self.down4 = DownBlock(AEdim*4, AEdim*8, drop=drop)

    def forward(self, x_in):
        x1, x1_pool = self.down1(x_in)
        x2, x2_pool = self.down2(x1_pool)
        x3, x3_pool = self.down3(x2_pool)
        x4, x4_pool = self.down4(x3_pool)
        return x1, x2, x3, x4, x4_pool


class MMDecoder(nn.Module):
    def __init__(self, full_features, out_channel, z_size, out_size):
        super(MMDecoder, self).__init__()
        self.bottleneck = BottleneckBlock(full_features[4], z_size)
        self.up0 = UpBlock(z_size, full_features[3],
                           func='relu', drop=0).cuda()
        self.up1 = UpBlock(full_features[3], out_channel,
                           func='None', drop=0).cuda()
        self.out_size = out_size

    def forward(self, z, z_text):
        zz = self.bottleneck(z)
        zz_norm = zz / zz.norm(dim=1).unsqueeze(dim=1)
        attn_map = (zz_norm * z_text.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdims=True)
        zz = zz * attn_map
        zz = self.up0(zz)
        zz = self.up1(zz)
        zz = F.interpolate(zz, size=self.out_size, mode="bilinear", align_corners=True)
        return F.sigmoid(zz)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop=0):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv1_drop = nn.Dropout2d(drop)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.conv2_drop = nn.Dropout2d(drop)
        self.BN2 = nn.BatchNorm2d(out_channels)
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x_in):
        x = self.conv1_drop(self.conv1(x_in))
        x = F.relu(self.BN1(x))
        x = self.conv2_drop(self.conv2(x))
        x = F.relu(self.BN2(x))
        return x
