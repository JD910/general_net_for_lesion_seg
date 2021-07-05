from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .attention import SpatialAttention, ChannelwiseAttention, CPFE


class pub(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(pub, self).__init__()
        layers = [
                    nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                    nn.ReLU(True)
                 ]
        if batch_norm:
            layers.insert(1, nn.BatchNorm2d(out_channels))
            layers.insert(len(layers)-1, nn.BatchNorm2d(out_channels))
        self.pub = nn.Sequential(*layers)

    def forward(self, x):
        return self.pub(x)

class unet2dDown(nn.Module):

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(unet2dDown, self).__init__()
        self.pub = pub(in_channels, out_channels, batch_norm)
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))

    def forward(self, x):
        x = self.pool(x)
        x = self.pub(x)
        return x

class unet2dUp(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True, sample=True):
        super(unet2dUp, self).__init__()
        self.pub = pub(int(in_channels/2+in_channels), out_channels, batch_norm)
        if sample:
            self.sample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.sample = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)

    def forward(self, x, x1):
        x = self.sample(x)

        x2 = torch.cat((x, x1), dim=1)

        x2 = self.pub(x2)

        return x2

class UNet(nn.Module):
    def __init__(self, init_channels=1, class_nums=1, batch_norm=True, sample=True):
        super(UNet, self).__init__()
        self.down1 = pub(init_channels, 64, batch_norm)
        self.down2 = unet2dDown(64, 128, batch_norm)
        self.down3 = unet2dDown(128, 256, batch_norm)
        self.down4 = unet2dDown(256, 512, batch_norm)
        self.up3 = unet2dUp(512, 256, batch_norm, sample)
        self.up2 = unet2dUp(256, 128, batch_norm, sample)
        self.up1 = unet2dUp(128, 64, batch_norm, sample)
        self.con_last = nn.Conv2d(64, class_nums, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        return x1, x2, x3, x4
        
        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.con_last(x)
        return self.sigmoid(x)

class SODModel(nn.Module):
    def __init__(self, class_nums=1, batch_norm=True, sample=True):
        super(SODModel, self).__init__()

        self.cpfe_conv1_2 = CPFE(feature_layer='conv1_2')
        self.cpfe_conv2_2 = CPFE(feature_layer='conv2_2')
        self.cpfe_conv3_3 = CPFE(feature_layer='conv3_3')
        self.cpfe_conv4_3 = CPFE(feature_layer='conv4_3')
        self.cpfe_conv5_3 = CPFE(feature_layer='conv5_3')

        self.cha_att = ChannelwiseAttention(in_dim=384)  

        self.spa_att = SpatialAttention(in_dim=64)

        self.unet = UNet()

        self.up3 = unet2dUp(512, 256, batch_norm, sample)
        self.up2 = unet2dUp(256, 128, batch_norm, sample)
        self.up1 = unet2dUp(128, 64, batch_norm, sample)
        self.con_last = nn.Conv2d(64, class_nums, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_):

        conv1_2, conv2_2, conv3_3, conv4_3 = self.unet(input_)
        
        conv3_3_cpfe = self.cpfe_conv3_3(conv3_3)
        conv3_3_ca = self.cha_att(conv3_3_cpfe)
        x = self.up3(conv4_3, conv3_3_ca)

        conv2_2_cpfe = self.cpfe_conv2_2(conv2_2)
        conv2_2_ca = self.cha_att(conv2_2_cpfe)
        x = self.up2(x, conv2_2_ca)

        conv1_2_cpfe = self.cpfe_conv1_2(conv1_2)
        conv1_2_ca = self.cha_att(conv1_2_cpfe)
        x = self.up1(x, conv1_2_ca)

        x = self.con_last(x)

        return self.sigmoid(x)
