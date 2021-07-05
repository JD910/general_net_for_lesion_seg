from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    BatchNorm2d, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
import torch.nn.functional as F
import torchvision.models as models


class SpatialAttention(Module):
    def __init__(self, in_dim, kernel_size=9):
        super(SpatialAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
        
    def forward(self, x):

        x = F.interpolate(x, scale_factor=0.5, mode="bilinear")
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        out = F.interpolate(x, scale_factor=2, mode="bilinear")
        return out


class ChannelwiseAttention(Module):
    def __init__(self, in_dim):
        super(ChannelwiseAttention, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)

    def forward(self, x):

        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x

        attention_weight = F.adaptive_avg_pool2d(x, (1, 1)).view((m_batchsize, C))
        attention_weight = torch.mean(attention_weight)
        
        return out

class CPFE(Module):
    def __init__(self, feature_layer=None, out_channels=32):
        super(CPFE, self).__init__()

        self.dil_rates = [1, 2, 3]

        if feature_layer == 'conv5_3':
            self.in_channels = 512
            self.out_channels = 128 #512/4
        elif feature_layer == 'conv4_3':
            self.in_channels = 512
            self.out_channels = 128 #512/4
        elif feature_layer == 'conv3_3':
            self.in_channels = 256
            self.out_channels = 64 #256/4
        elif feature_layer == 'conv2_2':
            self.in_channels = 128
            self.out_channels = 32 #128/4
        elif feature_layer == 'conv1_2':
            self.in_channels = 64
            self.out_channels = 16 #64/4

        self.conv_1_1 = Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, bias=True)
        self.conv_dil_3 = Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[0], padding=self.dil_rates[0], bias=True)
        self.conv_dil_5 = Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[1], padding=self.dil_rates[1], bias=True)
        self.conv_dil_7 = Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[2], padding=self.dil_rates[2], bias=True)

        self.bn = BatchNorm2d(self.out_channels*4)

    def forward(self, input_):

        conv_1_1_feats = self.conv_1_1(input_)
        conv_dil_3_feats = self.conv_dil_3(input_)
        conv_dil_5_feats = self.conv_dil_5(input_)
        conv_dil_7_feats = self.conv_dil_7(input_)

        # Aggregate features
        concat_feats = torch.cat((conv_1_1_feats, conv_dil_3_feats, conv_dil_5_feats, conv_dil_7_feats), dim=1)
        bn_feats = F.relu(self.bn(concat_feats))

        return bn_feats
        
