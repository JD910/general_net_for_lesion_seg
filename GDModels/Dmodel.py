from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tarfile, sys, math
from six.moves import urllib
from .ops import conv2d, deconv2d, Residual_G, Residual_D
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid
from torch.autograd import grad as torch_grad


class Discriminator(nn.Module):
    def __init__(self, spectral_normed, num_rotation,
                ssup, channel, resnet = False):
        super(Discriminator, self).__init__()
        self.resnet = False
        self.num_rotation = num_rotation
        self.ssup = ssup

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.conv1 = conv2d(channel, 32, kernel_size = 3, stride = 1, padding = 1,
                            spectral_normed = spectral_normed)
        self.conv2 = conv2d(32, 64, spectral_normed = spectral_normed,
                            padding = 0)
        self.conv3 = conv2d(64, 128, spectral_normed = spectral_normed,
                            padding = 0)
        self.conv4 = conv2d(128, 256, spectral_normed = spectral_normed,
                            padding = 0)
        self.fully_connect_gan1 = nn.Linear(230400, 1)
        self.fully_connect_rot1 = nn.Linear(230400, self.num_rotation)
        self.softmax = nn.Softmax(dim = 1)


    def forward(self, x):

        conv1 = self.lrelu(self.conv1(x))
        conv2 = self.lrelu(self.conv2(conv1))
        conv3 = self.lrelu(self.conv3(conv2))
        conv4 = self.lrelu(self.conv4(conv3))

        conv4 = conv4.view(conv4.size(0), -1)
        gan_logits = self.fully_connect_gan1(conv4)
        if self.ssup:
            rot_logits = self.fully_connect_rot1(conv4)
            rot_prob = self.softmax(rot_logits)

        if self.ssup:
            return self.sigmoid(gan_logits), gan_logits, rot_logits, rot_prob
        else:
            return self.sigmoid(gan_logits), gan_logits




