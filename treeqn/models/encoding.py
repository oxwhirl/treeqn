import torch.nn as nn
import numpy as np
from treeqn.utils.pytorch_utils import nn_init


def atari_encoder(in_channels):
    encoder = nn.Sequential(
        nn_init(nn.Conv2d(in_channels, 16, kernel_size=8, stride=4), w_scale=np.sqrt(2)),
        nn.ReLU(True),
        nn_init(nn.Conv2d(16, 32, kernel_size=4, stride=2), w_scale=np.sqrt(2)),
        nn.ReLU(True),
    )
    return encoder


def push_encoder(in_channels):
    encoder = nn.Sequential(
        nn_init(nn.Conv2d(in_channels, 24, kernel_size=3, stride=1), w_scale=1.0),
        nn.ReLU(inplace=True),
        nn_init(nn.Conv2d(24, 24, kernel_size=3, stride=1), w_scale=1.0),
        nn.ReLU(inplace=True),
        nn_init(nn.Conv2d(24, 48, kernel_size=4, stride=2), w_scale=1.0),
        nn.ReLU(inplace=True),
    )
    return encoder