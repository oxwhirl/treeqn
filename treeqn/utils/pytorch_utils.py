#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:04:33 2017

@author: greg
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from numbers import Number
import math

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class FnModule(nn.Module):
    def __init__(self, fn):
        super(FnModule, self).__init__()
        self.fn = fn

    def forward(self, *input):
        return self.fn(*input)


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def count_parameters(model):
    total = 0
    for var in model.parameters():
        n = 1
        for x in var.size():
            n *= x
        total += n
    return total


def ortho_init(tensor, scale=1.0):
    if isinstance(tensor, Variable):
        ortho_init(tensor.data, scale=scale)
        return tensor

    shape = tensor.size()
    if len(shape) == 2:
        flat_shape = shape
    elif len(shape) == 4:
        flat_shape = (shape[0] * shape[2] * shape[3], shape[1])  # NCHW
    else:
        raise NotImplementedError
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v  # pick the one with the correct shape
    q = q.reshape(shape)
    w = (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    tensor.copy_(torch.FloatTensor(w))
    return tensor


def nn_init(module, w_init=ortho_init, w_scale=1.0, b_init=nn.init.constant, b_scale=0.0):
    w_init(module.weight, w_scale)
    b_init(module.bias, b_scale)
    return module

USE_CUDA = torch.cuda.is_available()

def cudify(x):
    if USE_CUDA:
        return x.cuda()
    else:
        return x