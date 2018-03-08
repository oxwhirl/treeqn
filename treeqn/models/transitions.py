import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np

from treeqn.utils.pytorch_utils import View, nn_init

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


def build_transition_fn(name, embedding_dim, nonlin=nn.Tanh(), num_actions=3, kernel_size=None):
    if name == "matrix":
        transition_fun = Parameter(torch.Tensor(embedding_dim, embedding_dim, num_actions).type(dtype))
        return nn.init.xavier_normal(transition_fun)
    elif name == "two_layer":
        transition_fun1 = nn.Linear(embedding_dim, embedding_dim)
        transition_fun2 = Parameter(torch.Tensor(embedding_dim, embedding_dim, num_actions).type(dtype))
        return transition_fun1, nn.init.xavier_normal(transition_fun2),
    else:
        raise ValueError


class MLPRewardFn(nn.Module):
    def __init__(self, embed_dim, num_actions):
        super(MLPRewardFn, self).__init__()
        self.embedding_dim = embed_dim
        self.num_actions = num_actions
        self.mlp = nn.Sequential(
            nn_init(nn.Linear(embed_dim, 64), w_scale=np.sqrt(2)),
            nn.ReLU(inplace=True),
            nn_init(nn.Linear(64, num_actions), w_scale=0.01)
        )

    def forward(self, x):
        x = x.view(-1, self.embedding_dim)
        return self.mlp(x).view(-1, self.num_actions)