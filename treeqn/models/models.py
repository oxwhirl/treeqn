import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from treeqn.utils.pytorch_utils import View, nn_init
from treeqn.utils.einsum import einsum
from treeqn.models.transitions import build_transition_fn, MLPRewardFn
from treeqn.models.encoding import atari_encoder, push_encoder
from treeqn.utils.pytorch_utils import logsumexp

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


class TreeQNPolicy(nn.Module):
    def __init__(self,
                 ob_space,
                 ac_space,
                 nenv,
                 nsteps,
                 nstack,
                 use_actor_critic=False,
                 transition_fun_name="matrix",
                 transition_nonlin="tanh",
                 normalise_state=True,
                 residual_transition=True,
                 tree_depth=2,
                 embedding_dim=512,
                 predict_rewards=True,
                 gamma=0.99,
                 td_lambda=0.8,
                 input_mode="atari",
                 value_aggregation="softmax",
                 output_tree=False):
        super(TreeQNPolicy, self).__init__()
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nc * nstack, nh, nw)
        self.nenv = nenv
        self.num_actions = ac_space.n
        self.embedding_dim = embedding_dim
        self.use_actor_critic = use_actor_critic
        self.obs_scale = 1.0
        self.eps_threshold = 0
        self.predict_rewards = predict_rewards
        self.gamma = gamma
        self.output_tree = output_tree
        self.td_lambda = td_lambda
        self.residual_transition = residual_transition
        if transition_fun_name == "two_layer":
            # we are going to introduce residuals manually inside the transition function, so turning this off...
            self.residual_transition = False
        self.normalise_state = normalise_state
        self.value_aggregation = value_aggregation

        self.embedding_dim = embedding_dim

        if input_mode == "atari":
            encoder = atari_encoder(ob_shape[1])
            dummy = Variable(torch.zeros(1, *ob_shape[1:]))
            conv_dim_out = tuple(encoder(dummy).size())[1:]
            self.obs_scale = 255.0
        elif input_mode == "push":
            encoder = push_encoder(ob_shape[1])
            dummy = Variable(torch.zeros(1, *ob_shape[1:]))
            conv_dim_out = tuple(encoder(dummy).size())[1:]
        else:
            raise ValueError("Input mode not accepted. use atari, push")

        print("CONV DIM OUT", conv_dim_out)
        flat_conv_dim_out = int(np.prod(conv_dim_out))

        self.embed = nn.Sequential(
            encoder,
            View(-1, flat_conv_dim_out),
            nn_init(nn.Linear(flat_conv_dim_out, self.embedding_dim), w_scale=np.sqrt(2)),
            nn.ReLU(True)
        )

        self.value_fn = nn_init(nn.Linear(embedding_dim, 1), w_scale=.01)

        if self.use_actor_critic:
            self.ac_value_fn = nn_init(nn.Linear(embedding_dim, 1), w_scale=1.0)

        self.transition_fun_name = transition_fun_name
        if transition_nonlin == "tanh":
            self.transition_nonlin = nn.Tanh()
        elif transition_nonlin == "relu":
            self.transition_nonlin = nn.ReLU()
        else:
            raise ValueError

        if self.transition_fun_name == "two_layer":
            self.transition_fun1, self.transition_fun2 = \
                build_transition_fn(transition_fun_name, embedding_dim, nonlin=self.transition_nonlin,
                                    num_actions=self.num_actions)
        else:
            self.transition_fun = build_transition_fn(transition_fun_name, embedding_dim, nonlin=self.transition_nonlin,
                                                      num_actions=self.num_actions)

        if self.predict_rewards:
            self.tree_reward_fun = MLPRewardFn(embedding_dim, self.num_actions)

        self.tree_depth = tree_depth

    def forward(self, ob, volatile=False):
        """
        :param ob: [batch_size x channels x height x width]
        :return: [batch_size x num_actions], -- Q-values
                 [batch_size x 1], -- V = max_a(Q)
                 [batch_size x num_actions x embedding_dim], -- embeddings after first transition
                 [batch_size x num_actions] -- rewards after first transition
        """

        st = self.embed_obs(ob, volatile=volatile)

        if self.normalise_state:
            st = st / st.pow(2).sum(-1, keepdim=True).sqrt()

        Q, tree_result = self.planning(st)

        if self.use_actor_critic:
            V = self.ac_value_fn(st).squeeze()
        else:
            V = torch.max(Q, 1)[0]

        return Q, V, tree_result

    def obs_to_variable(self, ob, volatile=False):
        ob = ob.transpose(0, 3, 1, 2)
        ob = Variable(torch.from_numpy(ob / self.obs_scale).type(dtype), volatile=volatile)
        return ob

    def embed_obs(self, ob, volatile=False):
        ob = self.obs_to_variable(ob, volatile=volatile)

        # -- [batch_size x embedding_dim]
        st = self.embed(ob)
        return st

    def step(self, ob):
        Q, V, _ = self.forward(ob, volatile=True)
        a = self.sample(Q)
        return a, V

    def value(self, ob):
        _, V, _ = self.forward(ob, volatile=True)
        return V

    def sample(self, Q):
        if self.use_actor_critic:
            pi = F.softmax(Q, dim=-1)
            a = torch.multinomial(pi, 1).squeeze()
            return a.data.cpu().numpy()
        else:
            sample = random.random()
            if sample > self.eps_threshold:
                return Q.data.max(1)[1].cpu().numpy()
            else:
                return np.random.randint(0, self.num_actions, self.nenv)

    def tree_planning(self, x, return_intermediate_values=True):
        """
        :param x: [batch_size x embedding_dim]
        :return:
            dict tree_result:
            - "embeddings":
                list of length tree_depth, [batch_size * num_actions^depth x embedding_dim] state
                representations after tree planning
            - "values":
                list of length tree_depth, [batch_size * num_actions^depth x 1] values predicted
                from each embedding
            - "rewards":
                list of length tree_depth, [batch_size * num_actions^depth x 1] rewards predicted
                from each transition
        """

        tree_result = {
            "embeddings": [x],
            "values": []
        }
        if self.predict_rewards:
            tree_result["rewards"] = []

        if return_intermediate_values:
            tree_result["values"].append(self.value_fn(x))

        for i in range(self.tree_depth):
            if self.predict_rewards:
                r = self.tree_reward_fun(x)
                tree_result["rewards"].append(r.view(-1, 1))

            x = self.tree_transitioning(x)

            x = x.view(-1, self.embedding_dim)

            tree_result["embeddings"].append(x)

            if return_intermediate_values or i == self.tree_depth - 1:
                tree_result["values"].append(self.value_fn(x))

        return tree_result

    def tree_transitioning(self, x):
        """
        :param x: [? x embedding_dim]
        :return: [? x num_actions x embedding_dim]
        """

        if self.transition_fun_name == "matrix":
            temp = self.transition_nonlin(einsum("ij,jab->iba", x, self.transition_fun))
            temp = temp.contiguous()
            next_state = temp
        elif self.transition_fun_name == "two_layer":
            x1 = self.transition_nonlin(self.transition_fun1(x))
            x2 = x + x1
            x2 = x2.unsqueeze(1)
            x3 = self.transition_nonlin(einsum("ij,jab->iba", x, self.transition_fun2))
            x2 = x2.expand_as(x3)
            next_state = x2 + x3
        else:
            next_state = self.transition_fun(x)

        if self.residual_transition:
            next_state = x.unsqueeze(1).expand_as(next_state) + next_state

        if self.normalise_state:
            next_state = next_state / next_state.pow(2).sum(-1, keepdim=True).sqrt()

        return next_state

    def planning(self, x):
        """
        :param x: [batch_size x embedding_dim] state representations
        :return:
            - [batch_size x embedding_dim x num_actions] state-action values
            - [batch_size x num_actions x embedding_dim] state representations after planning one step
              used for regularizing/grounding the transition model
        """
        batch_size = x.size(0)
        if self.tree_depth > 0:
            tree_result = self.tree_planning(x)
        else:
            raise NotImplementedError

        q_values = self.tree_backup(tree_result, batch_size)

        return q_values, tree_result

    def tree_backup(self, tree_result, batch_size):
        backup_values = tree_result["values"][-1]
        for i in range(1, self.tree_depth + 1):
            one_step_backup = tree_result["rewards"][-i] + self.gamma*backup_values

            if i < self.tree_depth:
                one_step_backup = one_step_backup.view(batch_size, -1, self.num_actions)

                if self.value_aggregation == "max":
                    max_backup = one_step_backup.max(2)[0]
                elif self.value_aggregation == "logsumexp":
                    max_backup = logsumexp(one_step_backup, 2)
                elif self.value_aggregation == "softmax":
                    max_backup = (one_step_backup * F.softmax(one_step_backup, dim=2)).sum(dim=2)
                else:
                    raise ValueError("Unknown value aggregation function %s" % self.value_aggregation)

                backup_values = ((1 - self.td_lambda) * tree_result["values"][-i-1] +
                                 (self.td_lambda) * max_backup.view(-1, 1))
            else:
                backup_values = one_step_backup

        backup_values = backup_values.view(batch_size, self.num_actions)

        return backup_values


class DQNPolicy(TreeQNPolicy):
    """
    Vanilla DQN - just fully connected after conv encoder
    """

    def __init__(self, *args, extra_layers=0, **kwargs):
        super(DQNPolicy, self).__init__(*args, **kwargs)
        # Just to make sure that calculating number of parameters is correct
        self.transition_fun = None
        self.value_fn = None

        self.q_fn = nn.Linear(self.embedding_dim, self.num_actions)

        self.extra_layers = extra_layers
        if extra_layers > 0:
            self.transition_fun = \
                nn.init.xavier_normal(nn.Parameter(torch.Tensor(self.embedding_dim, self.embedding_dim)))

    def planning(self, x):
        tree_result = {
            "embeddings": [x],
        }

        if self.extra_layers > 0:
            for i in range(self.extra_layers):
                x = x + self.transition_nonlin(einsum("ij,ja->ia", x, self.transition_fun))
                x = x / x.pow(2).sum(-1, keepdim=True).sqrt()

        q = self.q_fn(x)

        return q, tree_result
