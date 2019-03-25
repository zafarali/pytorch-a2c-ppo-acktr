import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Categorical, DiagGaussian
from a2c_ppo_acktr.models import Policy

class MixCategorical(Categorical):
    def __init__(self, num_inputs, num_outputs):
        super(MixCategorical, self).__init__()
        self.exploration_parameters = None
        self.num_outputs = num_outputs

    def set_exploration_parameters(self, exp_ps):
        if self.exploration_parameters is None:
            self.exploration_parameters = torch.zeros_like(exp_ps)

        self.exploration_parameters.copy_(exp_ps)

    def forward(self, x, deterministic=False):
        x = F.softmax(self.linear(x), 1)
        if deterministic:
            mixed_x = x
        else:
            mixed_x = (
                x * self.exploration_parameters +
                (1-self.exploration_parameters) * 1/self.num_outputs
            )
        return FixedCategorical(probs=mixed_x)


class MixGauss(DiagGaussian):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('MixGauss is not implemented yet.')

class MixPolicy(Policy):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()


        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = MixCategorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = MixGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            raise NotImplementedError(
                    'Multibinary not implemented for MixPolicy.')
        else:
            raise NotImplementedError

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, exp_ps):
       pass
