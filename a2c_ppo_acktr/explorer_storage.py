"""
Modification of the RolloutStorage to also store information about
the exploration parameter used by each process/environment.
"""
import torch
from a2c_ppo_acktr import RolloutStorage

class RolloutStorageWithExploration(RolloutStorage):
    def __init__(
            self,
            num_steps,
            num_processes,
            obs_shape,
            action_space,
            recurrent_hidden_state_size):

        super(RolloutStorageWithExploration, self).__init__(
                num_steps,
                num_processes,
                obs_shape,
                action_space,
                recurrent_hidden_state_size)

        self.exploration_parameters = torch.zeros(num_prcesses, 1)

    def to(self, device):
        super(self).to(device)
        self.exploration_parameters = self.exploration_parameters.to(device)

    def insert(
            self,
            obs,
            recurrent_hidden_states,
            actions,
            action_log_probs,
            value_preds,
            rewards,
            masks
            bad_masks,
            exploration_parameters):
        super(self).insert(obs, recurrent_hidden_states, actions,
                           action_log_probs, value_preds, rewards,
                           masks, bad_masks)

        self.exploration_parameters.copy_(exploration_parameters)

# TODO(zaf): At some point might want to update compute_returns to correct for
# value function prediction with a different alpha.
