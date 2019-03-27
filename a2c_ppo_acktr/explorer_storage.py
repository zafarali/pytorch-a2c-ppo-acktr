"""
Modification of the RolloutStorage to also store information about
the exploration parameter used by each process/environment.
"""
import torch
from .storage import RolloutStorage

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

        self.exploration_parameters = torch.zeros(num_steps, num_processes, 1)

    def to(self, device):
        super().to(device)
        self.exploration_parameters = self.exploration_parameters.to(device)

    def insert(
            self,
            obs,
            recurrent_hidden_states,
            actions,
            action_log_probs,
            value_preds,
            rewards,
            masks,
            bad_masks,
            exploration_parameters):

        self.exploration_parameters[self.step].copy_(
                exploration_parameters)
        super().insert(obs, recurrent_hidden_states, actions,
                           action_log_probs, value_preds, rewards,
                           masks, bad_masks)

    def recurrent_generator(self, *args, **kwargs):
        raise NotImplementedError('Not implemented yet.')

    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps,
                      num_mini_batch))
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=False)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]
            exp_ps = self.exploration_parameters.view(-1, 1)[indices]
            yield (obs_batch, recurrent_hidden_states_batch,
                   actions_batch, value_preds_batch, return_batch,
                   masks_batch, old_action_log_probs_batch, adv_targ, exp_ps)

# TODO(zaf): At some point might want to update compute_returns to correct for
# value function prediction with a different alpha.
