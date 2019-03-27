import torch
import torch.nn as nn

from a2c_ppo_acktr.algo import A2C_ACKTR

class A2C_explorer(A2C_ACKTR):
    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        exp_ps = rollouts.exploration_parameters

        (values,
         action_log_probs,
         dist_entropy, _,
         correction_ratios) = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape),
            exp_ps.view(-1, 1))

        values = values.view(num_steps, num_processes, 1)
        correction_ratios = correction_ratios.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        advantages = rollouts.returns[:-1] - values
        value_loss = (correction_ratios*(advantages.pow(2))).mean()

        action_loss = -(correction_ratios * advantages.detach() * action_log_probs).mean()
        # Apply the corrections for each part of the loss.
        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss ).backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()

