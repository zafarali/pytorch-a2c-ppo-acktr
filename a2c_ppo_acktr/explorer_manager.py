"""
Manages the setting and updating of the exploration parameter.
"""

import numpy as np

class GaussianExplorer(object):
    def __init__(self, mu=0.9, upper_bound=0.8, lower_bound=0.1, spread=0.25, soft=0.95):
        self.mu = mu
        self.spread = spread
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.soft = soft

    def draw_exploration_coefficients(self, batch_size):
        return np.clip(
                np.random.normal(
                    loc=self.mu, scale=self.spread, size=(batch_size, 1)),
                self.lower_bound,
                self.upper_bound)

    def update_exploration_distribution(self, exp_ps, rewards):
        rewards_energy_dist = np.exp(rewards)
        rewards_energy_dist /= np.sum(np.exp(rewards))
        updated_value = np.sum(exp_ps * rewards_energy_dist)
        soft_update = self.soft * self.mu + (1-self.soft) * updated_value
        self.mu = np.clip(soft_update, self.lower_bound, self.upper_bound)

class DecayExplorer(object):
    def __init__(self, start_exploration=0.8, decay_rate=0.999995, lower_bound=0.1):
        self.start_exploration = start_exploration
        self.decay_rate = decay_rate
        self.current_exploration = start_exploration
        self.lower_bound = lower_bound

    def draw_exploration_coefficients(self, batch_size):
        return np.ones((batch_size, 1)) * self.current_exploration

    def update_exploration_distribution(self, exp_ps, rewards):
        self.current_exploration *= self.decay_rate
        self.current_exploration = np.clip(self.current_exploration, self.lower_bound, 1.0)

    @property
    def mu(self):
        return self.current_exploration
