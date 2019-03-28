"""
Manages the setting and updating of the exploration parameter.
"""

import numpy as np

class GaussianExplorer(object):
    def __init__(self, mu=0.5, upper_bound=0.8, lower_bound=0.1, spread=0.5):
        self.mu = mu
        self.spread = spread
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def draw_exploration_coefficients(self, batch_size):
        return np.clip(
                np.random.normal(
                    loc=self.mu, scale=self.spread, size=(batch_size, 1)),
                self.lower_bound,
                self.upper_bound)

    def update_exploration_distribution(self, exp_ps, rewards):
        rewards_energy_dist = np.exp(rewards)
        rewards_energy_dist /= np.sum(np.exp(rewards))
        self.mu = np.clip(
                np.sum(exp_ps * rewards_energy_dist),
                self.lower_bound, self.upper_bound)



