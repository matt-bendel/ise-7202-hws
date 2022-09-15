import numpy as np

class ActionSpace:
    def __init__(self, T, K):
        self.T = T
        self.K = K

        self.action_means = np.random.normal(size=self.K)
        self.mu_star = np.max(self.action_means)

    def sample(self, action):
        return np.random.normal(loc=self.action_means[action])

    def compute_regret(self, chosen_actions, t=1000):
        mus = [self.action_means[action] for action in chosen_actions]

        return t * self.mu_star - np.sum(mus)

    def reset(self):
        self.action_means = np.random.normal(size=self.K)
        self.mu_star = np.max(self.action_means)
