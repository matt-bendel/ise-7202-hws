import numpy as np

class GradientBanditAgent:
    def __init__(self, action_space, alpha, K, T):
        self.action_space = action_space
        self.alpha = alpha
        self.K = K
        self.T = T

        self.R = []
        self.H = [0 for i in range(self.K)]
        self.pi = [1 / self.K for i in range(self.K)]
        self.actions = []

    def run(self):
        for i in range(self.T):
            self.update_policy()
            action = np.argmax(self.pi)
            reward = self.action_space.sample(action)
            self.actions.append(action)
            self.R.append(reward)
            self.update_preferences(action, reward, first=True if i == 0 else False)

    def update_preferences(self, chosen_action, reward, first=False):
        mean = 0 if first else np.mean(self.R)
        self.H[chosen_action] = self.H[chosen_action] + self.alpha * (reward - mean) * (1 - self.pi[chosen_action])
        for action in range(self.K):
            if action == chosen_action:
                continue

            self.H[action] = self.H[action] - self.alpha * (reward - mean) * self.pi[action]

    def update_policy(self):
        self.pi = [np.exp(preference - np.max(self.H)) for preference in self.H]
        self.pi = self.pi / np.sum(self.pi)

    def reset(self, new_alpha):
        self.alpha = new_alpha
        self.R = []
        self.H = [0 for i in range(self.K)]
        self.pi = [1 / self.K for i in range(self.K)]
        self.actions = []
