import numpy as np

class UCBAgent:
    def __init__(self, action_space, c, K, T):
        self.action_space = action_space
        self.c = c
        self.K = K
        self.T = T

        self.Q = [0 for i in range(self.K)] #sum of rewards
        self.N = [0 for i in range(self.K)]
        self.R = []
        self.UCB = [0 for i in range(self.K)]
        self.actions = []
        self.regret = []

    def run(self):
        for i in range(self.K):
            action = i
            reward = self.action_space.sample(action)
            self.actions.append(action)
            self.R.append(reward)
            self.N[action] += 1
            self.Q[action] = self.Q[action] + (reward - self.Q[action]) / self.N[action]
            self.UCB[action] = self.Q[action] + self.c * np.sqrt(np.log(i+1) / self.N[action])
            self.regret.append(self.action_space.compute_regret(self.actions, t=i+1))

        for i in range(self.T - self.K):
            t = i + self.K + 1
            action = np.argmax(self.UCB)
            reward = self.action_space.sample(action)
            self.actions.append(action)
            self.R.append(reward)
            self.UCB[action] = self.Q[action] + self.c * np.sqrt(np.log(t) / self.N[action])
            self.N[action] += 1
            self.Q[action] = self.Q[action] + (reward - self.Q[action]) / self.N[action]
            self.regret.append(self.action_space.compute_regret(self.actions, t=t))

    def reset(self, new_c):
        self.c = new_c
        self.Q = [0 for i in range(self.K)]  # sume of rewards
        self.N = [0 for i in range(self.K)]
        self.R = []
        self.UCB = [0 for i in range(self.K)]
        self.actions = []
        self.regret = []
