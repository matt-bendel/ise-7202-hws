import numpy as np

class EpsilonGreedyAgent:
    def __init__(self, action_space, epsilon, K, T):
        self.action_space = action_space
        self.epsilon = epsilon
        self.K = K
        self.T = T

        self.Q = [0 for i in range(self.K)] #sum of rewards
        self.N = [0 for i in range(self.K)]
        self.R = []
        self.actions = []
        self.regret = []

    def explore(self, t=0):
        action = np.random.randint(10)
        reward = self.action_space.sample(action)
        self.actions.append(action)
        self.R.append(reward)
        self.N[action] += 1
        self.Q[action] = self.Q[action] + (reward - self.Q[action]) / self.N[action]
        self.regret.append(self.action_space.compute_regret(self.actions, t=t))

    def exploit(self, t=0):
        action = np.argmax(self.Q)
        reward = self.action_space.sample(action)
        self.actions.append(action)
        self.R.append(reward)
        self.N[action] += 1
        self.Q[action] = self.Q[action] + (reward - self.Q[action]) / self.N[action]
        self.regret.append(self.action_space.compute_regret(self.actions, t=t))

    def run(self):
        for i in range(self.T):
            p = np.random.rand()
            if (p < self.epsilon):
                self.explore(t=i+1)
            else:
                self.exploit(t=i+1)

    def reset(self, new_epsilon):
        self.epsilon = new_epsilon
        self.Q = [0 for i in range(self.K)]  # sume of rewards
        self.N = [0 for i in range(self.K)]
        self.R = []
        self.actions = []
        self.regret = []
