import numpy as np

class ExploreFirstGreedyAgent:
    def __init__(self, action_space, n_explore, K, T):
        self.action_space = action_space
        self.n_explore = n_explore
        self.K = K
        self.T = T
        self.current_time = 0

        self.Q = [0 for i in range(self.K)] #sum of rewards
        self.N = [0 for i in range(self.K)]
        self.R = []
        self.actions = []

    def explore(self):
        for action in range(self.K):
            for t in range(self.n_explore):
                reward = self.action_space.sample(action)
                self.N[action] += 1
                self.Q[action] = self.Q[action] + (reward - self.Q[action]) / self.N[action]
                self.R.append(reward)
                self.actions.append(action)
                self.current_time += 1

    def exploit(self):
        action = np.argmax(self.Q)
        reward = self.action_space.sample(action)
        self.actions.append(action)
        self.R.append(reward)

    def run(self):
        self.explore()

        for i in range(self.T - self.current_time):
            self.exploit()

    def reset(self, new_n):
        self.n_explore = new_n
        self.current_time = 0
        self.Q = [0 for i in range(self.K)]  # sume of rewards
        self.N = [0 for i in range(self.K)]
        self.R = []
        self.actions = []
