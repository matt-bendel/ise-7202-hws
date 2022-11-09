import numpy as np

from environments.cliff_world import CliffWorld

class QLearning:
    def __init__(self, num_episodes, epsilon=0.1, alpha=1e-2):
        self.environment = CliffWorld(num_rows=4, num_cols=12)

        self.epsilon = epsilon
        self.alpha = alpha
        self.num_episodes = num_episodes
        self.episode_reward_sum = []
        self.rewards = []
        self.Q = np.zeros(shape=(48, 4)) # action 0 => up, 1 => right, 2 => down, 3 => left
        self.num_rows = 4
        self.num_cols = 12

    def get_action(self, s):
        p = np.random.random()

        if p < self.epsilon:
            a = np.random.choice(4)
        else:
            a = np.argmax(self.Q[s, :])

        return a

    def get_reward(self, location):
        if location[1] == 0 and location[0] != 0 and location[0] != 11:
            return -100

        return -1

    def run(self):
        for i in range(self.num_episodes):
            print(f"Episode: {i+1}")
            reward_sum = 0
            s = 0

            max_iter = 10000
            count = 0

            while self.environment.location != (11, 0) and count < max_iter:
                a = self.get_action(s)
                s_prime = self.environment.get_next_state(a)
                reward = self.get_reward(self.environment.location)
                reward_sum += reward

                self.Q[s, a] = self.Q[s, a] + self.alpha * (reward + np.max(self.Q[s_prime, :]) - self.Q[s, a])

                if reward == -100:
                    break

                count += 1

                s = s_prime

            self.episode_reward_sum.append(reward_sum)
            self.environment = CliffWorld(num_rows=4, num_cols=12)

