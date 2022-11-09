import numpy as np

from environments.cliff_world import CliffWorld

class Sarsa:
    def __init__(self, num_episodes, epsilon=0.1, alpha=1e-2):
        self.environment = CliffWorld(num_rows=4, num_cols=12)

        self.epsilon = epsilon
        self.alpha = alpha
        self.num_episodes = num_episodes
        self.episode_reward_sum = []
        self.rewards = []
        self.Q = np.zeros(shape=(48, 4))
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
            self.environment = CliffWorld(num_rows=4, num_cols=12)
            print(f"Episode: {i+1}")
            reward_sum = 0
            s = 0
            a = self.get_action(s)
            s_prime = None
            a_prime = None

            max_iter = 10000
            count = 0

            while self.environment.location != (11, 0) and count < max_iter:
                s_prime = self.environment.get_next_state(a)
                reward = self.get_reward(self.environment.location)
                reward_sum += reward

                a_prime = self.get_action(s_prime)
                self.Q[s, a] = self.Q[s, a] + self.alpha * (reward + self.Q[s_prime, a_prime] - self.Q[s, a])

                if reward == -100:
                    break

                s = s_prime
                a = a_prime
                count += 1

            self.episode_reward_sum.append(reward_sum)