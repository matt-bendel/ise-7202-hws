import numpy as np

class RandomWalk:
    def __init__(self, n):
        self.n_states = n - 1

    def get_next_state_and_reward(self, s, a):
        s_prime = s + a

        if s_prime > self.n_states:
            return s_prime, 1

        if s_prime < 0:
            return s_prime, -1

        return s_prime, 0

    def generate_episode(self):
        terminal = False
        states = []
        actions = []
        rewards = []

        s = 499
        while not terminal:
            a = np.random.randint(0, 200) - 100
            if a >= 0:
                a += 1

            actions.append(a)

            s_prime, r = self.get_next_state_and_reward(s, a)

            if r != 0:
                terminal = True
                s_prime = 0 if r == -1 else 999

            rewards.append(r)
            states.append(s_prime)

            s = s_prime

        G = 0
        episode = []
        for i in range(len(rewards)):
            t = len(rewards) - i - 1
            G = rewards[t] + G
            episode.append((G, states[t]))

        episode.reverse()
        return episode