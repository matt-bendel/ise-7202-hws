import numpy as np

class GamblerEnvironment:
    def __init__(self, p_h):
        self.p_h = p_h

        self.policy = np.zeros(100)

        self.V = np.zeros(101)
        self.mc_V = np.zeros(101)

        self.rewards = np.zeros(101)
        self.rewards[100] = 1

        self.returns = [[] for i in range(101)]
        self.episodes = []

        self.theta = 1e-7

    def get_max_action(self, s, arg=False):
        actions = np.zeros(100)
        for a in range(np.minimum(s, 100 - s) + 1):
            actions[a] = self.p_h * (self.rewards[s + a] + self.V[s + a]) + (1 - self.p_h) * (self.rewards[s - a] + self.V[s - a])

        return np.max(actions) if not arg else np.argmax(np.round(actions[1:], decimals=5)) + 1 # To get account for precision error

    def policy_evaluation(self):
        delta = 1
        while delta > self.theta:
            delta = 0
            for s in range(100):
                v = self.V[s]
                self.V[s] = self.get_max_action(s)
                delta = np.maximum(delta, np.abs(v - self.V[s]))

    def value_iteration(self):
        self.policy_evaluation()

        for s in range(100):
            self.policy[s] = self.get_max_action(s, arg=True)

    def generate_episodes(self, n):
        for i in range(n):
            print(f'Generating Episode: {i+1}')
            episode = {
                'states': [],
                'actions': [],
                'rewards': []
            }

            terminal_state = False
            current_state = 50

            while not terminal_state:
                episode['states'].append(current_state)
                p = np.random.rand()
                a = self.policy[int(current_state)]
                episode['actions'].append(a)

                if p <= self.p_h:
                    current_state += a
                else:
                    current_state -= a

                terminal_state = True if current_state == 0 or current_state == 100 else False
                episode['rewards'].append(1 if current_state == 100 else 0)

            self.episodes.append(episode)

    def every_visit_monte_carlo(self):
        j = 1
        for episode in self.episodes:
            print(f'Evaluating Episode: {j}')
            j += 1
            G = 0
            for i in range(len(episode['states'])):
                t = len(episode['states']) - 1 - i
                G = G + episode['rewards'][t]
                self.returns[int(episode['states'][t])].append(G)
                self.mc_V[int(episode['states'][t])] = np.mean(self.returns[int(episode['states'][t])])
