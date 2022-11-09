import numpy as np

from environments.random_walk import RandomWalk

class MonteCarlo:
    def __init__(self, alpha, w_size, v_hat, v_hat_grad, n_states, feat_transform=None):
        self.alpha = alpha
        self.w_size = w_size
        self.environment = RandomWalk(n_states)
        self.v_hat = v_hat
        self.v_hat_grad = v_hat_grad
        self.feat_transform = feat_transform

    def run(self, n_episodes):
        print("Running MC Prediction")
        w = np.zeros(self.w_size)

        for i in range(n_episodes):
            print(f"At t = {i + 1}")
            episode = self.environment.generate_episode()
            for (G, s) in episode:
                w = w + self.alpha * (G - self.v_hat(s, w, self.feat_transform)) * self.v_hat_grad(s, w, self.feat_transform)

        return w