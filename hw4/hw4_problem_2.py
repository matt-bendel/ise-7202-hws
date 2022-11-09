import matplotlib.pyplot as plt
import numpy as np

from environments.random_walk import RandomWalk
from runners.monte_carlo import MonteCarlo

def get_value_function(n):
    environment = RandomWalk(n)
    delta = 100
    theta = 1e-3

    V = [0 for i in range(n)]

    print("Running Policy Evaluation to estimate value function...")
    while delta > theta:
        delta = 0
        for s in range(n):
            v = V[s]
            new_V_s = 0
            for a in range(100):
                a = a - 100

                s_prime, reward = environment.get_next_state_and_reward(s, a)
                if s_prime > n - 1 or s_prime < 0:
                    next_v = 0
                else:
                    next_v = V[s_prime]
                new_V_s += 1 / 200 * (reward + next_v)

            for a in range(100):
                a = a + 1

                s_prime, reward = environment.get_next_state_and_reward(s, a)
                if s_prime > n - 1 or s_prime < 0:
                    next_v = 0
                else:
                    next_v = V[s_prime]
                new_V_s += 1 / 200 * (reward + next_v)

            V[s] = new_V_s
            delta = max(delta, np.abs(v - V[s]))

    return V

def v_hat(s, w, feature_transform):
    return np.dot(feature_transform(s), w)

def v_hat_grad(s, w, feature_transform):
    return feature_transform(s)

def state_agg(s, w_size=10, s_size=1000):
    x = np.zeros(w_size)
    x[s * w_size // s_size] = 1

    return x

def polynomial_basis(s, order_n=5):
    return np.array([((s + 1) / 1000) ** i for i in range(order_n + 1)])

def fourier_basis(s, order_n=5):
    return np.array([np.cos(i * np.pi * (s + 1) / 1000) for i in range(order_n + 1)])

n_states = 1000
states = np.linspace(1, n_states, n_states)
V_true = get_value_function(n_states)
print("Got value function!")

mc_state_agg = MonteCarlo(2e-5, 10, v_hat, v_hat_grad, n_states, feat_transform=state_agg)
print("State Aggregation...")
w_state_agg = mc_state_agg.run(5000)
print("Got MC Prediction!")
V_state_agg = [v_hat(i, w_state_agg, state_agg) for i in range(n_states)]

mc_polynomial = MonteCarlo(1e-4, 6, v_hat, v_hat_grad, n_states, feat_transform=polynomial_basis)
print("Polynomial Basis...")
w_polynomial = mc_polynomial.run(5000)
print("Got MC Prediction!")
V_polynomial = [v_hat(i, w_polynomial, polynomial_basis) for i in range(n_states)]

mc_fourier = MonteCarlo(1e-4, 6, v_hat, v_hat_grad, n_states, feat_transform=fourier_basis)
print("Fourier Basis...")
w_fourier = mc_fourier.run(5000)
print("Got MC Prediction!")
V_fourier = [v_hat(i, w_fourier, fourier_basis) for i in range(n_states)]

plt.figure()
plt.plot(states, V_true)
plt.plot(states, V_state_agg)
plt.xlabel('State')
plt.ylabel('Value')
plt.legend(['True Value Function', 'State Agg. V_Hat'])
plt.savefig('prob_2_state_agg.png')

plt.figure()
plt.plot(states, V_true)
plt.plot(states, V_polynomial)
plt.xlabel('State')
plt.ylabel('Value')
plt.legend(['True Value Function', 'Polynomial Basis V_Hat'])
plt.savefig('prob_2_polynomial.png')

plt.figure()
plt.plot(states, V_true)
plt.plot(states, V_fourier)
plt.xlabel('State')
plt.ylabel('Value')
plt.legend(['True Value Function', 'Fourier Basis V_Hat'])
plt.savefig('prob_2_fourier.png')