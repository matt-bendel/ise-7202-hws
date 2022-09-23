import numpy as np
import random

grid_model = {
    0: {
        'U': (1, -1, 1),  # Up
    },
    1: {
        'U': (1, -1, 2),  # Up
    },
    2: {
        'R': (1, -1, 4),  # Right
    },
    3: {
        'R': (1, -1, 5),  # Right
    },
    4: {
        'R': (1, -1, 7),  # Right
    },
    5: {
        'U': (1, -1, 6),  # Up
    },
    6: {
        'U': (1, -1, 7),  # Up
    },
    7: {
        'R': (1, -1, 10),  # Right
    },
    8: {
        'L': (1, -1, 5), # Left
    }
}

loop_in_order = False
random_order = False
delta = 100
theta = 1e-6

V = np.random.randn(11)
V[9] = -10
V[10] = 10

iteration = 0

while delta > theta:
    delta = 0
    state_list = list(grid_model.keys()) if loop_in_order else reversed(list(grid_model.keys()))

    if random_order:
        random.shuffle(state_list)

    for s in state_list:
        v = V[s]
        new_V_s = 0
        # p(s, a, s') = 0 for all s' except the one a leads to
        for a in grid_model[s]:
            new_V_s += grid_model[s][a][0] * 1 * (grid_model[s][a][1] + V[grid_model[s][a][2]])
            # V(s) = pi(a|s) * p(s, a, s') * (r(s,a) + V(s'))
        V[s] = new_V_s
        delta = max(delta, np.abs(v - V[s]))

    iteration += 1

print(f"Converged in {iteration} iterations")
if random_order:
    print(f"Evaluated states in random order")

else:
    print(f"Evaluated states in {'increasing' if loop_in_order else 'decreasing'} order")

for s in grid_model:
    print(f"V({s}) = {V[s]:.2f}")

print(f"V(9) = {V[9]:.2f}")
print(f"V(10) = {V[10]:.2f}")