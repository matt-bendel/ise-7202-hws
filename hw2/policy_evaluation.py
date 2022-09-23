import numpy as np
import random

grid_model = {
    0: {
        'L': (0.25, -1, 0), # Left
        'D': (0.25, -1, 0), # Down
        'R': (0.25, -1, 3),  # Right
        'U': (0.25, -1, 1),  # Up
    },
    1: {
        'L': (0.25, -1, 1), # Left
        'D': (0.25, -1, 0), # Down
        'R': (0.25, -1, 1),  # Right
        'U': (0.25, -1, 2),  # Up
    },
    2: {
        'L': (0.25, -1, 2), # Left
        'D': (0.25, -1, 1), # Down
        'R': (0.25, -1, 4),  # Right
        'U': (0.25, -1, 2),  # Up
    },
    3: {
        'L': (0.25, -1, 0), # Left
        'D': (0.25, -1, 3), # Down
        'R': (0.25, -1, 5),  # Right
        'U': (0.25, -1, 3),  # Up
    },
    4: {
        'L': (0.25, -1, 2), # Left
        'D': (0.25, -1, 4), # Down
        'R': (0.25, -1, 7),  # Right
        'U': (0.25, -1, 4),  # Up
    },
    5: {
        'L': (0.25, -1, 3), # Left
        'D': (0.25, -1, 5), # Down
        'R': (0.25, -1, 8),  # Right
        'U': (0.25, -1, 6),  # Up
    },
    6: {
        'L': (0.25, -1, 6), # Left
        'D': (0.25, -1, 5), # Down
        'R': (0.25, -1, 9),  # Right
        'U': (0.25, -1, 7),  # Up
    },
    7: {
        'L': (0.25, -1, 4), # Left
        'D': (0.25, -1, 6), # Down
        'R': (0.25, -1, 10),  # Right
        'U': (0.25, -1, 7),  # Up
    },
    8: {
        'L': (0.25, -1, 5), # Left
        'D': (0.25, -1, 8), # Down
        'R': (0.25, -1, 8),  # Right
        'U': (0.25, -1, 9),  # Up
    }
}

loop_in_order = True
random_order = False
delta = 100
theta = 1e-3

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