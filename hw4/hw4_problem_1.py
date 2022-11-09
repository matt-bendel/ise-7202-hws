import numpy as np
import matplotlib.pyplot as plt

from runners.sarsa import Sarsa
from runners.q_learning import QLearning


def plot_policy(environment, Q, name):
    arrows = {
        0: (0, 1),
        1: (-1, 0),
        2: (0, -1),
        3: (1, 0)
    }
    scale = 0.25

    fig, ax = plt.subplots(figsize=(12, 4))

    for i in range(environment.num_cols):
        for j in range(environment.num_rows):
            s = environment.state_nums[(i, j)]
            a = np.argmax(Q[s, :])
            plt.arrow(i, j, scale*arrows[a][0], scale*arrows[a][1], head_width=0.1)

    plt.savefig(f'{name}_policy.png')
    plt.close(fig)

num_eps = 500
num_runs = 250
reward_sum_sarsa = [0 for i in range(num_eps)]
reward_sum_Q = [0 for i in range(num_eps)]

first_reward_sarsa = []
first_reward_Q = []

print("RUNNING SARSA")
for i in range(num_runs):
    sarsa = Sarsa(num_eps, alpha=0.5)
    sarsa.run()
    if i == 0:
        first_reward_sarsa = sarsa.episode_reward_sum
        plot_policy(sarsa.environment, sarsa.Q, "sarsa")

    reward_sum_sarsa = [x + y for x,y in zip(reward_sum_sarsa, sarsa.episode_reward_sum)]

for i in range(num_runs):
    q_learner = QLearning(num_eps, alpha=0.5)
    q_learner.run()
    if i == 0:
        first_reward_Q = q_learner.episode_reward_sum
        plot_policy(q_learner.environment, q_learner.Q, "q_learning")

    reward_sum_Q = [x + y for x,y in zip(reward_sum_Q, q_learner.episode_reward_sum)]
    optimal_policy_map = np.argmax(q_learner.Q, axis=1).reshape((4, 12))

episodes = np.linspace(1, num_eps, num_eps, endpoint=True)
reward_sum_sarsa = [reward_sum_sarsa[i] / num_runs for i in range(num_eps)]
reward_sum_Q = [reward_sum_Q[i] / num_runs for i in range(num_eps)]

plt.figure()
plt.plot(episodes, first_reward_sarsa, 'b', label='Sarsa')
plt.plot(episodes, first_reward_Q, 'r', label='Q-Learning')
plt.title('Sum of Rewards during episodes vs. # of episodes for 1 run')
plt.xlabel('Episode')
plt.ylabel('Sum of Rewards during episode')
plt.legend(['Sarsa', 'Q-Learning'])
# plt.ylim([-100, 0])
plt.savefig('sarsa_and_q_1_result.png')
plt.close()

plt.figure()
plt.plot(episodes, reward_sum_sarsa, 'b', label='Sarsa')
plt.plot(episodes, reward_sum_Q, 'r', label='Q-Learning')
plt.title(f'Sum of Rewards during episodes vs. # of episodes averaged over {num_runs} runs')
plt.xlabel('Episode')
plt.ylabel('Sum of Rewards during episode')
plt.legend(['Sarsa', 'Q-Learning'])
# plt.ylim([-100, 0])
plt.savefig('sarsa_and_q_avg_results.png')
plt.close()
