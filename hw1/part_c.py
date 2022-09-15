from classes.EpsilonGreedyAgent import EpsilonGreedyAgent
from classes.UCBAgent import UCBAgent
from classes.ActionSpace import ActionSpace
import matplotlib.pyplot as plt
import numpy as np

T = 1000
K = 10
trials = 1500

action_space = ActionSpace(T=T, K=K)

epsilon = 0
c = 0

epsilon_greedy_agent = EpsilonGreedyAgent(action_space=action_space, epsilon=0.1, K=K, T=T)
epsilon_greedy_regret = [0 for t in range(T)]

ucb_agent = UCBAgent(action_space=action_space, c=0.1, K=K, T=T)
ucb_regret = [0 for t in range(T)]

for t in range(trials):
    epsilon_greedy_agent.reset(epsilon)
    ucb_agent.reset(c)

    epsilon_greedy_agent.run()
    ucb_agent.run()

    epsilon_greedy_regret = epsilon_greedy_agent.regret
    ucb_regret = ucb_agent.regret

x_axis = np.arange(1000) + 1
plt.figure()
plt.plot(x_axis, [val/trials for val in epsilon_greedy_regret])
plt.xlabel('t')
plt.ylabel('regret')
plt.savefig('regret_v_t_epsilon.png')

plt.figure()
plt.plot(x_axis, [val/trials for val in ucb_regret])
plt.xlabel('t')
plt.ylabel('regret')
plt.savefig('regret_v_t_c.png')
