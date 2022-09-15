import numpy as np
import matplotlib.pyplot as plt
from classes.ActionSpace import ActionSpace
from classes.ExploreFirstGreedyAgent import ExploreFirstGreedyAgent
from classes.EpsilonGreedyAgent import EpsilonGreedyAgent
from classes.UCBAgent import UCBAgent
from classes.GradientBanditAgent import GradientBanditAgent

T = 1000
K = 10
trials = 1500

action_space = ActionSpace(T=T, K=K)

N_values = np.arange(100) + 1
greedy_regret = []
greedy_agent = ExploreFirstGreedyAgent(action_space=action_space, n_explore=5, K=K, T=T)

eps_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
epsilon_greedy_regret = []
epsilon_greedy_agent = EpsilonGreedyAgent(action_space=action_space, epsilon=0.1, K=K, T=T)

c_values = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
ucb_regret = []
ucb_agent = UCBAgent(action_space=action_space, c=0.1, K=K, T=T)

alpha_values = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
gradient_bandit_regret = []
gradient_bandit_agent = GradientBanditAgent(action_space=action_space, alpha=0.1, K=K, T=T)

# for N in N_values:
#     trial_regrets = []
#     for i in range(trials):
#         greedy_agent.reset(N)
#         greedy_agent.run()
#         trial_regrets.append(action_space.compute_regret(greedy_agent.actions))
#
#     greedy_regret.append(np.mean(trial_regrets))
#
# plt.figure()
# plt.plot(N_values, greedy_regret)
# plt.xlabel('N')
# plt.ylabel('regret')
# plt.savefig('regret_v_N.png')
# print(f"OPTIMAL VALUE FOR GREEDY: N = {N_values[np.argmin(greedy_regret)]}")

# for epsilon in eps_values:
#     trial_regrets = []
#     for i in range(trials):
#         epsilon_greedy_agent.reset(epsilon)
#         epsilon_greedy_agent.run()
#         trial_regrets.append(action_space.compute_regret(epsilon_greedy_agent.actions))
#
#     epsilon_greedy_regret.append(np.mean(trial_regrets))
#
# plt.figure()
# plt.plot(eps_values, epsilon_greedy_regret)
# plt.xlabel('\\epsilon')
# plt.ylabel('regret')
# plt.savefig('regret_v_epsilon.png')
# print(f"OPTIMAL VALUE FOR GREEDY EPSILON: eps = {eps_values[np.argmin(epsilon_greedy_regret)]}")

# for c in c_values:
#     trial_regrets = []
#     for i in range(trials):
#         ucb_agent.reset(c)
#         ucb_agent.run()
#         trial_regrets.append(action_space.compute_regret(ucb_agent.actions))
#
#     ucb_regret.append(np.mean(trial_regrets))
#
# plt.figure()
# plt.plot(c_values, ucb_regret)
# plt.xlabel('c')
# plt.ylabel('regret')
# plt.savefig('regret_v_c.png')
# print(f"OPTIMAL VALUE FOR UCB: c = {c_values[np.argmin(ucb_regret)]}")

for alpha in alpha_values:
    trial_regrets = []
    for i in range(trials):
        print(f"TRIAL: {i}")
        gradient_bandit_agent.reset(alpha)
        gradient_bandit_agent.run()
        trial_regrets.append(action_space.compute_regret(gradient_bandit_agent.actions))

    gradient_bandit_regret.append(np.mean(trial_regrets))

plt.figure()
plt.plot(alpha_values, gradient_bandit_regret)
plt.xlabel('\\alpha')
plt.ylabel('regret')
plt.savefig('regret_v_alpha.png')
print(f"OPTIMAL VALUE FOR GRADIENT BANDIT: \\alpha = {alpha_values[np.argmin(gradient_bandit_regret)]}")

