import numpy as np
import matplotlib.pyplot as plt

from GamblerEnvironment import GamblerEnvironment

gambler_environment_25 = GamblerEnvironment(0.25)
gambler_environment_55 = GamblerEnvironment(0.55)

print("PERFORMING VALUE ITERATION")
gambler_environment_25.value_iteration()
gambler_environment_55.value_iteration()

plt.figure()
plt.plot(range(100), gambler_environment_25.V[0:100])
plt.savefig('val.png')

plt.figure()
plt.bar(range(1, 100), gambler_environment_25.policy[1:100])
plt.xlabel('Capital')
plt.ylabel('Optimal Policy')
plt.title('Optimal Policy vs. Capital for p_h=0.25')
plt.savefig('p_h_25_plot.png')

plt.figure()
plt.bar(range(1, 100), gambler_environment_55.policy[1:100])
plt.xlabel('Capital')
plt.ylabel('Optimal Policy')
plt.title('Optimal Policy vs. Capital for p_h=0.55')
plt.savefig('p_h_55_plot.png')
plt.close()

print("PERFORMING Every-Visit MC Prediction")
gambler_environment_55.generate_episodes(300)
gambler_environment_55.every_visit_monte_carlo()

plt.figure()
print(gambler_environment_55.mc_V[0:100])
plt.plot(range(100), gambler_environment_55.V[0:100])
plt.plot(range(100), gambler_environment_55.mc_V[0:100])
plt.savefig('mc_v_vi.png')
plt.close()