import numpy as np
import matplotlib.pyplot as plt

phi_values = np.arange(-4,4.1,0.1)

msq, msq_p, lamb = -2, 2, 4
action = np.zeros((len(phi_values)))
action_p = np.zeros((len(phi_values)))

for p in range(len(phi_values)):
    action[p] = (msq / 2) * phi_values[p]**2 + (lamb / 24) * phi_values[p]**4
    action_p[p] = (msq_p / 2) * phi_values[p]**2 + (lamb / 24) * phi_values[p]**4

fig, (ax1) = plt.subplots(1,1) 
plt.xlabel("$\phi$", fontsize=28)
plt.ylabel("$S(\phi)$", fontsize=28)
plt.plot(phi_values, action, "b-", label="$m^{2}<0$")
plt.plot(phi_values, action_p, "k--", label="$m^{2}>0$")
plt.legend(fontsize=22)
plt.tick_params(labelsize=26)
plt.show()