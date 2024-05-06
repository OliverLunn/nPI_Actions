####Script to plot classical action.

import numpy as np
import matplotlib.pyplot as plt

phi_values = np.arange(-5,5.1,0.1)  #x axis values

msq, msq_p, lamb = -2, 2, 4 #setting constants
action = np.zeros((len(phi_values))) #set up arrays for data storage
action_p = np.zeros((len(phi_values)))

for p in range(len(phi_values)):    #calcuate actions
    action[p] = (msq / 2) * phi_values[p]**2 + (lamb / 24) * phi_values[p]**4
    action_p[p] = (msq_p / 2) * phi_values[p]**2 + (lamb / 24) * phi_values[p]**4

fig, (ax1) = plt.subplots(1,1) #plotting
plt.xlabel("$\Phi$", fontsize=30)
plt.ylabel("$S(\Phi)$", fontsize=30)
plt.plot(phi_values, action, "b-", label="$m^{2}<0$")
plt.plot(phi_values, action_p, "k--", label="$m^{2}>0$")
plt.legend(fontsize=26, loc="upper right")
plt.tick_params(labelsize=28)
ax1.set_aspect(0.075)
plt.show()