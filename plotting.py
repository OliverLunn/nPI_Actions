import matplotlib.pyplot as plt
import numpy as np

gamma = np.loadtxt("data17.txt")
pd_values = np.loadtxt("pd_values17.txt")

gamma = gamma - np.nanmin(gamma)
phi_vals = pd_values[0,:]
delta_vals = pd_values[1,:]
X, Y = np.meshgrid(phi_vals,delta_vals)

fig, (ax1) = plt.subplots(1,1)     #figure for plots
im1 = ax1.pcolormesh(phi_vals, delta_vals, gamma)
ax1.contour(X, Y, gamma, levels=[0.15, 0.5, 1, 2, 3],  colors=['black'])
ax1.set_xlabel("$\phi^{\prime}$", fontsize="28")
ax1.set_ylabel("$\Delta^{\prime}$", fontsize="28")
ax1.tick_params(labelsize=26)
cbar = plt.colorbar(im1)
ticklabs = cbar.ax.get_yticklabels()
cbar.ax.set_yticklabels(ticklabs, fontsize=24)
ax1.set_aspect('equal')
plt.show()
