import matplotlib.pyplot as plt
import numpy as np

gamma = np.loadtxt("data1.txt")
pd_values = np.loadtxt("pd_values1.txt")

phi_vals = pd_values[0,:]
delta_vals = pd_values[1,:]
X, Y = np.meshgrid(phi_vals,delta_vals)

fig, (ax1) = plt.subplots(1,1)     #figure for plots
im1 = ax1.pcolormesh(phi_vals, delta_vals, gamma)
ax1.contour(X,Y,gamma, colors=['black'])
ax1.set_xlabel("$\phi$", fontsize="28")
ax1.set_ylabel("$\Delta$", fontsize="28")
ax1.set_title("$\Gamma[\phi,\Delta]$", fontsize="28")
ax1.set_aspect('equal')

plt.tight_layout
plt.colorbar(im1)
plt.show()