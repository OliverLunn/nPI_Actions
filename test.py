####Plots the 2PI effective action calculated analytically (1 loop order)

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
   
    msq, xhi, lamb = 1, 0, 4 #set constants
    step_pd = 0.1
    phi_values = np.arange(-2,2+step_pd, step_pd)   #arrays of one- and two- point functions
    delta_values = np.arange(0,4+step_pd, step_pd)
    gamma = np.zeros((len(phi_values), len(delta_values)))  #empty array for 2PI
    
    for p in range(len(phi_values)):
         for d in range(len(delta_values)):
            action = (msq / 2) * phi_values[p]**2 + (lamb / 24) * phi_values[p]**4 #Classical action
            prop = msq + (lamb * phi_values[p]**2)/2    #propagator (2nd derivative of S(phi))
            gamma[d,p] = action + (1/2) * np.log(1/delta_values[d]) + (1/2) * prop * delta_values[d] - 1/2 #calculate 2PI action


    X, Y = np.meshgrid(phi_values, delta_values) #plotting
    fig, (ax1) = plt.subplots(1,1)     #figure for plots
    im1 = ax1.pcolormesh(phi_values, delta_values, gamma)
    ax1.contour(X,Y,gamma, colors=['black'])
    ax1.set_xlabel("$\phi$", fontsize="28")
    ax1.set_ylabel("$\Delta$", fontsize="28")
    ax1.tick_params(labelsize=26)
    cbar = plt.colorbar(im1)
    ticklabs = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(ticklabs, fontsize=24)
    ax1.set_aspect('equal')
    plt.tight_layout
    plt.show()