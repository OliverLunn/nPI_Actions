import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from matplotlib import colormaps
from mpl_toolkits.mplot3d import Axes3D

def integrand(phi, m, xhi, lamb, j, k):
    integrand = np.exp(-(m / 2) * (phi ** 2) - (1/3) * xhi * (phi**3)- (lamb/24) * (phi ** 4) + (j * phi) + (k * phi**2))
    return integrand

def integration(j_values, k_values, m, xhi, lamb, z):
    for j in range(len(j_values)):
        for k in range(len(k_values)):
            z_intermediate = integrate.quad(integrand, -np.inf, np.inf, args=(m, xhi, lamb, j_values[j], k_values[k]))
            z[k,j] = z_intermediate[0]
    return z

def gamma_j(phi, j_values, k_values, w, delta, array):
    """
    Calculates the Legendre-Fenchel transformation of W[J]. Returns array of \Gamma_j[phi] values.
    """
    for j in range(len(j_values)):
            for k in range(len(k_values)):
                array[k,j] = w[k,j] + j_values[j] * phi + k_values[k] * (delta)
    return array

if __name__ == '__main__':
   
    m, xhi, lamb = -2, 0, 6
    phi, delta = 2,0
    step, min_val, max_val = 0.5, -10, 10
    j_values = np.arange(min_val, max_val+step, step)                  
    k_values = np.arange(min_val, max_val+step, step)
    
    z = np.zeros((len(j_values), len(k_values)))                                           
    g_j = np.zeros((len(j_values), len(k_values)))
    
    z = integration(j_values, k_values, m, xhi, lamb, z)
    w = -np.log(z)
    g = gamma_j(phi, j_values, k_values, w, delta, g_j)       #calling g_j  funct to generate g_j data
    max_coord = np.unravel_index(np.nanargmax(g), g.shape)
   
    #fig, (ax3) = plt.subplots(1,1)             #figures for plots
    fig1, (ax1) = plt.subplots(1,1)

    #im = ax3.pcolormesh(j_values, k_values, w, cmap='viridis')
    #ax3.set_xlabel("J")
    #ax3.set_ylabel("K")
    #ax3.set_title("$W[J,K]$")

    im1 = ax1.pcolormesh(j_values, k_values, g, cmap='viridis')
    #X, Y = np.meshgrid(j_values, k_values)
    #im1=ax1.plot_surface(X, Y, g, cmap="viridis")
    ax1.set_xlabel("J")
    ax1.set_ylabel("K")
    ax1.set_title("$\Gamma_{J,K}[$"+str(phi)+","+str(delta)+"$]$")
    ax1.plot(j_values[max_coord[1]], k_values[max_coord[0]], ".k")
    #plt.colorbar(im1)
    plt.tight_layout
    plt.show()