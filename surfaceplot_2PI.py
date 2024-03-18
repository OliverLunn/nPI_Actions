import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib import colormaps
'''
We work in terms of rescaled coordinates:

 phi_prime = phi
 delta_prime = phi^2 + delta
 J_prime = J
 K_prime = K / 2

'''

def integrand(phi, m, xhi, lamb, j, k):
    integrand = np.exp(-(m / 2) * (phi ** 2) - (1/6) * xhi * (phi**3)- (lamb/24) * (phi ** 4) + (j * phi) + (k * phi**2))
    return integrand

def integration(j_values, k_values, m, xhi, lamb, z):
    for j in range(len(j_values)):
        for k in range(len(k_values)):
            z_intermediate = integrate.quad(integrand, -np.inf, np.inf, args=(m, xhi, lamb, j_values[j], k_values[k]))
            z[k,j] = z_intermediate[0]
    return z

def gamma_jk(phi, j_values, k_values, w, delta, array):
    """
    Calculates the Legendre-Fenchel transformation of W[J]. Returns array of \Gamma_j[phi] values.
    """
    for j in range(len(j_values)):
            for k in range(len(k_values)):
                array[k,j] = w[k,j] + j_values[j] * phi + k_values[k] * (delta)
    return array

def max_gamma(phi_values, delta_values, j_values, k_values, w, g_jk, gamma):
    """
    Extremises \Gamma_{JK} to find \Gamma{\phi,\Delta}
    """
    for p in range(len(phi_values)):
         for d in range(len(delta_values)):
            g = gamma_jk(phi_values[p], j_values, k_values, w, delta_values[d], g_jk)       
            gamma[d,p] = np.max(g)
    return gamma

if __name__ == '__main__':
   
    m, xhi, lamb = -2, 0, 6
    
    step, min_val, max_val = 0.1, -10, 10
    j_values = np.arange(min_val, max_val+step, step)                  
    k_values = np.arange(min_val, max_val+step, step)
    phi_values = np.arange(-2,2+step,step)
    delta_values = np.arange(-2,2+step,step)

    z = np.zeros((len(j_values), len(k_values)))                                           
    g_jk = np.zeros((len(j_values), len(k_values)))     #array for \Gamma_{JK}
    gamma = np.zeros((len(phi_values), len(delta_values)))  #array for \Gamma[\phi, \Delta]

    z = integration(j_values, k_values, m, xhi, lamb, z)
    w = -np.log(z)
    gamma = max_gamma(phi_values, delta_values, j_values, k_values, w, g_jk, gamma)
    ext_coord = np.unravel_index(gamma.argmin(), gamma.shape)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    X,Y = np.meshgrid(phi_values, delta_values)
    im1=ax1.plot_surface(X, Y, gamma, cmap="viridis")
    ax1.set_xlabel("$\phi$")
    ax1.set_ylabel("$\Delta$")
    ax1.set_title("$\Gamma[\phi,\Delta]$    (m=-2, $\lambda=6$)")

    plt.tight_layout
    plt.show()




