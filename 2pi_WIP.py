import matplotlib.pyplot as plt
import numpy as np
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
    integrand = np.exp(-((m / 2) * (phi ** 2)) - ((1/6) * xhi * (phi**3)) - ((lamb/24) * (phi ** 4)) + (j * phi) +  (k * phi**2))
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
   
    msq, xhi, lamb = -1, 0, 2
    
    step, min_val, max_val = 0.15, -20, 20
    j_values = np.arange(min_val, max_val+step, step)                  
    k_values = np.arange(min_val, max_val+step, step)
    phi_values = np.arange(-1.5,1.5+step,step)
    delta_values = np.arange(0,3+step,step)

    z = np.zeros((len(j_values), len(k_values)))                                           
    g_jk = np.zeros((len(j_values), len(k_values)))     #array for \Gamma_{JK}
    gamma = np.zeros((len(phi_values), len(delta_values)))  #array for \Gamma[\phi, \Delta]

    #2PI Calculation
    z = integration(j_values, k_values, msq, xhi, lamb, z)
    w = -np.log(z)
    gamma = max_gamma(phi_values, delta_values, j_values, k_values, w, g_jk, gamma)

    ext_coord = np.unravel_index(gamma.argmin(), gamma.shape)
    X, Y = np.meshgrid(phi_values,delta_values)

    fig, (ax1) = plt.subplots(1,1)     #figure for plots
    im1 = ax1.pcolormesh(phi_values, delta_values, gamma)
    ax1.contour(X,Y,gamma, colors=['black'])
    ax1.set_xlabel("$\phi$")
    ax1.set_ylabel("$\Delta'$")
    ax1.set_title("$\Gamma[\phi,\Delta]$")
    ax1.plot(phi_values[ext_coord[1]], delta_values[ext_coord[0]], ".w")
    plt.colorbar(im1)
    plt.tight_layout
    plt.show()