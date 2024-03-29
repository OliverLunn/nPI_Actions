import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from matplotlib import colormaps
from tqdm import tqdm
'''
We work in terms of rescaled coordinates:

 phi_prime = phi
 delta_prime = phi^2 + delta
 J_prime = J
 K_prime = K / 2

'''
def integrand(phi, m, xhi, lamb, j, k):
    integrand = np.exp(-((m / 2) * (phi ** 2)) - ((1/6) * xhi * (phi**3)) - ((lamb/24) * (phi ** 4)) + (j * phi) +  (1/2 * k * phi**2))
    return integrand

def integration(j_values, k_values, m, xhi, lamb, z):
    for j in tqdm(range(len(j_values))):
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
                array[k,j] = w[k,j] + j_values[j] * phi + 1/2 * k_values[k] * (phi**2+delta)
    return array


def maximise(g):
    '''
    Finds and maximises G_{JK} (if max is at boundary returns NaN)
    '''
    max_coord = np.unravel_index(np.nanargmax(g), g.shape)
    g_padded = np.pad(g, 1, mode='constant')

    if np.any(g_padded[:,max_coord[1]+2]) == False or np.any(g_padded[:,max_coord[1]-1]) == False:
        max_gamma = np.nan
    elif np.any(g_padded[max_coord[0]+1,:]) == False or np.any(g_padded[max_coord[0]-2,:]) == False:
        max_gamma = np.nan
    else:
        max_gamma = np.nanmax(g)
    return max_gamma

def max_gamma(phi_values, delta_values, j_values, k_values, w, g_jk, gamma):
    """
    Extremises \Gamma_{JK} to find \Gamma{\phi,\Delta}
    """
    for p in tqdm(range(len(phi_values))):
         for d in range(len(delta_values)):
            g = gamma_jk(phi_values[p], j_values, k_values, w, delta_values[d], g_jk)
            gamma[d,p] = maximise(g)
    return gamma


if __name__ == '__main__':
   
    msq, xhi, lamb = -2, 0, 6
    
    step, min_val, max_val = 0.5, -30, 30
    step_pd = 0.05
    j_values = np.arange(min_val, max_val+step, step)                  
    k_values = np.arange(min_val, max_val+step, step)
    k_values_1pi = np.zeros(len(k_values))
    phi_values = np.arange(-2,2+step_pd, step_pd)
    delta_values = np.arange(0.1,4.1+step_pd, step_pd)

    z = np.zeros((len(j_values), len(k_values)))                                           
    g_jk = np.zeros((len(j_values), len(k_values)))     #array for \Gamma_{JK}
    gamma = np.zeros((len(phi_values), len(delta_values)))  #array for \Gamma[\phi, \Delta]
    g_1pi = np.zeros((len(j_values), len(k_values)))
    gamma_1pi = np.zeros((len(phi_values), len(delta_values)))

    #2PI Calculation
    z = integration(j_values, k_values, msq, xhi, lamb, z)
    print("Z calculated")
    w = -np.log(z)
    gamma = max_gamma(phi_values, delta_values, j_values, k_values, w, g_jk, gamma)

    #1PI Line
    z_1pi = integration(j_values, k_values_1pi, msq, xhi, lamb, z)
    w = -np.log(z)
    gamma_1PI = max_gamma(phi_values, delta_values, j_values, k_values_1pi, w, g_1pi, gamma_1pi)

    X, Y = np.meshgrid(phi_values,delta_values)
    
    fig, (ax1) = plt.subplots(1,1)     #figure for plots
    im1 = ax1.pcolormesh(phi_values, delta_values, gamma)
    ax1.contour(X,Y,gamma, colors=['black'])
    ax1.plot(phi_values, abs(gamma_1PI[0,:]), ".k")
    ax1.set_xlabel("$\phi$")
    ax1.set_ylabel("$\Delta'$")
    ax1.set_title("$\Gamma[\phi,\Delta]$")
    plt.colorbar(im1)
    plt.tight_layout
    plt.show()