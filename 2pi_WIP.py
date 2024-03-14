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
    integrand = np.exp(-(m / 2) * (phi ** 2) - (1/3) * xhi * (phi**3)- (lamb/24) * (phi ** 4) + (j * phi) + (1/2 * k * phi**2))
    return integrand

def integrand_1pi(phi, m, xhi, lamb, j):
    integrand = np.exp(-(m / 2) * (phi ** 2) - (1/3) * xhi * (phi**3)- (lamb/24) * (phi ** 4) + (j * phi))
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
                array[k,j] = w[k,j] + j_values[j] * phi + 1/2 * k_values[k] * (phi**2 + delta)
    return array

def gamma_1PI(phi_values, j_values, w, array):
    """
    Calculates the Legendre-Fenchel transformation of W[J]. Returns array of \Gamma_j[phi] values.(1PI)
    """
    for i in range(len(phi_values)):
        for j in range(len(j_values)):
            array[j,i] = w[j] - j_values[j] * phi_values[i]
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
   
    msq, xhi, lamb = -2, 0, 6
    
    step, min_val, max_val = 1, -10, 10
    j_values = np.arange(min_val, max_val+step, step)                  
    k_values = np.arange(min_val, max_val+step, step)
    phi_values = np.arange(-2,2+step,step)
    delta_values = np.arange(0,4+step,step)

    z = np.zeros((len(j_values), len(k_values)))                                           
    g_jk = np.zeros((len(j_values), len(k_values)))     #array for \Gamma_{JK}
    gamma = np.zeros((len(phi_values), len(delta_values)))  #array for \Gamma[\phi, \Delta]

    #2PI Calculation
    z = integration(j_values, k_values, msq, xhi, lamb, z)
    w = -np.log(z)
    gamma = max_gamma(phi_values, delta_values, j_values, k_values, w, g_jk, gamma)
    ext_coord = np.unravel_index(gamma.argmin(), gamma.shape)
    print("2PI CALCULATED")
    
    #1PI action (K=0 case)
    zk0 = []                                            #empty z values
    max_gamma1pi = []                                      #empty 1pi action values
    gk0 = np.zeros((len(j_values), len(phi_values)))    #array of gamma_j values to be used

    for j in j_values:
        z1pi = integrate.quad(integrand_1pi, -np.inf, np.inf, args=(msq,xhi,lamb,j))
        zk0.append(z1pi[0])


    w1pi = -np.log(zk0)
    g1pi = gamma_1PI(phi_values, j_values, w1pi, gk0)       #calling g_j  funct to generate g_j data
    max_1pi = np.max(g1pi, axis=0)                           #find max of g_j
    max_gamma1pi = np.append(max_gamma1pi, max_1pi, axis=0)       #store
    print("1PI CALCULATED")


    fig1, (ax1) = plt.subplots(1,1)     #figures for plots
    im1 = ax1.imshow(gamma, cmap='viridis')
    ax1.plot(max_gamma1pi, phi_values, "-k")
    ax1.set_xticks(np.arange(0, len(gamma), step*16), phi_values[::4])
    ax1.set_yticks(np.arange(0, len(gamma), step*16), delta_values[::4])
    ax1.set_xlabel("$\phi$")
    ax1.set_ylabel("$\Delta$")
    ax1.set_title("$\Gamma[\phi,\Delta]$")
    ax1.plot(ext_coord[1],ext_coord[0], ".w")
    ax1.invert_yaxis()
    plt.colorbar(im1)
    plt.tight_layout
    plt.show()
