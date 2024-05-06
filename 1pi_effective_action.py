import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import scipy.constants as con

def classical_action(phi_values, m, xhi, lamb, array):
    """
    Calculates an array of values corresponding to the classical action over a specified range
    Inputs:
        phi_values [numpy.array] : range of phi values
        m [float] : mass term
        xhi [float] : cubic term
        lamb [float] : quartic term
        array [numpy.array] : array to store values
    Outputs:
        array [numpy.array] : array of values for S[phi]
    
    """
    for phi in phi_values:
        s = 0.5 * (m) * (phi ** 2) + (xhi / (3)) * (phi**3) + (lamb / 24) * (phi ** 4)
        array.append(s)
    return array


def integrand(phi, m, xhi, j):
    integrand = np.exp(-(m / 2) * (phi ** 2) - (1/3) * xhi * (phi**3)- (lamb/24) * (phi ** 4) + (j * phi))
    return integrand

def gamma_j(phi_values, j_values, w, array):
    """
    Calculates the Legendre-Fenchel transformation of W[J]. Returns array of \Gamma_j[phi] values.
    """
    for i in range(len(phi_values)):
        for j in range(len(j_values)):
            array[j,i] = w[j] + j_values[j] * phi_values[i]
    return array

if __name__ == '__main__':
   
    m, xhi, lamb = -2, 0, 6
    j_values = np.arange(-50, 50, 0.01)                  #values for j
    phi_values = np.arange(-3, 3, 0.01)                  #values for mean-field phi

    z_j = []                                            #empty z values
    max_gamma = []                                      #empty 1pi action values
    g_j = np.zeros((len(j_values), len(phi_values)))    #array of gamma_j values to be used

    for j in j_values:
        z = integrate.quad(integrand, -np.inf, np.inf, args=(m,xhi,j))
        z_j.append(z[0])

    w = -np.log(z_j)
    g = gamma_j(phi_values, j_values, w, g_j)       #calling g_j  funct to generate g_j data
    max = np.max(g, axis=0)                           #find max of g_j
    max_gamma = np.append(max_gamma, max, axis=0)       #store
    
    fig, (ax2) = plt.subplots(1,1)             #figures for plots
    fig1, (ax4) = plt.subplots(1,1)
    fig2, (ax3) = plt.subplots(1,1)

    ax2.plot(j_values, w, ".")
    ax2.set_xlabel("J", fontsize="30")
    ax2.set_ylabel("W(J)", fontsize="30")
    ax2.tick_params(labelsize=26)
    
    for i in range(len(phi_values)):
        ax3.plot(j_values, g_j[:,i], linewidth=2.5)
        ax3.set_xlabel("J", fontsize="30")
        ax3.set_ylabel("$\Gamma_J  (\phi)$", fontsize="30")
        ax3.tick_params(labelsize=26)

    s_class = []
    s_classical = classical_action(phi_values, m, xhi, lamb, s_class)

    ax4.plot(phi_values, s_classical, "k", label="$S(\phi)$")
    ax4.plot(phi_values, max_gamma, ".", label="$\Gamma(\phi)$")
    ax4.set_xlabel("$\phi$", fontsize="30")
    ax4.set_ylabel("$\Gamma(\phi)$", fontsize="30")
    ax4.tick_params(labelsize=26)
    ax4.legend(loc="upper right", fontsize=28)

    plt.show()
