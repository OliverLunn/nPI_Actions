import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from matplotlib import colormaps
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

def integrand(phi, m, xhi, lamb, j, k):
    integrand = np.exp(-(m / 2) * (phi ** 2) - (1/3) * xhi * (phi**3) - (lamb/24) * (phi ** 4) + (j * phi) + (1/2 * k * phi**2))
    return integrand

def integration(j_values, k_values, m, xhi, lamb, z):
    for j in tqdm(range(len(j_values))):
        for k in range(len(k_values)):
            z_intermediate = integrate.quad(integrand, -np.inf, np.inf, args=(m, xhi, lamb, j_values[j], k_values[k]))
            z[k,j] = z_intermediate[0]
    return z

def gamma_j(phi, j_values, k_values, w, delta, array):
    """
    Calculates the Legendre-Fenchel transformation of W[J]. Returns array of \Gamma_j[phi] values.
    """
    for j in tqdm(range(len(j_values))):
            for k in range(len(k_values)):
                array[k,j] = w[k,j] + j_values[j] * phi + 1/2 * k_values[k] * (phi**2 + delta)
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
    return max_gamma, max_coord

if __name__ == '__main__':
   
    m, xhi, lamb = -2, 0, 4
    phi, delta = 0, 0
    step, min_val, max_val = 0.1, -10, 10
    j_values = np.arange(min_val, max_val+step, step)                  
    k_values = np.arange(min_val, max_val+step, step)
    
    z = np.zeros((len(j_values), len(k_values)))         
    g_j = np.zeros((len(j_values), len(k_values)))
    
    z = integration(j_values, k_values, m, xhi, lamb, z)
    w = -np.log(z)
    g = gamma_j(phi, j_values, k_values, w, delta, g_j)       #calling g_j  funct to generate g_j data
    g = g-np.nanmin(g)
    max_gamma, max_coord = maximise(g)
    #print(max_gamma)
    fig1, (ax1) = plt.subplots(1,1)
    #fig = plt.figure()

    X,Y = np.meshgrid(j_values, k_values)

    im1 = ax1.pcolormesh(j_values, k_values, g, cmap='viridis')
    ax1.contour(X,Y,g, colors=['black'], linestyles="solid")
    ax1.set_xlabel("J", fontsize="30")
    ax1.set_ylabel("K", fontsize="30")
    ax1.set_xticks([-10,-5,0,5,10])
    ax1.set_yticks([-10,-5,0,5,10])
    ax1.plot(j_values[max_coord[1]], k_values[max_coord[0]], ".k", markersize="12.5")
    ax1.tick_params(labelsize=28)
    cbar = plt.colorbar(im1)
    cbar.ax.tick_params(labelsize=28)
    ax1.set_aspect('equal')
    '''
    ax2 = fig.add_subplot(111, projection='3d')
    im2=ax2.plot_surface(X, Y, g, rstride=10, cstride=10, cmap="viridis", edgecolors='k', lw=0.6)
    ax2.set_xlabel("$J$", fontsize="28")
    ax2.set_ylabel("$K$", fontsize="28")
    ax2.set_zlabel("W(J,K)", fontsize="28")
    ax2.tick_params(labelsize=28)
    ax2.yaxis.labelpad = 25
    ax2.xaxis.labelpad = 25
    ax2.zaxis.labelpad = 25
    '''
    plt.tight_layout()
    plt.show()
