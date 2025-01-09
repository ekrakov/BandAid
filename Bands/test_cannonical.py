import re
import numpy as np
# from test_gaunt import calculate_gaunt_skriver
from math import factorial
from scipy.linalg import eig 
from scipy.special import sph_harm
import matplotlib.pyplot as plt

plt.style.use("~/.matplotlib/styles/style.mplstyle")


def calculate_gaunt_skriver(lp, l, mp, m):
    """
    Calculate Skriver's glm parameter 
    """
    mpp = mp - m
    lpp = l+lp
    glm_numerator = (2 * lp + 1) * (2 * l + 1) * factorial(lpp + mpp) * factorial(lpp - mpp)
    glm_denominator = (2 * lpp + 1) * factorial(lp + mp) * factorial(lp - mp) * factorial(l + m) * factorial(l - m)
    glm = (-1)**(int(m + 1)) * 2 * np.sqrt(glm_numerator / glm_denominator)

    return glm 

def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi).
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return r, theta, phi



def spherical_harmonic(l,m,X,Y,Z):

    r, theta, phi = cartesian_to_spherical(X,Y,Z)
    # note that in scipy, it is theta, phi but they define their theta as the azimuthal angle.
    sph_harmonic = sph_harm(m, l, phi, theta, out=None)

    return sph_harmonic

def genR(a1, a2, a3, rad):
    N = rad+1
    """
    Generate the repeated R vectors to do the bloch sum

    """
    n = np.arange(-N, N + 1)
    n1, n2, n3 = np.meshgrid(n, n, n, indexing='ij')
    R = (n1[..., None]*a1 +
         n2[..., None]*a2 +
         n3[..., None]*a3)
    
    R_list = R.reshape(-1, 3)
    norms = np.linalg.norm(R_list, axis=1)
    mask = (norms != 0) & (norms <=rad)
    R_list= R_list[mask]

    # sort by in accending order. Could that be the issue ?
    norms = np.linalg.norm(R_list, axis=1)
    sorted_idx = np.argsort(norms)
    
    R_sort = R_list[sorted_idx]

    return R_sort


def bcc_lattice(bcc = 1):
    """
    Returns primitive real-space vectors, recip vectors and volume for an FCC lattice
    """


    a1 = np.array([0.5, 0.5, -.5]) * bcc
    a2 = np.array([-.5, 0.5, 0.5]) * bcc
    a3 = np.array([0.5, -.5, 0.5]) * bcc


    vol = np.dot(a1, np.cross(a2, a3))

    b1 = 2 * np.pi * np.cross(a2, a3) / vol
    b2 = 2 * np.pi * np.cross(a3, a1) / vol
    b3 = 2 * np.pi * np.cross(a1, a2) / vol


    S3 = (vol*3)/(4*np.pi)
    S = S3**(1/3)

    # BCC
    symmetry_points = {
    "Γ": np.array([0.0, 0.0, 0.0]),
    "H": np.array([-0.5, 0.5, 0.5]),
    "P": np.array([0.25, 0.25, 0.25]),
    "N": np.array([0.0,0.0, 0.5])
    }
    
    return a1, a2, a3, b1, b2, b3, vol, S, symmetry_points


def calculate_k_path(start_vec, finish_vec, num_kpoints):
    # Generate K along a given path 
    path = start_vec + np.linspace(0, 1, num_kpoints)[:, None] * (finish_vec - start_vec)
    return path

def recip_path(b1,b2,b3,symmetry_points):
    """
    find cartesian representation of symmetry points given in symmetry points 
    """

    recip_vectors = {
        key: value[0] * b1 + value[1] * b2 + value[2] * b3
        for key, value in symmetry_points.items()
    }

    return recip_vectors


def compute_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def Sk(K,R,S,gaunt,lpp,mpp):
    """
    Calculate Structure constants for a specified k value. (reference skriver equation)
    """
    val_total = np.longdouble(0)

    for ls in R:
        ls = ls.astype(np.longdouble)
        frac = (S/np.linalg.norm(ls))**(lpp+1)
        const = np.conjugate(np.sqrt(4*np.pi)*((1j)**lpp)*spherical_harmonic(lpp,mpp,ls[0],ls[1],ls[2]))
        exp = np.exp(1j*np.dot(K,ls))
        val = exp*const*gaunt*frac
        val_total += val

    return val_total


a1, a2, a3, b1, b2, b3, vol, S, symmetry_points= bcc_lattice()
recip_path = recip_path(b1,b2,b3,symmetry_points)
 

# set l for unhybridized bands 

def calculate_k_path(start_vec, finish_vec, num_kpoints):
    # Generate K along a given path 
    path = start_vec + np.linspace(0, 1, num_kpoints)[:, None] * (finish_vec - start_vec)
    return path

path = calculate_k_path(np.array([0,0,0]),np.array([2*np.pi,0,0]),100)

def loop(l,value,R_list):
    shape = 2 * l + 1  
    lp=l
    lpp=l+lp
    result_matrix = np.zeros((shape, shape))
    for i, m in enumerate(np.arange(-l,l+1)):
        for j, mp in enumerate(np.arange(-l,l+1)):
            mpp = mp - m
            gaunt = calculate_gaunt_skriver(lp, l, mp, m)
            result = Sk(value, R_list, S, gaunt, lpp, mpp)
            res=np.real(result)
            # Store in matrix - convert from (-l:l) indexing to (0:2l) indexing
            result_matrix[i, j] = result

    # print(key,result_matrix.T.diagonal())
    eigenvalues,eigenvectors = eig(result_matrix)
    eigsort=np.sort(eigenvalues)
    return eigsort
R_list = genR(a1,a2,a3,6) # generate lattice repeats 

itter=[]
it=[]
o=0
for i in path:
    a = loop(2, i, R_list)   # Suppose this returns an array
    itter.append(a)
    it.append(o)
    o += 1
import matplotlib.pyplot as plt

for xval, arr in zip(it, itter):
    plt.scatter([xval]*len(arr), arr, color="black",s=10)

# Replace x-axis ticks:
#   - At x=0, show the label Γ (via LaTeX: r'$\Gamma$')
#   - At x=1, show the label H
plt.xticks([0, 100], [r'$\Gamma$', 'H'])

# Label the y-axis as s_d
plt.ylabel(r"$s_d$")

# Optionally remove x-axis label if you don't want any text there
plt.xlabel("")  # or leave it out entirely

plt.show()


for xval, arr in zip(it, itter):
    # xval is your iteration index
    # arr is the array of points you want to plot at this xval
    plt.scatter([xval]*len(arr), arr,color="black")
print(it,'hererere')

plt.xlabel("Iteration index")
plt.ylabel("Values from itter")
plt.show()

import numpy as np
itter_array = np.array(itter)  # shape: (number_of_iters, m)

# it is [0, 1, 2, ..., number_of_iters-1]
# Transpose to get columns (each row in new_array is the k-th element across all iters)
# new_array[k, :] = [ itter[0][k], itter[1][k], ..., itter[num_iters-1][k] ]
new_array = itter_array.T
for k in range(new_array.shape[0]):
    # y-values for the k-th element across each iteration
    yvals = new_array[k, :]
    plt.plot(it, yvals, color="black")  # A line for the k-th element across iterations

plt.xlabel("Iteration index")
plt.ylabel("Values from itter")
plt.show()
# def get_specialpoints(l):

#     lp=l
#     lpp=l+lp
#     rcut = 10
#     R_list = genR(a1,a2,a3,rcut) # generate lattice repeats 

#     eig_result = []
#     val_result = []
#     k_value = []
#     for key, value in recip_path.items():
#         shape = 2 * l + 1  
#         result_matrix = np.zeros((shape, shape))
#         for i, m in enumerate(np.arange(-l,l+1)):
#             for j, mp in enumerate(np.arange(-l,l+1)):
#                 mpp = mp - m
#                 gaunt = calculate_gaunt_skriver(lp, l, mp, m)
#                 result = Sk(value, R_list, S, gaunt, lpp, mpp)
#                 res=np.real(result)
                
#                 # Store in matrix - convert from (-l:l) indexing to (0:2l) indexing
#                 result_matrix[i, j] = result

#         # print(key,result_matrix.T.diagonal())
#         eigenvalues,eigenvectors = eig(result_matrix)
#         eig_result.append(eigenvalues)
#         val_result.append(key)
#         k_value.append(value)
#     return val_result, np.real(eig_result),k_value


# paths = [
#     ("Γ", "X"),
#     ("X", "W"),
#     ("W", "L"),
#     ("L", "Γ"),
#     ("Γ", "X")

# ]


# def format_print(val_result, eig_result,k_result):
#     print("Table 2.2 Skriver")
#     print("-" * 40)
#     for key, eigenvalues,k_point in zip(val_result, eig_result,k_result):
#         print(f": {key}: {k_point}")
#         print(f"Eigenvalues: {', '.join(f'{ev:.4f}' for ev in eigenvalues)}")
#         print("-" * 40)

# if __name__ == "__main__":
#     val_result, eig_result,k_point = get_specialpoints(2)
#     format_print(val_result, eig_result,k_point)