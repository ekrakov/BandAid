from operator import ne
import numpy as np
import scipy.linalg
import streamlit as st
import matplotlib.pyplot as plt

def basis(n, x, b):
    """
    Use sine function as basis function since dealing with square well. maybe extend later???
    Asin
    """
    A = np.sqrt(2.0/b)
    return A*np.sin(n*np.pi*x/b)

def int_sinsquare(m, n, x1, x2, b):
    """
   Numerical integration was taking too long so I tried manually integrating sin function. The main issue is the discontinuity in the potential. 
    """
    alpha = m*np.pi/b
    beta  = n*np.pi/b
    A2  = 2.0/b  # A squared
    if n==m:
        def F(x):
            return 0.5*x - np.sin(2*alpha*x)/(4.0*alpha)
    else:# m neq n 
        def F(x):
            return (np.sin((alpha - beta)*x)/(2.0*(alpha-beta))- np.sin((alpha + beta)*x)/(2.0*(alpha+beta)))
    return A2*(F(x2) - F(x1))


def square_potential(nWells, wWidth, bWidth, bHeight):
    """
    Returns a list of (start, end, V_value) intervals describing
    the piecewise potential. 
    """
    intervals = []
    for i in range(nWells):
        well_start = i*(wWidth + bWidth) + bWidth
        well_end   = well_start + wWidth
        intervals.append((well_start, well_end, -abs(bHeight)))
    return intervals


def build_hamiltonian(nWells, wWidth, bWidth, bHeight, nMax):
    """
    Here we construct the hamiltonian H_mn = T_mn + V_mn
    """
    b = nWells*wWidth + (nWells+1)*bWidth
    well_intervals = square_potential(nWells, wWidth, bWidth, bHeight)
    H = np.zeros((nMax, nMax), dtype=float)

    for m in range(1, nMax+1):
        for n in range(1, nMax+1):
            # Compute kinetic energy (I would  rather do this numerically. Cite source ) 
            if m == n:
                kinetic = (n**2 * np.pi**2)/(2.0*b**2)
            else:
                kinetic = 0.0
            # get potential for each corresponding area. We do this in a peacewise way and it only workss for a square potential. 
            pot_val = 0.0
            for (start, end, V_val) in well_intervals:
                overlap = int_sinsquare(m, n, start, end, b)
                pot_val += V_val * overlap

            H[m-1, n-1] = kinetic + pot_val

    return H


def get_eig(nWells, wWidth, bWidth, bHeight, nMax):
    """
   Construct hamiltonian and diagonalize
   reconstruct states by multiplying by basis
    """
    b = nWells*wWidth + (nWells+1)*bWidth
    H = build_hamiltonian(nWells, wWidth, bWidth, bHeight, nMax)
    # Diagonalize
    eigenvals, eigenvects = scipy.linalg.eigh(H)  # sorted for plotting band structure

    # Reconstruct wavefunctions based on cofficients computed from diagonalization
    x_vals = np.linspace(0, b, 500)
    wavefunctions = []
    for i in range(nMax):
        coeffs = eigenvects[:, i]
        psi_i = np.zeros_like(x_vals, dtype=float)
        for j in range(nMax):
            psi_i += coeffs[j]*basis(j+1, x_vals, b)
        wavefunctions.append(psi_i)

    # Potential construction
    V_vals = np.zeros_like(x_vals)
    for idx, xv in enumerate(x_vals):
        val = 0.0
        for i in range(nWells):
            start = i*(wWidth+bWidth) + bWidth
            end = start + wWidth
            if start <= xv < end:
                val = -abs(bHeight)
                break
        V_vals[idx] = val

    return x_vals, V_vals, wavefunctions, eigenvals



############################################################
######################################################
############ This is using streamlit!!!###############
st.title("Solutions to Finite Potential Well")

nWells = st.sidebar.slider("Number of Wells", 1, 20, 5)
wWidth = st.sidebar.slider("Well Width", 0.1, 5.0, 2.0, 0.1)
bWidth = st.sidebar.slider("Barrier Width", 0.01, 1.0, 0.2, 0.01)
bHeight= st.sidebar.slider("Well Depth", 0.1, 20.0, 5.0, 0.1)
nMax   = st.sidebar.slider("Basis Size", 5, 200, 40, 5)
x_vals, V_vals, wavefuncs, evals = get_eig(nWells, wWidth, bWidth, bHeight, nMax)

# make sure only bound states are plotted 
mask = (evals < 0)
negative_indices = np.where(mask)[0]   
negative_evals   = evals[mask]         
shape = np.shape(negative_evals)[0]

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(x_vals, V_vals, 'g-',color='maroon')
total_length = len(negative_evals)
scale = 0.1 * abs(negative_evals[0])
for i in range(total_length):
    ax.plot(x_vals, wavefuncs[i]*scale + evals[i])

ax.axhline(0, color='k', ls='--', lw=0.5)
ax.set_xlabel("position (x)")
ax.set_ylabel("Energy")
ax.legend(loc="best")

st.pyplot(fig)


chosen_idx = st.slider("Select eigenvalue index to view corresponding wavefunction", 0, shape-1, 0)

fig1, ax1 = plt.subplots(figsize=(7, 3))
indices = np.arange(shape)
ax1.scatter(negative_indices, negative_evals)
ax1.plot(chosen_idx, negative_evals[chosen_idx], 'ro', markersize=8, label="current index",color='maroon')
ax1.set_xlabel("Eigenvalue Index n")
ax1.set_ylabel("Energy")
ax1.set_title("Band structure!")
ax1.legend()
st.pyplot(fig1)


fig2, ax2 = plt.subplots(figsize=(7,4))
psi_chosen = wavefuncs[chosen_idx]
ax2.plot(x_vals, psi_chosen, 'b-', label="psi",color = 'purple')
ax2.plot(x_vals, psi_chosen**2, 'r--', label="|psi|^2",color='pink')
ax2.set_xlabel("x")
ax2.set_ylabel("Amplitude")
ax2.set_title(f"Wavefunction and probability amplitude for eigenvalue {chosen_idx}\n(E={evals[chosen_idx]:.3f})")
ax2.legend()
st.pyplot(fig2)