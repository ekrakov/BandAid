import numpy as np
import scipy.linalg
from scipy.integrate import quad
import matplotlib.pyplot as plt
import json
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import json



import numpy as np
import scipy.linalg
from scipy.integrate import quad
import matplotlib.pyplot as plt
import json

# Manually defined parameter lists
bHeights_list = [10, 15]  # Specific well depths
nMax = 30  # Number of basis functions
wWidth_list = [0.8, 0.9, 1.0]  # Specific well widths
bWidth_list = [0.8, 1.4, 2.0]  # Specific barrier widths

# Define sine basis functions
def basis(n, x, b):
    return np.sqrt(2/b) * np.sin(n * np.pi * x / b)

# Function to compute Hamiltonian and eigenstates using QUAD for integration
def compute_eigenstates(nWells, bHeight, wWidth, bWidth):
    b = nWells * wWidth + (nWells + 1) * bWidth  # System width

    # Define potential function for multiple wells
    def V_dynamic(x):
        for i in range(nWells):
            well_start = i * (wWidth + bWidth) + bWidth  # Shift wells after barriers
            well_end = well_start + wWidth
            if well_start <= x < well_end:
                return -abs(bHeight)  # Negative well depth
        return 0  # Zero outside the wells

    # Construct Hamiltonian matrix using QUAD
    H_dynamic = np.zeros((nMax, nMax))
    for m in range(1, nMax + 1):
        for n in range(1, nMax + 1):
            # Kinetic term
            if m == n:
                H_dynamic[m-1, n-1] = (n**2 * np.pi**2 / (2 * b**2))

            # Potential term using QUAD integration
            integral, _ = quad(lambda x: basis(m, x, b) * V_dynamic(x) * basis(n, x, b), 
                               0, b, limit=200)
            H_dynamic[m-1, n-1] += integral

    # Solve for eigenvalues and eigenvectors
    eigenvalues, eigenvectors = scipy.linalg.eigh(H_dynamic)

    # Compute wavefunctions on grid
    x_vals = np.linspace(0, b, 500)
    wavefunctions = [sum(eigenvectors[j, i] * basis(j+1, x_vals, b) for j in range(nMax)) for i in range(4)]
    potential_vals = np.array([V_dynamic(x) for x in x_vals])

    return x_vals, potential_vals, wavefunctions, eigenvalues

# Compute eigenstates for different well configurations, depths, and widths
data_store = {}
for nWells in [1, 2, 3]:  # Limiting to 1-3 wells
    for bHeight in bHeights_list:
        for wWidth in wWidth_list:
            for bWidth in bWidth_list:
                key = f"{nWells}_wells_{bHeight}_depth_{wWidth:.1f}_wWidth_{bWidth:.1f}_bWidth"
                x_vals, potential_vals, wavefunctions, eigenvalues = compute_eigenstates(nWells, bHeight, wWidth, bWidth)

                data_store[key] = {
                    "x_vals": x_vals.tolist(),
                    "potential": potential_vals.tolist(),
                    "wavefunctions": [psi.tolist() for psi in wavefunctions],
                    "eigenvalues": eigenvalues[:4].tolist()
                }

# Save to JSON file
json_file = "wv2.json"
with open(json_file, "w") as f:
    json.dump(data_store, f, indent=4)

# Select an example to plot (2 wells, depth 10, width 0.9, barrier 1.4)
example_key = "2_wells_10_depth_0.9_wWidth_1.4_bWidth"
if example_key not in data_store:
    raise ValueError(f"Key {example_key} not found. Available keys: {list(data_store.keys())[:5]}")

example_data = data_store[example_key]

# Extract data for plotting
x_vals = np.array(example_data["x_vals"])
potential_vals = np.array(example_data["potential"])
wavefunctions = np.array(example_data["wavefunctions"])
eigenvalues = np.array(example_data["eigenvalues"])

# Plot potential and wavefunctions
fig, ax = plt.subplots(figsize=(8, 5))

# Plot potential
ax.plot(x_vals, potential_vals, color='green', lw=2, label="Potential")

# Plot eigenfunctions at their energy levels
scaling_factor = 0.1  # Normalize wavefunction size
for i, psi in enumerate(wavefunctions):
    ax.plot(x_vals, psi * scaling_factor + eigenvalues[i], color=f"C{i}", label=f"State {i}")

ax.set_xlabel("x")
ax.set_ylabel("Energy / Wavefunction")
ax.set_title(f"Example: {example_key.replace('_', ' ')}")
ax.axhline(0, color="black", lw=0.5, linestyle="dashed")  # Mark zero energy
ax.legend()
ax.grid()

plt.show()

# Provide JSON file for download
json_file














# # Expanded parameter ranges
# bHeights_list = [10, 15,20]  # Specific well depths
# nMax = 50  # Reduced number of basis functions for faster performance
# wWidth_list = [0.8, 0.9, 1.0,2.0]  # Specific well widths
# bWidth_list = [0.8,1.0, 1.4, 2.0]  # Specific barrier widths

# # Define sine basis functions
# def basis(n, x, b):
#     return np.sqrt(2/b) * np.sin(n * np.pi * x / b)

# # Construct Hamiltonian matrix using a fast approach
# def compute_eigenstates(nWells, bHeight, wWidth, bWidth):
#     b = nWells * wWidth + (nWells + 1) * bWidth  # System width

#     # Define potential function
#     def V_dynamic(x):
#         for i in range(nWells):
#             well_start = i * (wWidth + bWidth) + bWidth  # Shift wells after barriers
#             well_end = well_start + wWidth
#             if well_start <= x < well_end:
#                 return -abs(bHeight)  # Negative well depth
#         return 0  # Zero outside the wells

#     # Construct Hamiltonian matrix
#     H_dynamic = np.zeros((nMax, nMax))
#     x_vals = np.linspace(0, b, 500)

#     # Compute potential values once to avoid redundant function calls
#     V_vals = np.array([V_dynamic(x) for x in x_vals])

#     # Precompute basis functions on grid
#     basis_matrix = np.array([[basis(n + 1, x, b) for x in x_vals] for n in range(nMax)])

#     # Compute kinetic term (diagonal)
#     for m in range(nMax):
#         H_dynamic[m, m] = ((m + 1) ** 2 * np.pi ** 2) / (2 * b ** 2)

#     # Compute potential term using a fast matrix approach
#     for m in range(nMax):
#         for n in range(nMax):
#             H_dynamic[m, n] += np.trapz(basis_matrix[m] * V_vals * basis_matrix[n], x_vals)

#     # Solve for eigenvalues and eigenvectors
#     eigenvalues, eigenvectors = scipy.linalg.eigh(H_dynamic)

#     # Compute wavefunctions on grid
#     wavefunctions = [
#         np.sum(eigenvectors[:, i, np.newaxis] * basis_matrix, axis=0) for i in range(4)
#     ]

#     return x_vals, V_vals, wavefunctions, eigenvalues[:4]

# # Compute eigenstates for different well configurations
# data_store = {}
# for nWells in [1, 2, 3]:
#     for bHeight in bHeights_list:
#         for wWidth in wWidth_list:
#             for bWidth in bWidth_list:
#                 key = f"{nWells}_wells_{bHeight}_depth_{wWidth:.1f}_wWidth_{bWidth:.1f}_bWidth"
#                 x_vals, potential_vals, wavefunctions, eigenvalues = compute_eigenstates(nWells, bHeight, wWidth, bWidth)

#                 data_store[key] = {
#                     "x_vals": x_vals.tolist(),
#                     "potential": potential_vals.tolist(),
#                     "wavefunctions": [psi.tolist() for psi in wavefunctions],
#                     "eigenvalues": eigenvalues.tolist()
#                 }

# # Save to JSON file
# json_file = "wv2.json"
# with open(json_file, "w") as f:
#     json.dump(data_store, f, indent=4)

# # Select an example to plot (2 wells, depth 20, width 3, barrier 1)
# example_key = "1_wells_20_depth_0.8_wWidth_0.8_bWidth"
# example_data = data_store[example_key]

# # Extract data for plotting
# x_vals = np.array(example_data["x_vals"])
# potential_vals = np.array(example_data["potential"])
# wavefunctions = np.array(example_data["wavefunctions"])
# eigenvalues = np.array(example_data["eigenvalues"])

# # Plot potential and wavefunctions
# fig, ax = plt.subplots(figsize=(8, 5))

# # Plot potential
# ax.plot(x_vals, potential_vals, color='green', lw=2, label="Potential")

# # Plot eigenfunctions at their energy levels
# scaling_factor = 0.1  # Normalize wavefunction size
# for i, psi in enumerate(wavefunctions):
#     ax.plot(x_vals, psi * scaling_factor + eigenvalues[i], color=f"C{i}", label=f"State {i}")

# ax.set_xlabel("x")
# ax.set_ylabel("Energy / Wavefunction")
# ax.set_title(f"Example: {example_key.replace('_', ' ')}")
# ax.axhline(0, color="black", lw=0.5, linestyle="dashed")  # Mark zero energy
# ax.legend()
# ax.grid()

# plt.show()

# # Provide JSON file for download
# json_file
