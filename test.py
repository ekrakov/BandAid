# import numpy as np
# import scipy.linalg
# from scipy.integrate import quad
# import matplotlib.pyplot as plt

# # -------------------------------
# # ⚙️ Set Parameters (Modify Here)
# # -------------------------------
# nWells = 15   # Number of wells
# nMax = 80        # Number of basis functions
# wWidth = 2      # Well width
# bHeight = 5    # Well depth (NEGATIVE potential)
# bWidth = .2   # Barrier width
# # -------------------------------

# # Define sine basis functions
# def basis(n, x, b):
#     return np.sqrt(2/b) * np.sin(n * np.pi * x / b)

# # Function to compute Hamiltonian and eigenstates for given barrier parameters
# def compute_eigenstates(bHeight, bWidth):
#     b = nWells * wWidth + (nWells + 1) * bWidth  # System width

#     # Define potential function correctly (negative wells)
#     def V_dynamic(x):
#         for i in range(nWells):
#             well_start = i * (wWidth + bWidth) + bWidth  # Shift wells after barriers
#             well_end = well_start + wWidth
#             if well_start <= x < well_end:
#                 return -abs(bHeight)  # NEGATIVE well depth
#         return 0  # Zero outside the wells

#     # Construct Hamiltonian matrix correctly
#     H_dynamic = np.zeros((nMax, nMax))
#     for m in range(1, nMax + 1):
#         for n in range(1, nMax + 1):
#             # Kinetic term
#             if m == n:
#                 H_dynamic[m-1, n-1] = (n**2 * np.pi**2 / (2 * b**2))

#             # Potential term (ensuring proper negative shift)
#             integral, _ = quad(lambda x: basis(m, x, b) * V_dynamic(x) * basis(n, x, b), 
#                                0, b, limit=200)
#             H_dynamic[m-1, n-1] += integral  # Adding potential term

#     # Solve for eigenvalues and eigenvectors
#     eigenvalues, eigenvectors = scipy.linalg.eigh(H_dynamic)

#     # Print the first 10 energy eigenvalues (should be negative for bound states)
#     print("Energy Eigenvalues (should be negative for bound states):")
#     print(eigenvalues[:30])

#     # Compute wavefunctions on grid
#     x_vals = np.linspace(0, b, 500)
#     wavefunctions = [sum(eigenvectors[j, i] * basis(j+1, x_vals, b) for j in range(nMax)) for i in range(30)]
#     potential_vals = np.array([V_dynamic(x) for x in x_vals])

#     return x_vals, potential_vals, wavefunctions, eigenvalues

# # Compute initial eigenstates correctly
# x_vals, potential_vals, wavefunctions, eigenvalues = compute_eigenstates(bHeight, bWidth)

# fig, ax = plt.subplots(figsize=(8, 5))

# # Plot potential
# ax.plot(x_vals, potential_vals, color='green', lw=2, label="Potential")

# scaling_factor = (abs(eigenvalues[0]) if eigenvalues[0] < 0 else 1) * 0.1  # Normalize wavefunction size
# for i, psi in enumerate(wavefunctions):
#     ax.plot(x_vals, psi * scaling_factor + eigenvalues[i], color=f"C{i}", label=f"State {i}")

# ax.set_xlabel("x")
# ax.set_ylabel("Energy")
# ax.set_title(f"Multi-Well Potential (Well Depth = {bHeight}, Barrier Width = {bWidth})")
# ax.axhline(0, color="black", lw=0.5, linestyle="dashed")  # Mark zero energy
# ax.legend()

# plt.show()
