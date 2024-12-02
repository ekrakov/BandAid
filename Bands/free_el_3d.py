import numpy as np
import matplotlib.pyplot as plt

def get_plane_waves(b1, b2, b3, ecut):
    """
    Find the integer linear combinations of reciprocal lattice vectors
    with free-electron particle energy less than ecut.

    Parameters:
        b1, b2, b3: np.array
            Reciprocal lattice vectors.
        ecut: float
            Energy cutoff (Hartree).

    Returns:
        plane_waves: np.array
            Plane wave vectors (3 x num_plane_waves).
        num_plane_waves: int
            Number of plane waves.
    """
    max_index = int(np.ceil(np.sqrt(ecut) / (np.linalg.norm(b1) / np.sqrt(2))))
    plane_waves = []
    num_plane_waves = 0

    for i in range(-max_index, max_index + 1):
        for j in range(-max_index, max_index + 1):
            for k in range(-max_index, max_index + 1):
                g_vec = i * b1 + j * b2 + k * b3
                energy = np.linalg.norm(g_vec)**2 / 2
                if energy < ecut:
                    plane_waves.append(g_vec)
                    num_plane_waves += 1

    plane_waves = np.array(plane_waves).T  # Convert to a 3 x num_plane_waves array
    return plane_waves, num_plane_waves

# Atomic units
hartree_to_ev = 27.2
bohr_to_angstrom = 0.529177

# Parameters
ecut = 30 # Hartrees
fcc_conventional_cell_lattice_constant = 7.65  # for aluminum

# Define lattice vectors
a1 = np.array([0.5, 0.5, 0]) * fcc_conventional_cell_lattice_constant
a2 = np.array([0, 0.5, 0.5]) * fcc_conventional_cell_lattice_constant
a3 = np.array([0.5, 0, 0.5]) * fcc_conventional_cell_lattice_constant

vol = np.dot(a1, np.cross(a2, a3))
b1 = 2 * np.pi * np.cross(a2, a3) / vol
b2 = 2 * np.pi * np.cross(a3, a1) / vol
b3 = 2 * np.pi * np.cross(a1, a2) / vol

# Generate plane waves
plane_waves, num_plane_waves = get_plane_waves(b1, b2, b3, ecut)
print(f"Number of plane waves: {num_plane_waves}")

num_kvecs = 30
k_vecs = np.array([m / (num_kvecs - 1) * (b1 + b2 + b3) / 2 for m in range(num_kvecs)]).T

H_pot = np.zeros((num_plane_waves, num_plane_waves))

eigenvalues = []
for k in range(num_kvecs):
    H = H_pot.copy()
    for m in range(num_plane_waves):
        H[m, m] += np.linalg.norm(k_vecs[:, k] - plane_waves[:, m])**2 / 2
    eigvals = np.linalg.eigvalsh(H)
    eigenvalues.append(np.sort(eigvals))

eigenvalues = np.array(eigenvalues)



plt.figure()
ev_reversed = eigenvalues[::-1] * hartree_to_ev  # Reverse order, convert to eV
plt.plot(ev_reversed, linewidth=2, color='k')
plt.ylabel('Energy (eV)', fontsize=12)
plt.xlim([0, num_kvecs - 1])
plt.ylim([0, 30])
plt.xticks([0, num_kvecs - 1], ['L', 'Γ'])
plt.show()
