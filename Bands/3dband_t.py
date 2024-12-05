import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def get_plane_waves(b1, b2, b3, ecut):
   
    max_index = int(np.ceil(np.sqrt(ecut) / (np.linalg.norm(b1) / np.sqrt(2))))
    
    # Generate all combinations of i, j, k
    indices = np.arange(-max_index, max_index + 1)
    i, j, k = np.meshgrid(indices, indices, indices, indexing='ij')
    
    wave_vectors = (i.ravel()[:, None] * b1 +
                    j.ravel()[:, None] * b2 +
                    k.ravel()[:, None] * b3)
    
    # Compute norms squared and apply energy cutoff
    norms_squared = np.sum(wave_vectors ** 2, axis=1) / 2
    valid_indices = norms_squared < ecut
    valid_wave_vectors = wave_vectors[valid_indices]
    
    return valid_wave_vectors.T, valid_wave_vectors.shape[0]


def compute_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def calculate_k_path(start_vec, finish_vec, num_kpoints):
    path = start_vec + np.linspace(0, 1, num_kpoints)[:, None] * (finish_vec - start_vec)
    return path

symmetry_points = {
    "Γ": np.array([0, 0, 0]),
    "X": np.array([0, 1/2, 1/2]),
    "L": np.array([1/2, 1/2, 1/2]),
    "W": np.array([1/4, 3/4, 1/2]),
    "U": np.array([1/4, 5/8, 5/8]),
    "K": np.array([3/8, 3/4, 3/8])
}
ecut = 12 # Hartrees
fcc_lc = 7.65  # for aluminum

a1 = np.array([0.5, 0.5, 0]) * fcc_lc
a2 = np.array([0, 0.5, 0.5]) * fcc_lc
a3 = np.array([0.5, 0, 0.5]) * fcc_lc

# RLV
vol = np.dot(a1, np.cross(a2, a3))
b1 = 2 * np.pi * np.cross(a2, a3) / vol
b2 = 2 * np.pi * np.cross(a3, a1) / vol
b3 = 2 * np.pi * np.cross(a1, a2) / vol

recip_vectors = {
    key: value[0] * b1 + value[1] * b2 + value[2] * b3
    for key, value in symmetry_points.items()
}


plane_waves, num_plane_waves = get_plane_waves(b1, b2, b3, ecut)
print("Number of plane waves:", num_plane_waves)


paths = [
    ("Γ", "X"),
    ("X", "W"),
    ("W", "L"),
    ("L", "Γ"),
    ("Γ", "X")

]

eigenvals_path = []
kpoints_arr = [] 
H_pot = np.zeros((num_plane_waves, num_plane_waves))

for start, finish in paths:
    start_vec = recip_vectors[start]
    finish_vec = recip_vectors[finish]
    distance = compute_distance(start_vec, finish_vec)
    num_kpoints = int(np.ceil(distance*80))  #UHHH find better
    k_points = calculate_k_path(start_vec, finish_vec, num_kpoints)
    eigenvalues = np.zeros((num_kpoints, num_plane_waves))

    # Loop over k-points
    for k in range(num_kpoints):
        H = np.copy(H_pot)  # Start with the potential part of H
        # Add kinetic energy terms to the diagonal of H
        for m in range(num_plane_waves):
            H[m, m] += np.linalg.norm(k_points[k] - plane_waves[:, m]) ** 2 / 2
        # Compute eigenvalues of H
        eigenvalues[k, :] = np.sort(np.linalg.eigvalsh(H))
    
    eigenvals_path.append(eigenvalues)
plt.figure(figsize=(10, 6))

cumulative_k = 0  # Tracks the starting position for the current path
xticks = [0] 
for eigenvalues, path in zip(eigenvals_path, paths):
    num_kpoints = eigenvalues.shape[0]  # Number of k-points in the current bz path
    k_points = np.linspace(0, num_kpoints - 1, num_kpoints)  # Local k-axis for this path

    for band in range(num_plane_waves):
        plt.plot(k_points + cumulative_k, eigenvalues[:, band], color='k')
# cry i need a better way
    cumulative_k += num_kpoints - 1 
    xticks.append(cumulative_k)

symmetry_labels = ["Γ", "X", "W", "L", "Γ","X"]

for x in xticks:
    plt.axvline(x=x, color='gray', linestyle='--', linewidth=1)

plt.xticks(xticks, symmetry_labels)
plt.ylim([0, 1.5])

# Plot formatting
plt.title("FCC Free Electron Band")
plt.xlabel("k-path")
plt.ylabel("E (eV)")
plt.show()
