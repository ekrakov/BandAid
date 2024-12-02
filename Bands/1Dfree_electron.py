import numpy as np
import matplotlib.pyplot as plt

def free_electron_band(k_values, a=1.0, m=1.0, hbar=1.0):
    """
    Calculate the folded free electron band structure

    Args:
        k_values: Array of wave vectors.
        a: Lattice constant (default: 1.0).
        m: Electron mass (default: 1.0).
        hbar: Reduced Planck constant (default: 1.0).

    Returns:
        k_folded: Array of folded k-values in the first Brillouin Zone.
        E_folded: Energies of k_values
    """
    # Free electron energy
    E = (hbar**2 * k_values**2) / (2 * m)
    
    # we want to fold in BZ so I take mod (-π/a to π/a)
    k_folded = (k_values + np.pi / a) % (2 * np.pi / a) - np.pi / a
    
    sorted_indices = np.argsort(k_folded)
    k_folded = k_folded[sorted_indices]
    E_folded = E[sorted_indices]
    
    return k_folded, E_folded

def band_per_k(num_kpoints,a,m,hbar,band_number):
   
    k_values = np.linspace(-band_number * np.pi / a, band_number* np.pi / a, num_kpoints)  # values in bz
    # get folded bands 
    k_folded, E_folded = free_electron_band(k_values, a=a, m=m, hbar=hbar)

    return k_folded, E_folded,a
    
    
# Plot band structure
def main():
    k_folded, E_folded,a = band_per_k(num_kpoints=1000,a=1,m=1,hbar=1,band_number=4)
    plt.scatter(k_folded, E_folded, s=1, color='purple')
    # Add boundaries at BZ to help with visualization
    plt.axvline(-np.pi / a, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(np.pi / a, color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel("Wave Vector $k$ ($\pi/a$)")
    plt.ylabel("Energy $E$")
    plt.show()

    

if __name__ == "__main__":
    main()
