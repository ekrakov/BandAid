import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
import argparse

class PotentialFunctions:
    "class to compute different types of potental functions. Add your own!"

    def __init__(self, xval):
        self.x = xval

    def radial(self,l):
        """
        Radial potential function.
        Parameters:
        -l: Angular momentum quantum number
        Returns: Potential energy values
        """
        return l*(l+1)/self.x**2 - 2/self.x

    def Infsquare(self):
        """
        Infinite square well 
        Parameters:
        none 
        """

        return self.x *0

    def harmonic(self):
        pass 

    # Add other potential functions here as needed

def finite_diff(delta,potential):

    """
    Finite difference method for solving shrodinger equation equation. 
    Parameters:
    - delta: step size
    - potential: array of potential values
    Returns: eigenvalues and eigenvectors 
    """
    diag = 2./delta**2 + potential[1:-1]

    off = np.repeat(-1/delta**2,np.shape(diag)[0]-1)
    val, vec = eigh_tridiagonal(diag, off)
    return val, vec


def plot_eigenstates(x, eigenvectors, eigenvalues, title, xlabel, ylabel):
    """
    Function to plot eigenstates.
    """
    plt.figure(figsize=(10, 6))
    E0 = 13.6
    for i, vec in enumerate(eigenvectors):

        plt.plot(x, vec, label=f'{eigenvalues[i]:.3f}  eV')
    plt.title(title, fontsize=30)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.legend()
    plt.show()

def main(l):
    """
    Runs case for radial function for a given l
    Parameters:
    - delta: step size
    - potential: array of potential values
    Returns: eigenvalues and eigenvectors 
    """

    # Constants
    E0 = 13.6  # Energy conversion factor
    # Parameters
    n_grid = 10000  # number of grid points 
    l_max = 300 # maximum length

    # Create x values
    x_values = np.linspace(0.00001, l_max, n_grid)
    delta = x_values[1] - x_values[0]

    # Instantiate PotentialFunctions and compute potential
    potential_obj = PotentialFunctions(x_values)
    # radial_potential = potential_obj.radial(l)
    radial_potential = potential_obj.radial(l)


    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = finite_diff(delta, radial_potential)
    eigenvalues = np.sort(eigenvalues)

    # Plot the results
    plot_eigenstates(x_values[1:-1], eigenvectors.T[:3], eigenvalues[:3] *E0 ,
                     'Eigenstates for (l={})'.format(str(l)), '$x/a_0$', '$\psi(r)$')

    # Display the first few eigenvalues
    for i in range(12):
        n=i+1

        # the true eigenvalue is -E0/n**2 where E0 is the ground state for hydrogen. 
        # For solutions with angular momentum l, the lowest possible energy they can occupy is the n+l state, which is why the true eigenvalue reads E0=(n_l)**2

        true_eigenvalue = -E0/ ((n+l) ** 2)
        calculated_eigenvalue = eigenvalues[i] * E0

        # Check if values are different and print in red if so
        if round(true_eigenvalue, 2) != round(calculated_eigenvalue, 2):
            print(f"\033[91mtrue eigenvalue :{true_eigenvalue:8.3f}, calculated eigenvalue :{calculated_eigenvalue:8.3f}\033[0m")
        else:
            print(f"true eigenvalue :{true_eigenvalue:8.3f}, calculated eigenvalue :{calculated_eigenvalue:8.3f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate and plot eigenstates for different angular momentum quantum numbers using finite difference method")
    parser.add_argument("l",nargs='?', default=3, type=int,help="Angular momentum quantum number")
    args = parser.parse_args()
    print(args)
    main(args.l)
