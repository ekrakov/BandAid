import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

class ShootingSolver:
    def __init__(self, l, boundary_conditions, r_range):
        self.l = l
        self.boundary_conditions = boundary_conditions
        self.r_values = np.logspace(*r_range, 100000)
        self.energies = []

    def first_order(self, r, y, En):
        """
        Converts the Schroedinger equation to two first-order differential equations.
        """
        u, up = y
        l = self.l
        return np.array([up, (l * (l + 1) / r**2 - 2 / r - En) * u], dtype="object")

    def solve(self,E0):
        ur = solve_ivp(self.first_order, (self.r_values[0], self.r_values[-1]),
        self.boundary_conditions, method='DOP853', t_eval=self.r_values, args=(E0,))
        ur = ur.y[0, :]
        return ur 

    def residual(self, E0):
        """
        Residual function for least squares optimization.
        """
        res = self.boundary_conditions[0] - self.solve(E0)[-1]
        return res

    def scan_for_roots(self, E_range, tol=1e-7):
        """
        Scans for roots of the function using the original method.
        """
        # found_roots = []
        for E_guess in E_range:
            root = fsolve(self.residual, E_guess)
            if not any(np.abs(root - r) < tol for r in self.energies):
                self.energies.append(root[0])
        return self.energies

    def plot_wavefunction(self,energies):
        """
        Plots the wavefunction for the last calculated energy level.
        """
        wavefunctions = [self.solve(E) for E in energies]
        plt.plot(self.r_values, np.array(wavefunctions).T)
        plt.xlabel(r'$r$')
        plt.ylabel(r'$\psi(r)$')
        plt.title('Wavefunctions for Different Energy Levels')
        plt.grid(True)
        plt.show()

# Usage of QuantumSolver Class
solver = ShootingSolver(l=0, boundary_conditions=[0, 1.0], r_range=(-6, .6))

# Define your energy range
E_range = np.linspace(-1, -.01, 20)  # Adjust as needed
found_energies = solver.scan_for_roots(E_range)

solver.plot_wavefunction([-.9])

def true_energy (n,l=0):
    E0=13.6
    E=-1/ ((n+l) ** 2)
    return E
n_values = range(1, 5)  # Adjust the range as needed
true_energies = [true_energy(n) for n in n_values]
print(f"{'Found Energy':>15} {'Closest True Energy':>22} {'n Value':>8} {'Difference':>11}")

for found_energy in found_energies:
    closest_true_energy = min(true_energies, key=lambda x: abs(x - found_energy))
    closest_n = n_values[true_energies.index(closest_true_energy)]
    difference = abs(found_energy - closest_true_energy)

    # Formatting the output
    found_energy_str = f"{found_energy:.3f}"
    closest_true_energy_str = f"{closest_true_energy:.3f}"
    difference_str = f"{difference:.3f}"

    if difference > 0.01:
        # Print in red
        print(f"\033[91m{found_energy_str:>15} {closest_true_energy_str:>22} {closest_n:>8} {difference_str:>11}\033[0m")
    else:
        # Normal print
        print(f"{found_energy_str:>15} {closest_true_energy_str:>22} {closest_n:>8} {difference_str:>11}")
