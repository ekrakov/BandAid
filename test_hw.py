import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# Choose a grid resolution:
Ngrid = 200
J1_vals = np.linspace(-10, 10, Ngrid)
J2_vals = np.linspace(-10, 10, Ngrid)

# We will store an integer "phase index" at each (J1,J2) point:
#   0 -> FM
#   1 -> AFM
#   2 -> Helimagnet
phase_map = np.zeros((Ngrid, Ngrid), dtype=int)

for i, J1 in enumerate(J1_vals):
    for j, J2 in enumerate(J2_vals):

        # 1) FM energy:
        #    E_FM = -2 * (J1 + J2)
        E_FM = -2.0 * (J1 + J2)

        # 2) AFM energy:
        #    E_AFM = -2 * [ -J1 + J2 ] = 2 * (J1 - J2)
        E_AFM = 2.0 * (J1 - J2)

        # 3) Helimagnet (spiral):
        #    We only define it if |J1/(4*J2)| <= 1, else it's not a real angle.
        #    Then E_HM = 2 * [ J2 + (J1^2) / (8 * J2 ) ]
        #    If condition not satisfied, set E_HM large so it won't be chosen.
        if J2 != 0 and abs(J1/(4.0*J2)) <= 1.0:
            E_HM = 2.0 * ( J2 + (J1**2) / (8.0 * J2) )
        else:
            E_HM = 1e9  # A large number so it's effectively "not stable"

        # Now pick whichever is lowest:
        energies = [E_FM, E_AFM, E_HM]
        min_index = np.argmin(energies)  # 0->FM, 1->AFM, 2->HM
        phase_map[j, i] = min_index
        # note: indexing j-> rows is for J2, i->columns for J1 
        #       so phase_map is row-> J2, column-> J1

# Now we have a 2D "phase_map" with integer codes (0,1,2).
# Let's do a simple color-plot:
plt.figure(figsize=(7,6))

# Choose a discrete colormap or define our own:
# e.g.  0=red(FM), 1=blue(AFM), 2=green(helimagnet)
cmap = mcolors.ListedColormap(['red','blue','green'])

plt.imshow(phase_map,
           origin='lower',
           cmap=cmap,
           extent=[J1_vals[0], J1_vals[-1], J2_vals[0], J2_vals[-1]],
           interpolation='nearest', aspect='auto')

plt.colorbar(ticks=[0,1,2], label='Phase index')
plt.clim(-0.5, 2.5)  # to match discrete colors

plt.xlabel(r'$J_1$')
plt.ylabel(r'$J_2$')
plt.title('FM (red), AFM (blue), Helimagnet (green)')

plt.show()
