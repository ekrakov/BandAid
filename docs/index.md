# What is band structure

To a first-order approximation, one could say that computational material science is essentially the prediction of material behavior. For example, we might wish to see if some arrangement of atoms forms a semiconductor or a metal, or to go further and compute the thermal and electrical conductivity of the material. Here, we focus on crystalline solids where we may assume the atoms are arranged periodically. The properties of materials are largely determined by the behavior of the electrons under the potential landscape created by this arrangement. 

Solving for electron behavior in an isolated atom yields the familiar $s$, $p$, $d$, $\ldots$, orbitals we get from solving the Schrödinger equation in spherical coordinates. However, a material contains many atoms, resulting in a periodic potential. Electrons now display complex \textit{interatomic} behavior, with linear combinations of bonding and antibonding states. This drastically changes the energy landscape: single-atom electron orbitals have neatly-spaced discrete energies, but electrons in a many-atom system have orbitals that hybridize and combine, yielding a continuous spectrum of allowed energies around the single-atom energies. These solutions are numerous enough to form a continuous band in reciprocal space, and the result is what is known as a band structure.  


Out interest is in solving the energy states of the schrodinger equation under a periodic potential such that $V(\br)=V(\br+\bR)$. The following tutorial will provide an introduction into how to think about the 


- [1D examples](1Dfinite.md)
- [Band structure tutorial](bands.md)
<!-- - [Usage]( -->