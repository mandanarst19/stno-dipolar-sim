"""
Physical constants (SI) and the current-polarity convention used throughout the package.

Polarity convention
-------------------
The spin-transfer torque enters the equation of motion as

    dm/dt|_STT = gamma * mu0 * a_J * g(m.s) * m x (m x s)        (a_J in A/m)

where s is the polarizer direction. The sign of a_J is set by the direction of the current:

* POLARITY_ANTIPARALLEL (+1): a_J > 0, the torque pushes m ANTI-parallel to s (m.s < 0).
  Physically: electrons flow from the free layer towards the polarizer.
* POLARITY_PARALLEL (-1): a_J < 0, the torque pushes m PARALLEL to s (m.s > 0).
  Physically: electrons flow from the polarizer into the free layer.
  This is the branch simulated by Chen et al., J. Appl. Phys. 115, 134306 (2014).

The convention was checked against Y. Zhou, PhD thesis (KTH, 2009), Eq. 2.8, and
T. Taniguchi & H. Kubota, Phys. Rev. B 93, 174401 (2016). Both polarities give
self-oscillations, but with different frequencies and current ranges because the
Slonczewski efficiency g(m.s) is asymmetric.
"""
import numpy as np

GAMMA = 1.76e11            # gyromagnetic ratio, rad / (s T)
MU0 = 4e-7 * np.pi         # vacuum permeability, T m / A
HBAR = 1.054571817e-34     # reduced Planck constant, J s
QE = 1.602176634e-19       # elementary charge, C
KB = 1.380649e-23          # Boltzmann constant, J / K

POLARITY_ANTIPARALLEL = +1   # m pushed anti-parallel to the polarizer
POLARITY_PARALLEL = -1       # m pushed parallel to the polarizer (Chen et al. 2014)
