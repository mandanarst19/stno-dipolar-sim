"""
Device and material parameters, and quantities derived from them.

All quantities are in SI units. The default parameter set reproduces the
perpendicular-polarizer STNO (PERP-STNO) with a Co free layer of
Chen et al., J. Appl. Phys. 115, 134306 (2014).
"""
import numpy as np

from .constants import GAMMA, MU0, HBAR, QE


def default_params(**overrides):
    """Return the default parameter dictionary (Chen et al. 2014), optionally overridden.

    Keys
    ----
    Ms               saturation magnetization of the free layer (A/m)
    alpha            Gilbert damping (dimensionless)
    thickness        free-layer thickness (m)
    size_a, size_b   lateral dimensions of the pillar (m)
    cross_section    'ellipse' (area = pi a b / 4) or 'rect' (area = a b).
                     'ellipse' reproduces Chen's spin-torque strength u' = 0.0354 at 1 mA.
    k_inplane        in-plane uniaxial anisotropy along x, in units of Ms (H_k = k Ms)
    Nz               thin-film demagnetizing factor
    K_perp           perpendicular (interface) anisotropy (J/m^3); reduces the effective Nz
    H_applied        applied static field vector (A/m)
    polarizer_dir    unit vector s of the fixed-layer spin polarization
    torque_form      'slonczewski': g = 1 / (-4 + C(P) (3 + m.s))      (Chen Eq. 1)
                     'lambda':      g = P L^2 / ((L^2+1) + (L^2-1) m.s) (Csaba 2012)
    Lambda           spin-torque asymmetry parameter for torque_form='lambda'
    fieldlike_ratio  field-like torque b_J / a_J (0 = pure damping-like torque, GMR case)
    disk_correction  if True, multiply the point-dipole field by (1 + R^2/r^2)^-3/2
                     (finite-disk correction, Chen App. A; derived for vertically stacked disks)
    disk_radius      radius R used in the finite-disk correction (m)
    temperature      temperature for the thermal (Brown) field (K); 0 = deterministic
    """
    params = dict(
        Ms=8.66e5,
        alpha=0.02,
        thickness=3e-9,
        size_a=60e-9,
        size_b=70e-9,
        cross_section='ellipse',
        k_inplane=0.008,
        Nz=1.0,
        K_perp=0.0,
        H_applied=np.zeros(3),
        polarizer_dir=np.array([0.0, 0.0, 1.0]),
        torque_form='slonczewski',
        Lambda=1.5,
        fieldlike_ratio=0.0,
        disk_correction=False,
        disk_radius=30e-9,
        temperature=0.0,
    )
    params.update(overrides)
    return params


def cross_section_area(params, size_a=None, size_b=None):
    """Area of the pillar cross-section (m^2)."""
    a = params['size_a'] if size_a is None else size_a
    b = params['size_b'] if size_b is None else size_b
    if params['cross_section'] == 'ellipse':
        return np.pi * a * b / 4.0
    return a * b


def free_layer_volume(params):
    """Free-layer volume V = area x thickness (m^3)."""
    return cross_section_area(params) * params['thickness']


def spin_torque_field_per_amp(params, area=None):
    """Spin-torque prefactor a_J per ampere of current (A/m per A).

    a_J = hbar J / (2 e mu0 Ms d), with J = I / area the current density.
    The angular efficiency g(m.s) is applied separately in the dynamics.
    """
    area = cross_section_area(params) if area is None else area
    return HBAR / (2.0 * QE * MU0 * params['Ms'] * params['thickness'] * area)


def effective_demag_factor(params):
    """Effective out-of-plane demagnetizing factor including perpendicular anisotropy.

    N_eff = Nz - 2 K_perp / (mu0 Ms^2). A negative value means net perpendicular anisotropy.
    """
    return params['Nz'] - 2.0 * params['K_perp'] / (MU0 * params['Ms'] ** 2)


def slonczewski_C(spin_polarization):
    """Slonczewski coefficient C(P) = (1 + P)^3 / (4 P^(3/2))."""
    P = np.asarray(spin_polarization, dtype=float)
    return (1.0 + P) ** 3 / (4.0 * P ** 1.5)


def saturation_frequency(params):
    """f_M = gamma mu0 Ms / (2 pi) (Hz). In the large easy-plane limit f = f_M |m_z|."""
    return GAMMA * MU0 * params['Ms'] / (2.0 * np.pi)


def dipolar_coupling_strength(volume, distance):
    """Scaled point-dipole coupling d(r) = V / (4 pi r^3) (Chen et al. 2014, Eq. 5)."""
    return volume / (4.0 * np.pi * distance ** 3)


def distance_for_coupling(coupling, volume):
    """Centre-to-centre distance r that gives the scaled coupling d(r) = coupling."""
    return (volume / (4.0 * np.pi * coupling)) ** (1.0 / 3.0)
