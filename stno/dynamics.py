"""
Macrospin dynamics of N dipolar-coupled spin-torque nano-oscillators.

Equation of motion (Landau-Lifshitz-Gilbert-Slonczewski, written in explicit LL form)
------------------------------------------------------------------------------------
For each oscillator i with unit magnetization m_i:

    dm/dt = -c [ m x H + alpha m x (m x H) ]  +  c a [ m x (m x s) - alpha m x s ]

    c = gamma mu0 / (1 + alpha^2),     a = a_J(t) * g(m.s)   (A/m)

Effective field H_i (A/m):
    H_x = k Ms m_x + H_applied,x + H_ac,x(t) + H_thermal,x + sum_j H_dip(j -> i)
    H_y =             H_applied,y + H_ac,y(t) + H_thermal,y + sum_j H_dip(j -> i)
    H_z = -N_eff Ms m_z + H_applied,z + H_ac,z(t) + H_thermal,z + sum_j H_dip(j -> i)

Point-dipole field of oscillator j at oscillator i (e = unit vector from j to i, r = distance):
    H_dip = Ms V_j / (4 pi r^3) [ 3 (m_j . e) e - m_j ]

Thermal noise: Brown field with <H_a(t) H_b(t')> = 2 alpha kB T / (gamma mu0^2 Ms V) delta_ab delta(t-t').

Integration: classical RK4 at T = 0; stochastic Heun (converges to the Stratonovich solution)
at T > 0. |m| is renormalized after every step.
"""
import math

import numpy as np
from numba import njit

from .constants import GAMMA, MU0, KB, POLARITY_PARALLEL
from .params import (default_params, cross_section_area, spin_torque_field_per_amp,
                     effective_demag_factor, slonczewski_C)


@njit(cache=True)
def _llgs_rhs(m, t, pos, dip_factor, volumes, aJ_dc, aJ_ac, f_current_ac, phase_current_ac,
              C_P, spin_pol, torque_form, Lam, fieldlike_ratio, s_vec,
              H_k, N_eff, Ms, H_applied, H_ac, f_field_ac, H_thermal, alpha, coupling_on):
    """Time derivative dm/dt for all oscillators.

    Array arguments (N = number of oscillators):
      m (N,3)            unit magnetizations
      pos (N,3)          positions (m)
      dip_factor (N,N)   multiplicative correction of the point-dipole field (1 = pure dipole)
      volumes (N,)       free-layer volumes (m^3)
      aJ_dc, aJ_ac (N,)  DC and AC spin-torque prefactors incl. polarity sign (A/m)
      phase_current_ac   phase of the AC current per oscillator (rad)
      C_P, spin_pol (N,) Slonczewski C(P) and spin polarization P
      s_vec (3,)         polarizer direction
      H_applied, H_ac(3,) static and AC applied fields (A/m)
      H_thermal (N,3)    thermal field for the current step (A/m)
    Scalars: f_current_ac, f_field_ac (Hz); torque_form (0 Slonczewski, 1 Lambda);
             Lam; fieldlike_ratio; H_k = k Ms (A/m); N_eff; Ms (A/m); alpha; coupling_on.
    """
    N = m.shape[0]
    dm = np.zeros_like(m)
    ac_field_factor = math.sin(2.0 * math.pi * f_field_ac * t)
    c = GAMMA * MU0 / (1.0 + alpha * alpha)
    L2 = Lam * Lam
    sx = s_vec[0]
    sy = s_vec[1]
    sz = s_vec[2]
    for i in range(N):
        mx = m[i, 0]
        my = m[i, 1]
        mz = m[i, 2]

        # --- effective field: anisotropy (x), demagnetization (z), applied, AC, thermal ---
        Hx = H_k * mx + H_applied[0] + H_ac[0] * ac_field_factor + H_thermal[i, 0]
        Hy = H_applied[1] + H_ac[1] * ac_field_factor + H_thermal[i, 1]
        Hz = -N_eff * Ms * mz + H_applied[2] + H_ac[2] * ac_field_factor + H_thermal[i, 2]

        # --- dipolar stray fields of all other oscillators (point dipoles) ---
        if coupling_on:
            for j in range(N):
                if j == i:
                    continue
                rx = pos[i, 0] - pos[j, 0]
                ry = pos[i, 1] - pos[j, 1]
                rz = pos[i, 2] - pos[j, 2]
                r = math.sqrt(rx * rx + ry * ry + rz * rz)
                ex = rx / r
                ey = ry / r
                ez = rz / r
                m_dot_e = m[j, 0] * ex + m[j, 1] * ey + m[j, 2] * ez
                prefactor = Ms * volumes[j] / (4.0 * math.pi * r * r * r) * dip_factor[i, j]
                Hx += prefactor * (3.0 * m_dot_e * ex - m[j, 0])
                Hy += prefactor * (3.0 * m_dot_e * ey - m[j, 1])
                Hz += prefactor * (3.0 * m_dot_e * ez - m[j, 2])

        # --- spin-transfer torque: angular efficiency g(m.s) ---
        m_dot_s = mx * sx + my * sy + mz * sz
        if torque_form == 0:
            g = 1.0 / (-4.0 + C_P[i] * (3.0 + m_dot_s))
        else:
            g = spin_pol[i] * L2 / ((L2 + 1.0) + (L2 - 1.0) * m_dot_s)
        aJ = aJ_dc[i] + aJ_ac[i] * math.sin(2.0 * math.pi * f_current_ac * t + phase_current_ac[i])
        a = aJ * g                                     # damping-like torque amplitude (A/m)

        # field-like torque b_J m x s, implemented as an effective field along s
        Hx += fieldlike_ratio * a * sx
        Hy += fieldlike_ratio * a * sy
        Hz += fieldlike_ratio * a * sz

        # --- cross products ---
        px = my * Hz - mz * Hy                         # precession:  m x H
        py = mz * Hx - mx * Hz
        pz = mx * Hy - my * Hx
        dx = my * pz - mz * py                         # damping:     m x (m x H)
        dy = mz * px - mx * pz
        dz = mx * py - my * px
        ux = my * sz - mz * sy                         #              m x s
        uy = mz * sx - mx * sz
        uz = mx * sy - my * sx
        tx = my * uz - mz * uy                         # spin torque: m x (m x s)
        ty = mz * ux - mx * uz
        tz = mx * uy - my * ux

        dm[i, 0] = -c * (px + alpha * dx) + c * a * (tx - alpha * ux)
        dm[i, 1] = -c * (py + alpha * dy) + c * a * (ty - alpha * uy)
        dm[i, 2] = -c * (pz + alpha * dz) + c * a * (tz - alpha * uz)
    return dm


@njit(cache=True)
def _integrate(m0, pos, dip_factor, volumes, aJ_dc, aJ_ac, f_current_ac, phase_current_ac,
               C_P, spin_pol, torque_form, Lam, fieldlike_ratio, s_vec, H_k, N_eff, Ms,
               H_applied, H_ac, f_field_ac, alpha, coupling_on, dt, n_steps, n_save,
               thermal_sigma, seed):
    """Integrate the LLGS equation; returns m sampled at n_save evenly spaced steps."""
    np.random.seed(seed)
    N = m0.shape[0]
    m = m0.copy()
    every = max(1, n_steps // n_save)
    out = np.zeros((n_save, N, 3))
    H_thermal = np.zeros((N, 3))
    noisy = False
    for i in range(N):
        if thermal_sigma[i] > 0.0:
            noisy = True
    k_save = 0
    t = 0.0
    for step in range(n_steps):
        if noisy:
            # stochastic Heun: the same random field is used in predictor and corrector
            for i in range(N):
                for q in range(3):
                    H_thermal[i, q] = thermal_sigma[i] * np.random.normal()
            k1 = _llgs_rhs(m, t, pos, dip_factor, volumes, aJ_dc, aJ_ac, f_current_ac,
                           phase_current_ac, C_P, spin_pol, torque_form, Lam, fieldlike_ratio,
                           s_vec, H_k, N_eff, Ms, H_applied, H_ac, f_field_ac, H_thermal,
                           alpha, coupling_on)
            m_pred = m + dt * k1
            k2 = _llgs_rhs(m_pred, t + dt, pos, dip_factor, volumes, aJ_dc, aJ_ac, f_current_ac,
                           phase_current_ac, C_P, spin_pol, torque_form, Lam, fieldlike_ratio,
                           s_vec, H_k, N_eff, Ms, H_applied, H_ac, f_field_ac, H_thermal,
                           alpha, coupling_on)
            m = m + 0.5 * dt * (k1 + k2)
        else:
            # classical fourth-order Runge-Kutta
            k1 = _llgs_rhs(m, t, pos, dip_factor, volumes, aJ_dc, aJ_ac, f_current_ac,
                           phase_current_ac, C_P, spin_pol, torque_form, Lam, fieldlike_ratio,
                           s_vec, H_k, N_eff, Ms, H_applied, H_ac, f_field_ac, H_thermal,
                           alpha, coupling_on)
            k2 = _llgs_rhs(m + 0.5 * dt * k1, t + 0.5 * dt, pos, dip_factor, volumes, aJ_dc,
                           aJ_ac, f_current_ac, phase_current_ac, C_P, spin_pol, torque_form, Lam,
                           fieldlike_ratio, s_vec, H_k, N_eff, Ms, H_applied, H_ac, f_field_ac,
                           H_thermal, alpha, coupling_on)
            k3 = _llgs_rhs(m + 0.5 * dt * k2, t + 0.5 * dt, pos, dip_factor, volumes, aJ_dc,
                           aJ_ac, f_current_ac, phase_current_ac, C_P, spin_pol, torque_form, Lam,
                           fieldlike_ratio, s_vec, H_k, N_eff, Ms, H_applied, H_ac, f_field_ac,
                           H_thermal, alpha, coupling_on)
            k4 = _llgs_rhs(m + dt * k3, t + dt, pos, dip_factor, volumes, aJ_dc, aJ_ac,
                           f_current_ac, phase_current_ac, C_P, spin_pol, torque_form, Lam,
                           fieldlike_ratio, s_vec, H_k, N_eff, Ms, H_applied, H_ac, f_field_ac,
                           H_thermal, alpha, coupling_on)
            m = m + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        for i in range(N):                              # keep |m| = 1
            norm = math.sqrt(m[i, 0] ** 2 + m[i, 1] ** 2 + m[i, 2] ** 2)
            m[i, 0] /= norm
            m[i, 1] /= norm
            m[i, 2] /= norm
        t += dt
        if step % every == 0 and k_save < n_save:
            out[k_save] = m
            k_save += 1
    return out


def dipole_correction_factors(positions, params):
    """(N,N) multiplicative factors applied to the point-dipole field (1 unless disk_correction)."""
    N = len(positions)
    factors = np.ones((N, N))
    if params['disk_correction']:
        for i in range(N):
            for j in range(N):
                if i != j:
                    r = np.linalg.norm(positions[i] - positions[j])
                    factors[i, j] = (1.0 + (params['disk_radius'] / r) ** 2) ** -1.5
    return factors


def thermal_field_sigma(params, volumes, dt):
    """Standard deviation (A/m) of each Cartesian component of the Brown thermal field per step."""
    if params['temperature'] <= 0:
        return np.zeros(len(volumes))
    return np.sqrt(2.0 * params['alpha'] * KB * params['temperature']
                   / (GAMMA * MU0 ** 2 * params['Ms'] * volumes * dt))


def _per_oscillator(value, n):
    """Broadcast a scalar or sequence to a float array of length n."""
    return np.broadcast_to(np.asarray(value, dtype=float), (n,)).copy()


def simulate(n_osc=1, positions=None, current=1e-3, spin_pol=0.38, polarity=POLARITY_PARALLEL,
             duration=50e-9, dt=0.1e-12, n_save=20000, coupling=True, params=None, seed=0,
             m0=None, current_ac=0.0, f_current_ac=0.0, phase_current_ac=0.0,
             H_ac=None, f_field_ac=0.0, sizes=None):
    """Simulate n_osc coupled oscillators.

    Parameters
    ----------
    positions        (n_osc, 3) array of positions (m); default: all at the origin
    current          DC current (A), scalar or per oscillator
    spin_pol         spin polarization P, scalar or per oscillator
    polarity         +1 / -1 (see stno.constants), scalar or per oscillator
    duration, dt     simulated time and time step (s)
    n_save           number of saved samples
    coupling         include dipolar coupling between oscillators
    seed             seed for the random initial phases and the thermal noise
    m0               (n_osc, 3) initial magnetizations; default: near the film plane, random phase
    current_ac       AC current amplitude (A), scalar or per oscillator, at f_current_ac (Hz)
    H_ac             AC field amplitude vector (A/m), common to all oscillators, at f_field_ac (Hz)
    sizes            optional (n_osc, 2) per-oscillator lateral sizes (a, b) in m

    Returns
    -------
    t (n_save,) times in s, and m (n_save, n_osc, 3) magnetization trajectories.
    """
    params = default_params() if params is None else params
    N = n_osc
    I_dc = _per_oscillator(current, N)
    P = _per_oscillator(spin_pol, N)
    pol_sign = _per_oscillator(polarity, N)
    I_ac = _per_oscillator(current_ac, N)
    phase_ac = _per_oscillator(phase_current_ac, N)

    if sizes is None:
        sizes = np.tile([params['size_a'], params['size_b']], (N, 1))
    areas = np.array([cross_section_area(params, a, b) for a, b in sizes])
    volumes = areas * params['thickness']
    aJ_per_amp = np.array([spin_torque_field_per_amp(params, area) for area in areas])
    aJ_dc = pol_sign * I_dc * aJ_per_amp
    aJ_ac = pol_sign * I_ac * aJ_per_amp

    positions = np.zeros((N, 3)) if positions is None else np.asarray(positions, dtype=float)
    dip_factor = dipole_correction_factors(positions, params)

    if m0 is None:
        rng = np.random.default_rng(seed)
        phi = rng.uniform(0, 2 * np.pi, N)
        m0 = np.stack([0.99 * np.cos(phi), 0.99 * np.sin(phi), np.full(N, 0.141)], axis=1)
    m0 = np.array(m0, dtype=float).reshape(N, 3)
    m0 /= np.linalg.norm(m0, axis=1)[:, None]

    sigma = thermal_field_sigma(params, volumes, dt)
    H_ac = np.zeros(3) if H_ac is None else np.asarray(H_ac, dtype=float)
    n_steps = max(1, int(round(duration / dt)))
    n_save = min(n_save, n_steps)
    torque_form = 0 if params['torque_form'] == 'slonczewski' else 1

    m = _integrate(m0, positions, dip_factor, volumes, aJ_dc, aJ_ac, float(f_current_ac),
                   phase_ac, slonczewski_C(P), P, torque_form, float(params['Lambda']),
                   float(params['fieldlike_ratio']), np.asarray(params['polarizer_dir'], dtype=float),
                   float(params['k_inplane'] * params['Ms']), float(effective_demag_factor(params)),
                   float(params['Ms']), np.asarray(params['H_applied'], dtype=float), H_ac,
                   float(f_field_ac), float(params['alpha']), bool(coupling), dt, n_steps, n_save,
                   sigma, int(seed))
    every = max(1, n_steps // n_save)
    t = (np.arange(n_save) * every + 1) * dt
    return t, m
