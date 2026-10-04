"""
Analytic references for a single PERP-STNO in the large easy-plane limit (Slonczewski torque).

Energy balance
--------------
In the steady state the magnetization precesses about z on a cone of constant |m_z|, with
frequency f = f_M |m_z| (f_M = gamma mu0 Ms / 2 pi). The cone is fixed by the balance of
Gilbert damping and spin-transfer torque:

    F(m_z) = alpha Ms m_z - a_J g(s m_z) = 0,     g(x) = 1 / (-4 + C(P) (3 + x))

with s = -polarity (so m.s = s |m_z|). For polarity +1, F is non-monotonic: the lower root
is the stable precession and the two roots merge at a fold (upper critical current).
For polarity -1, F is monotonic and precession persists until m_z -> 1.
"""
import numpy as np
from scipy.optimize import brentq

from .params import (default_params, spin_torque_field_per_amp, slonczewski_C,
                     saturation_frequency)


def slonczewski_g(m_dot_s, spin_pol):
    """Slonczewski angular efficiency g(m.s) = 1 / (-4 + C(P) (3 + m.s))."""
    return 1.0 / (-4.0 + slonczewski_C(spin_pol) * (3.0 + m_dot_s))


def energy_balance(current, spin_pol, polarity, params=None, n_grid=4000):
    """Stable steady-state frequency (Hz) and |m_z| from the energy balance.

    Returns (nan, nan) if no precessional state exists (static state).
    """
    params = default_params() if params is None else params
    s = -np.sign(polarity)
    aJ = abs(current) * spin_torque_field_per_amp(params)

    def F(mz):
        return params['alpha'] * params['Ms'] * mz - aJ * slonczewski_g(s * mz, spin_pol)

    grid = np.linspace(1e-4, 0.9999, n_grid)
    values = F(grid)
    sign_changes = np.where(np.sign(values[:-1]) != np.sign(values[1:]))[0]
    if len(sign_changes) == 0:
        return np.nan, np.nan
    i = sign_changes[0]                              # lowest root = stable branch
    mz = brentq(F, grid[i], grid[i + 1])
    return saturation_frequency(params) * mz, mz


def upper_critical_current(spin_pol, polarity, params=None):
    """Current (A) above which precession stops, with the corresponding |m_z| and f (Hz).

    polarity +1: fold of F (F = F' = 0) at m_z* = (3 - 4/C) / 2.
    polarity -1: m_z reaches 1, i.e. alpha Ms = a_J g(+1).
    """
    params = default_params() if params is None else params
    C = float(slonczewski_C(spin_pol))
    aJ_per_amp = spin_torque_field_per_amp(params)
    if polarity > 0:
        mz_star = (3.0 - 4.0 / C) / 2.0
        if not 0.0 < mz_star < 1.0:
            return np.nan, np.nan, np.nan
        aJ_crit = params['alpha'] * params['Ms'] * mz_star * (C * (3.0 - mz_star) - 4.0)
        return aJ_crit / aJ_per_amp, mz_star, saturation_frequency(params) * mz_star
    aJ_crit = params['alpha'] * params['Ms'] * (4.0 * C - 4.0)
    return aJ_crit / aJ_per_amp, 1.0, saturation_frequency(params)


def pendulum_thresholds(spin_pol, params=None):
    """Low-current thresholds of the phase equation of Chen et al. 2014, Eq. (9):

        phi'' + alpha phi' = -(u + (k/2) sin 2 phi)      (scaled time)

    This is an underdamped pendulum, so it is bistable:
      * the static state is destabilized when u_eff > k/2 (onset, sweeping up from rest);
      * a running precession is retrapped when u_eff < (2/pi) alpha sqrt(k).
    u_eff = a_J g(0) / Ms. Returns (onset current, retrapping current) in A.
    """
    params = default_params() if params is None else params
    u_eff_per_amp = spin_torque_field_per_amp(params) * float(slonczewski_g(0.0, spin_pol)) / params['Ms']
    k = params['k_inplane']
    onset = (k / 2.0) / u_eff_per_amp
    retrap = (2.0 / np.pi) * params['alpha'] * np.sqrt(k) / u_eff_per_amp
    return onset, retrap
