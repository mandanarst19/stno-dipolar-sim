"""Fast regression tests (about 1 minute including Numba compilation). Run: pytest -q"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stno import simulate, default_params, POLARITY_PARALLEL  # noqa: E402
from stno.constants import MU0, KB  # noqa: E402
from stno.params import free_layer_volume, distance_for_coupling  # noqa: E402
from stno.analysis import oscillator_state  # noqa: E402
from stno.theory import energy_balance  # noqa: E402


def test_single_oscillator_matches_energy_balance():
    params = default_params()
    t, m = simulate(n_osc=1, current=1e-3, polarity=POLARITY_PARALLEL, duration=30e-9, params=params)
    f_sim = abs(oscillator_state(t, m)['freq'][0])
    f_theory, _ = energy_balance(1e-3, 0.38, POLARITY_PARALLEL, params)
    assert abs(f_sim - f_theory) / f_theory < 0.02


def test_chen2014_locked_frequency():
    params = default_params()
    positions = np.zeros((4, 3))
    positions[:, 2] = np.arange(4) * distance_for_coupling(0.01, free_layer_volume(params))
    t, m = simulate(n_osc=4, positions=positions, current=1e-3, spin_pol=[0.36, 0.38, 0.40, 0.42],
                    polarity=POLARITY_PARALLEL, duration=40e-9, seed=3, params=params)
    freqs = np.abs(oscillator_state(t, m)['freq'])
    assert np.ptp(freqs) < 1e6                       # all four oscillators share one frequency
    assert abs(freqs.mean() - 10.304e9) < 30e6        # Chen et al. 2014 report 10.32 GHz


def test_thermal_equipartition():
    H_x = 0.1 / MU0
    params = default_params(temperature=300.0, H_applied=np.array([H_x, 0.0, 0.0]))
    t, m = simulate(n_osc=1, current=0.0, duration=20e-9, params=params, m0=[1, 0, 0], seed=5)
    my2 = (m[len(t) // 10:, 0, 1] ** 2).mean()
    theory = KB * 300.0 / (MU0 * params['Ms'] * free_layer_volume(params)
                           * (H_x + params['k_inplane'] * params['Ms']))
    assert 0.85 < my2 / theory < 1.15
