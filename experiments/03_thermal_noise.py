"""
Experiment 3 - thermal noise (Brown field, stochastic Heun integration).

(a) Validation: a static magnet (I = 0) in a 100 mT field along its easy axis x must satisfy
    equipartition, <m_y^2> = kB T / (mu0 Ms V (H + H_k)) and
    <m_z^2> = kB T / (mu0 Ms V (H + H_k + Nz Ms)).
(b) Linewidth of a single oscillator at 300 K (from phase diffusion).
(c) Phase jitter of a dipolar-locked stacked pair at 300 K, compared with the
    equipartition estimate of Chen et al. 2014, App. E (~0.1 rad).

Run:  python experiments/03_thermal_noise.py [--fast] [--show]
"""
import numpy as np

from _common import parse_args, time_scale, Report, RESULTS_DIR

args = parse_args(__doc__)
import matplotlib.pyplot as plt  # noqa: E402

from stno import simulate, default_params, POLARITY_PARALLEL  # noqa: E402
from stno.constants import MU0, KB  # noqa: E402
from stno.params import free_layer_volume, distance_for_coupling  # noqa: E402
from stno.analysis import inplane_phase, phase_linewidth, locking_metrics  # noqa: E402

ts = time_scale(args)
report = Report("03_thermal_noise")
T_K = 300.0

# ---------------------------------------------------------------- (a) equipartition
H_x = 0.1 / MU0                                     # 100 mT along x (A/m)
params = default_params(temperature=T_K, H_applied=np.array([H_x, 0.0, 0.0]))
V = free_layer_volume(params)
t, m = simulate(n_osc=1, current=0.0, duration=20e-9 * ts, params=params, m0=[1, 0, 0], seed=5)
i0 = len(t) // 10
my2, mz2 = (m[i0:, 0, 1] ** 2).mean(), (m[i0:, 0, 2] ** 2).mean()
H_k = params['k_inplane'] * params['Ms']
my2_theory = KB * T_K / (MU0 * params['Ms'] * V * (H_x + H_k))
mz2_theory = KB * T_K / (MU0 * params['Ms'] * V * (H_x + H_k + params['Nz'] * params['Ms']))
report(f"(a) equipartition at {T_K:.0f} K: <my^2> sim/theory = {my2 / my2_theory:.3f}, "
       f"<mz^2> sim/theory = {mz2 / mz2_theory:.3f}  (expected ~1)")

# ---------------------------------------------------------------- (b) single-oscillator linewidth
params = default_params(temperature=T_K)
t, m = simulate(n_osc=1, current=1e-3, polarity=POLARITY_PARALLEL, duration=100e-9 * ts,
                params=params, seed=6, n_save=50000)
phase = inplane_phase(m)[:, 0]
linewidth = phase_linewidth(t, phase)
report(f"(b) single oscillator, 1 mA, {T_K:.0f} K: linewidth (FWHM) ~ {linewidth / 1e6:.0f} MHz")

# ---------------------------------------------------------------- (c) locked pair jitter
spacing = distance_for_coupling(0.01, free_layer_volume(params))
positions = np.array([[0, 0, 0], [0, 0, spacing]])
t2, m2 = simulate(n_osc=2, positions=positions, current=1e-3, polarity=POLARITY_PARALLEL,
                  duration=60e-9 * ts, params=params, seed=7, n_save=30000)
phases = inplane_phase(m2)
metrics = locking_metrics(t2, phases, 0, 1)
i0 = len(t2) // 2
deviation = np.angle(np.exp(1j * (phases[i0:, 0] - phases[i0:, 1] - np.pi)))
report(f"(c) stacked pair, d = 0.01, {T_K:.0f} K: R = {metrics['R']:.3f}, "
       f"std(dphi - pi) = {deviation.std():.3f} rad (Chen App. E estimate ~0.1 rad)")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
residual = phase - np.polyval(np.polyfit(t, phase, 1), t)
axes[0].plot(t * 1e9, residual)
axes[0].set_xlabel('time (ns)')
axes[0].set_ylabel('phase - mean drift (rad)')
axes[0].set_title(f'single oscillator at {T_K:.0f} K (phase diffusion)')
axes[1].hist(deviation, bins=60, density=True)
axes[1].set_xlabel('phase difference - pi (rad)')
axes[1].set_ylabel('probability density')
axes[1].set_title(f'locked stacked pair at {T_K:.0f} K')
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/03_thermal_noise.png", dpi=150)
report.save()
if args.show:
    plt.show()
