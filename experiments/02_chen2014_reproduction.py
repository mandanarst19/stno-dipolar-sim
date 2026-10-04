"""
Experiment 2 - reproduction of Chen et al., J. Appl. Phys. 115, 134306 (2014).

Four vertically stacked PERP-STNOs with spin polarizations P = 0.36, 0.38, 0.40, 0.42
(i.e. four different natural frequencies), scaled dipolar coupling d = V/(4 pi r^3) = 0.01,
current 1 mA, polarity -1 (Chen's branch), elliptical cross-section.

(a) Spectra without and with dipolar coupling (Chen Fig. 3b): four separate peaks become
    one common peak (Chen reports 10.32 GHz), with anti-phase order between neighbours.
    Shown for the summed readout and for alternating reference layers (Chen Fig. 1d).
(b) Frequency spread f4 - f1 vs current with and without coupling (Chen Fig. 4).

Run:  python experiments/02_chen2014_reproduction.py [--fast] [--show]
"""
import numpy as np

from _common import parse_args, time_scale, Report, RESULTS_DIR

args = parse_args(__doc__)
import matplotlib.pyplot as plt  # noqa: E402

from stno import simulate, default_params, POLARITY_PARALLEL  # noqa: E402
from stno.params import free_layer_volume, distance_for_coupling  # noqa: E402
from stno.analysis import (oscillator_state, inplane_phase, locking_metrics, power_spectrum,  # noqa: E402
                           readout_signals, cancellation_factor)
from stno.theory import energy_balance  # noqa: E402

ts = time_scale(args)
report = Report("02_chen2014_reproduction")
params = default_params()
SPIN_POLS = [0.36, 0.38, 0.40, 0.42]
N = len(SPIN_POLS)
CHEN_LOCKED_GHZ = 10.32

# stack along z with nearest-neighbour coupling d = 0.01
spacing = distance_for_coupling(0.01, free_layer_volume(params))
positions = np.zeros((N, 3))
positions[:, 2] = np.arange(N) * spacing
f_theory = [energy_balance(1e-3, P, POLARITY_PARALLEL, params)[0] / 1e9 for P in SPIN_POLS]
report(f"Stack spacing r = {spacing * 1e9:.1f} nm (d = 0.01); analytic free frequencies "
       f"{np.round(f_theory, 3)} GHz")

# ---------------------------------------------------------------- (a) spectra
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for coupled, color in [(False, 'tab:blue'), (True, 'tab:red')]:
    t, m = simulate(n_osc=N, positions=positions, current=1e-3, spin_pol=SPIN_POLS,
                    polarity=POLARITY_PARALLEL, duration=60e-9 * ts, coupling=coupled,
                    seed=3, params=params)
    freqs = np.abs(oscillator_state(t, m)['freq'])
    phase = inplane_phase(m)
    cos_nn = [locking_metrics(t, phase, i, i + 1)['cos'] for i in range(N - 1)]
    summed, alternating = readout_signals(m)
    label = 'coupled' if coupled else 'uncoupled'
    for ax, signal, title in [(axes[0], summed, 'summed readout'),
                              (axes[1], alternating, 'alternating reference layers')]:
        f, power = power_spectrum(t, signal)
        ax.semilogy(f / 1e9, power + 1e-30, color=color, label=label)
        ax.set_title(title)
    report(f"{label:9s}: f = {[round(float(x) / 1e9, 3) for x in freqs]} GHz; "
           f"<cos(phi_i - phi_i+1)> = {[round(c, 3) for c in cos_nn]}; "
           f"readout coherence summed / alternating = {cancellation_factor(m, summed):.3f} / "
           f"{cancellation_factor(m, alternating):.3f}")
    if coupled:
        f_locked = float(np.mean(freqs)) / 1e9
        report(f"-> locked frequency {f_locked:.3f} GHz vs Chen et al. {CHEN_LOCKED_GHZ} GHz "
               f"({100 * abs(f_locked - CHEN_LOCKED_GHZ) / CHEN_LOCKED_GHZ:.2f} % difference)")
for ax in axes:
    top = max(line.get_ydata().max() for line in ax.get_lines())
    ax.set_ylim(top * 1e-10, top * 5)               # show 10 decades below the highest peak
    ax.set_xlim(8.5, 12.5)
    ax.set_xlabel('frequency (GHz)')
    ax.set_ylabel('power (arb. units)')
    ax.legend()
fig.suptitle('Four stacked PERP-STNOs, d = 0.01, I = 1 mA (cf. Chen et al. 2014, Fig. 3b)')
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/02a_chen2014_spectra.png", dpi=150)

# ---------------------------------------------------------------- (b) f4 - f1 vs current
currents = np.linspace(0.6e-3, 5.0e-3, 10 if not args.fast else 3)
spread = {}
for coupled in (False, True):
    values = []
    for I in currents:
        t, m = simulate(n_osc=N, positions=positions, current=I, spin_pol=SPIN_POLS,
                        polarity=POLARITY_PARALLEL, duration=40e-9 * ts, coupling=coupled,
                        seed=3, params=params)
        st = oscillator_state(t, m)
        f = np.abs(st['freq'])
        values.append(f[-1] - f[0] if st['precessing'].all() else np.nan)
    spread[coupled] = np.array(values)
    report(f"f4 - f1 (GHz), coupled={coupled}: {[round(float(x) / 1e9, 3) + 0.0 for x in spread[coupled]]}")
report(f"currents (mA): {[round(float(x) * 1e3, 2) for x in currents]}")
report("(nan = at least one oscillator has stopped precessing: m_z -> 1 above ~3.6-4.4 mA)")

plt.figure(figsize=(5.5, 3.8))
plt.plot(currents * 1e3, spread[False] / 1e9, '^-', label='uncoupled (d = 0)')
plt.plot(currents * 1e3, spread[True] / 1e9, 's-', label='coupled (d = 0.01)')
plt.xlabel('current (mA)')
plt.ylabel('f4 - f1 (GHz)')
plt.title('Frequency spread (cf. Chen et al. 2014, Fig. 4)')
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/02b_chen2014_spread.png", dpi=150)
report.save()
if args.show:
    plt.show()
