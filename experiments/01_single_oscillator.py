"""
Experiment 1 - single PERP-STNO: validation against analytic theory.

(a) Frequency and |m_z| vs current for both current polarities, compared with the
    energy-balance solution (stno.theory.energy_balance) and the upper critical currents.
(b) Hysteresis: sweeping the current up from rest and back down, compared with the onset and
    retrapping thresholds of the pendulum-like phase equation of Chen et al. 2014, Eq. (9).

Run:  python experiments/01_single_oscillator.py [--fast] [--show]
"""
import numpy as np

from _common import parse_args, time_scale, Report, RESULTS_DIR

args = parse_args(__doc__)
import matplotlib.pyplot as plt  # noqa: E402  (backend selected in parse_args)

from stno import simulate, default_params, POLARITY_ANTIPARALLEL, POLARITY_PARALLEL  # noqa: E402
from stno.analysis import oscillator_state  # noqa: E402
from stno.theory import energy_balance, upper_critical_current, pendulum_thresholds  # noqa: E402

ts = time_scale(args)
report = Report("01_single_oscillator")
params = default_params()
SPIN_POL = 0.38

branches = [
    (POLARITY_ANTIPARALLEL, "polarity +1 (m pushed antiparallel)", 1.5e-3, "tab:blue"),
    (POLARITY_PARALLEL, "polarity -1 (Chen et al. 2014 branch)", 5.0e-3, "tab:red"),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

# ---------------------------------------------------------------- (a) f(I) vs analytic
for polarity, label, I_max, color in branches:
    I_up, mz_up, f_up = upper_critical_current(SPIN_POL, polarity, params)
    report(f"{label}: upper critical current {I_up * 1e3:.3f} mA (|mz| = {mz_up:.3f}, f = {f_up / 1e9:.2f} GHz)")
    report("   I (mA)   f_sim (GHz)   f_theory (GHz)   |mz|_sim   |mz|_theory")
    currents = np.linspace(0.3e-3, I_max, 16 if not args.fast else 5)
    rows = []
    for I in currents:
        t, m = simulate(n_osc=1, current=I, spin_pol=SPIN_POL, polarity=polarity,
                        duration=40e-9 * ts, params=params)
        st = oscillator_state(t, m)
        f_sim = abs(st['freq'][0]) if st['precessing'][0] else np.nan
        f_th, mz_th = energy_balance(I, SPIN_POL, polarity, params)
        rows.append((I, f_sim, f_th, abs(st['mz'][0]), mz_th))
        report(f"   {I * 1e3:6.2f}   {f_sim / 1e9:11.2f}   {f_th / 1e9:14.2f}   {abs(st['mz'][0]):8.3f}   {mz_th:11.3f}")
    rows = np.array(rows)
    both = np.isfinite(rows[:, 1]) & np.isfinite(rows[:, 2])
    if both.any():
        err = np.abs(rows[both, 1] - rows[both, 2]) / rows[both, 2]
        report(f"   -> max relative frequency error: {100 * err.max():.2f} % over {both.sum()} points\n")

    I_dense = np.linspace(0.2e-3, I_max, 300)
    theory = np.array([energy_balance(I, SPIN_POL, polarity, params) for I in I_dense])
    axes[0].plot(rows[:, 0] * 1e3, rows[:, 1] / 1e9, 'o', color=color, label=f'{label}: sim')
    axes[0].plot(I_dense * 1e3, theory[:, 0] / 1e9, '-', color=color, lw=1, label='energy balance')
    axes[1].plot(rows[:, 0] * 1e3, rows[:, 3], 'o', color=color)
    axes[1].plot(I_dense * 1e3, theory[:, 1], '-', color=color, lw=1)

# ---------------------------------------------------------------- (b) hysteresis from rest
onset_th, retrap_th = pendulum_thresholds(SPIN_POL, params)
report(f"Pendulum model (Chen Eq. 9): onset from rest {onset_th * 1e3:.3f} mA, retrapping {retrap_th * 1e3:.3f} mA")
sweep = np.arange(0.05e-3, 1.0001e-3, 0.05e-3)
if args.fast:
    sweep = sweep[::4]
for polarity, label, _, color in branches:
    m_state = np.array([[1.0, 1e-3, 1e-3]])          # start at rest along the easy axis (x)
    up, down = [], []
    for I in sweep:                                   # sweep up, keeping the final state
        t, m = simulate(n_osc=1, current=I, spin_pol=SPIN_POL, polarity=polarity,
                        duration=15e-9 * ts, params=params, m0=m_state)
        up.append(bool(oscillator_state(t, m)['precessing'][0]))
        m_state = m[-1].copy()
    for I in sweep[::-1]:                             # sweep back down
        t, m = simulate(n_osc=1, current=I, spin_pol=SPIN_POL, polarity=polarity,
                        duration=15e-9 * ts, params=params, m0=m_state)
        down.append(bool(oscillator_state(t, m)['precessing'][0]))
        m_state = m[-1].copy()
    down = down[::-1]
    onset = sweep[np.argmax(up)] if any(up) else np.nan
    lowest = sweep[np.argmax(down)] if any(down) else np.nan
    report(f"{label}: onset (sweep up) {onset * 1e3:.2f} mA, lowest current still precessing "
           f"(sweep down) {lowest * 1e3:.2f} mA")
    offset = 0.05 if polarity > 0 else 0.0
    axes[2].step(sweep * 1e3, np.array(up) + offset, where='mid', color=color, label=f'{label[:11]} up')
    axes[2].step(sweep * 1e3, 0.9 * np.array(down) + offset, where='mid', ls='--', color=color,
                 label=f'{label[:11]} down')

axes[2].axvline(onset_th * 1e3, color='gray', lw=0.8)
axes[2].axvline(retrap_th * 1e3, color='gray', lw=0.8, ls=':')
axes[0].set_ylabel('frequency (GHz)')
axes[1].set_ylabel('|m_z|')
axes[2].set_ylabel('precessing (1) / static (0)')
for ax in axes:
    ax.set_xlabel('current (mA)')
axes[0].legend(fontsize=6)
axes[2].legend(fontsize=6)
fig.suptitle('Single PERP-STNO: simulation vs analytic theory (P = 0.38)')
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/01_single_oscillator.png", dpi=150)
report.save()
if args.show:
    plt.show()
