"""
Analysis of simulated trajectories: precession phase and frequency, oscillation state,
phase-locking metrics, spectra, readout signals and linewidth.

Unless stated otherwise, quantities are evaluated over the last part of the trajectory
(from fraction `frac` of the samples onward) to discard the transient.
"""
import numpy as np

CONE_MIN = 0.05      # cone amplitude below which an oscillator counts as static
F_MIN = 0.3e9        # |frequency| (Hz) below which an oscillator counts as static


def start_index(n_samples, frac):
    """Index from which the steady-state part of a trajectory starts."""
    return int(n_samples * frac)


def inplane_phase(m):
    """Unwrapped precession phase about z, shape (n_samples, n_osc). Valid when the axis is z."""
    return np.unwrap(np.arctan2(m[:, :, 1], m[:, :, 0]), axis=0)


def precession_axis(m, frac=0.5):
    """Time-averaged magnetization direction of each oscillator (falls back to z if ~0)."""
    mean = m[start_index(m.shape[0], frac):].mean(axis=0)
    axes = np.zeros_like(mean)
    for i in range(mean.shape[0]):
        norm = np.linalg.norm(mean[i])
        axes[i] = mean[i] / norm if norm > 1e-3 else np.array([0.0, 0.0, 1.0])
    return axes


def cone_amplitude(m, axes, frac=0.5):
    """Mean component of m perpendicular to the precession axis (~0 for a static state)."""
    parallel = np.einsum('tij,ij->ti', m[start_index(m.shape[0], frac):], axes)
    return np.sqrt(np.clip(1.0 - parallel ** 2, 0.0, None)).mean(axis=0)


def axis_frame(axis):
    """Two unit vectors (e1, e2) perpendicular to `axis`, forming a right-handed frame."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = ref - axis * np.dot(ref, axis)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(axis, e1)


def axis_phase(m, axes):
    """Unwrapped precession phase of each oscillator about its own axis."""
    phase = np.zeros(m.shape[:2])
    for i in range(m.shape[1]):
        e1, e2 = axis_frame(axes[i])
        phase[:, i] = np.arctan2(m[:, i, :] @ e2, m[:, i, :] @ e1)
    return np.unwrap(phase, axis=0)


def frequency_from_phase(t, phase, frac=0.5):
    """Mean (signed) frequency in Hz from the phase advance over the steady-state window."""
    i0 = start_index(len(t), frac)
    return (phase[-1] - phase[i0]) / (t[-1] - t[i0]) / (2.0 * np.pi)


def oscillator_state(t, m, frac=0.5):
    """Per oscillator: precession axis, cone amplitude, signed frequency, mean m_z, precessing flag."""
    axes = precession_axis(m, frac)
    cone = cone_amplitude(m, axes, frac)
    freq = frequency_from_phase(t, axis_phase(m, axes), frac)
    mz_mean = m[start_index(len(t), frac):, :, 2].mean(axis=0)
    precessing = (cone > CONE_MIN) & (np.abs(freq) > F_MIN)
    return dict(axis=axes, cone=cone, freq=freq, mz=mz_mean, precessing=precessing)


def locking_metrics(t, phase, i, j, frac=0.5):
    """Phase-locking metrics of the phase difference dphi = phi_i - phi_j.

    R      |<exp(i dphi)>|: 1 = locked at any fixed phase, ~0 = drifting
    dphi   mean locked phase difference (deg)
    cos    <cos dphi>: -1 anti-phase, +1 in-phase
    slips  net number of 2 pi phase slips over the window
    df     mean frequency difference (Hz)
    """
    i0 = start_index(len(t), frac)
    dphi = phase[i0:, i] - phase[i0:, j]
    z = np.exp(1j * dphi).mean()
    slips = (dphi[-1] - dphi[0]) / (2.0 * np.pi)
    return dict(R=float(abs(z)), dphi=float(np.degrees(np.angle(z))),
                cos=float(np.cos(dphi).mean()), slips=float(slips),
                df=float(slips / (t[-1] - t[i0])))


def is_locked(metrics):
    """Locked = strong phase coherence and less than half a phase slip over the window."""
    return metrics['R'] > 0.9 and abs(metrics['slips']) < 0.5


def power_spectrum(t, signal, frac=0.5):
    """One-sided power spectrum (Hann window) of a signal: returns (frequencies in Hz, power)."""
    i0 = start_index(len(t), frac)
    s = signal[i0:] - signal[i0:].mean()
    power = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
    return np.fft.rfftfreq(len(s), t[1] - t[0]), power


def readout_signals(m):
    """Magnetoresistive readout proxies of a vertical stack.

    summed      sum_i m_x,i      (all reference layers parallel)
    alternating sum_i (-1)^i m_x,i (alternating reference layers, Chen et al. 2014 Fig. 1d)
    """
    signs = (-1.0) ** np.arange(m.shape[1])
    return m[:, :, 0].sum(axis=1), (m[:, :, 0] * signs).sum(axis=1)


def cancellation_factor(m, signal, frac=0.5):
    """std(readout) / sum_i std(m_x,i): 1 = signals add coherently, ~0 = they cancel."""
    i0 = start_index(m.shape[0], frac)
    return float(signal[i0:].std() / m[i0:, :, 0].std(axis=0).sum())


def phase_linewidth(t, phase, frac=0.2):
    """Full linewidth (FWHM, Hz) from phase diffusion, assuming a Lorentzian line:
    <[phi(t + tau) - phi(t)]^2> = 2 pi * FWHM * tau  (after removing the mean frequency)."""
    i0 = start_index(len(t), frac)
    tt, pp = t[i0:], phase[i0:]
    residual = pp - np.polyval(np.polyfit(tt, pp, 1), tt)
    lags = np.unique(np.logspace(0, np.log10(max(2, len(tt) // 10)), 20).astype(int))
    variance = [np.var(residual[lag:] - residual[:-lag]) for lag in lags]
    slope = np.polyfit(lags * (tt[1] - tt[0]), variance, 1)[0]
    return slope / (2.0 * np.pi)
