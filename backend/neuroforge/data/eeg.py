"""Synthetic EEG simulator (sensing layer).

Generates multi-channel EEG whose spectral content depends on a hidden *latent state*,
then extracts band-power features. The simulator deliberately injects realistic nuisances:

* 1/f (pink) background noise,
* eye-blink and EMG artifacts,
* session-to-session non-stationarity (amplitude drift + per-session offset).

None of this is real neural data — it exists so the downstream pipeline has a noisy,
non-stationary signal to decode, exactly as a real BCI would face.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from ..config import EEG_BANDS, SETTINGS
from ..models import EEGFeatures

# How each latent construct (0 = healthy, higher = worse) modulates band amplitudes.
# Values are multiplicative gains applied per band.
_LATENT_TO_BAND_GAIN: dict[str, dict[str, float]] = {
    # Neuroinflammation -> diffuse slowing (more delta/theta).
    "neuroinflammation": {"delta": 0.9, "theta": 0.7, "alpha": -0.2},
    # Parkinsonian -> exaggerated beta, reduced alpha.
    "dopaminergic_deficit": {"beta": 1.0, "alpha": -0.3, "gamma": 0.2},
    # Serotonergic deficit -> alpha changes (drives frontal asymmetry below).
    "serotonergic_deficit": {"alpha": 0.2, "theta": 0.3},
    # Pain -> elevated gamma / beta.
    "pain_index": {"gamma": 0.6, "beta": 0.4},
    # Mood -> theta + asymmetry.
    "mood_index": {"theta": 0.4},
    # Seizure risk -> spiky gamma & beta.
    "seizure_risk": {"gamma": 0.9, "beta": 0.5, "theta": 0.3},
}

# Base per-band amplitudes for a healthy resting EEG.
_BASE_BAND_AMP: dict[str, float] = {
    "delta": 1.0,
    "theta": 0.8,
    "alpha": 1.2,
    "beta": 0.6,
    "gamma": 0.3,
}

# Channels used; left/right frontal pair enables frontal alpha asymmetry.
_CHANNELS = ("F3", "F4", "Cz", "Pz")


class EEGSimulator:
    """Generate synthetic EEG and extract band-power features.

    Parameters
    ----------
    seed:
        Base RNG seed. Each :meth:`simulate` call may add a ``session`` offset to model
        non-stationarity across recordings of the same patient.
    """

    def __init__(self, seed: int | None = None):
        self.seed = SETTINGS.default_seed if seed is None else seed

    # ------------------------------------------------------------------ #
    def simulate(
        self,
        latent_state: dict[str, float],
        duration_s: float | None = None,
        fs: float | None = None,
        session: int = 0,
        artifact_level: float = 0.5,
    ) -> tuple[np.ndarray, EEGFeatures]:
        """Return ``(raw, features)`` where ``raw`` is ``(n_channels, n_samples)``."""
        fs = SETTINGS.eeg_fs if fs is None else fs
        duration_s = SETTINGS.eeg_duration_s if duration_s is None else duration_s
        rng = np.random.default_rng(self.seed + 1000 * session)

        n = int(duration_s * fs)
        t = np.arange(n) / fs

        # Per-band target amplitudes given the latent state.
        band_amp = self._band_amplitudes(latent_state)

        # Session non-stationarity: slow amplitude drift + global gain offset.
        drift = 1.0 + 0.15 * np.sin(
            2 * np.pi * t / max(duration_s, 1e-6) + rng.uniform(0, 2 * np.pi)
        )
        session_gain = 1.0 + 0.1 * rng.standard_normal()

        raw = np.zeros((len(_CHANNELS), n))
        for ci, ch in enumerate(_CHANNELS):
            sig = np.zeros(n)
            for band, (lo, hi) in EEG_BANDS.items():
                amp = band_amp[band] * session_gain
                # Frontal channels carry mood-driven alpha asymmetry.
                if band == "alpha" and ch in ("F3", "F4"):
                    asym = 0.4 * latent_state.get("mood_index", 0.0)
                    amp *= (1.0 - asym) if ch == "F3" else (1.0 + asym)
                # A few oscillators per band at jittered frequencies/phases.
                for _ in range(3):
                    f = rng.uniform(lo, hi)
                    phase = rng.uniform(0, 2 * np.pi)
                    sig += (amp / 3.0) * np.sin(2 * np.pi * f * t + phase)
            sig *= drift
            sig += self._pink_noise(n, rng) * 0.6  # 1/f background
            raw[ci] = sig

        raw, artifact_ratio = self._inject_artifacts(raw, rng, artifact_level, fs)
        features = self._extract_features(raw, fs, artifact_ratio)
        return raw, features

    # ------------------------------------------------------------------ #
    def _band_amplitudes(self, latent_state: dict[str, float]) -> dict[str, float]:
        amp = dict(_BASE_BAND_AMP)
        for construct, gains in _LATENT_TO_BAND_GAIN.items():
            sev = float(latent_state.get(construct, 0.0))
            for band, g in gains.items():
                amp[band] = max(0.05, amp[band] * (1.0 + g * sev))
        return amp

    @staticmethod
    def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
        """Approximate 1/f noise via spectral shaping of white noise."""
        white = rng.standard_normal(n)
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n)
        freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
        spectrum = spectrum / np.sqrt(freqs)
        pink = np.fft.irfft(spectrum, n=n)
        std = pink.std()
        return pink / std if std > 0 else pink

    @staticmethod
    def _inject_artifacts(
        raw: np.ndarray, rng: np.random.Generator, level: float, fs: float
    ) -> tuple[np.ndarray, float]:
        n = raw.shape[1]
        mask = np.zeros(n, dtype=bool)
        # Eye blinks: occasional large low-frequency transients (frontal-dominant).
        n_blinks = rng.poisson(2.0 * level)
        for _ in range(n_blinks):
            c = rng.integers(0, n)
            width = int(0.2 * fs)
            lo, hi = max(0, c - width), min(n, c + width)
            window = np.hanning(hi - lo) * (6.0 * level)
            raw[0, lo:hi] += window
            raw[1, lo:hi] += window * 0.8
            mask[lo:hi] = True
        # EMG bursts: high-frequency noise on a random channel.
        n_emg = rng.poisson(1.5 * level)
        for _ in range(n_emg):
            c = rng.integers(0, n)
            width = int(0.1 * fs)
            lo, hi = max(0, c - width), min(n, c + width)
            ch = rng.integers(0, raw.shape[0])
            raw[ch, lo:hi] += rng.standard_normal(hi - lo) * (4.0 * level)
            mask[lo:hi] = True
        return raw, float(mask.mean())

    @staticmethod
    def _extract_features(raw: np.ndarray, fs: float, artifact_ratio: float) -> EEGFeatures:
        nperseg = min(raw.shape[1], int(fs * 2))
        freqs, psd = signal.welch(raw, fs=fs, nperseg=nperseg, axis=1)
        psd_mean = psd.mean(axis=0)  # average across channels

        band_power: dict[str, float] = {}
        for band, (lo, hi) in EEG_BANDS.items():
            idx = (freqs >= lo) & (freqs < hi)
            band_power[band] = float(np.trapezoid(psd_mean[idx], freqs[idx])) if idx.any() else 0.0
        total = sum(band_power.values()) or 1.0
        rel = {b: p / total for b, p in band_power.items()}

        # Frontal alpha asymmetry: log(F4 alpha) - log(F3 alpha).
        alpha_idx = (freqs >= EEG_BANDS["alpha"][0]) & (freqs < EEG_BANDS["alpha"][1])
        f3_alpha = float(np.trapezoid(psd[0, alpha_idx], freqs[alpha_idx])) + 1e-9
        f4_alpha = float(np.trapezoid(psd[1, alpha_idx], freqs[alpha_idx])) + 1e-9
        faa = float(np.log(f4_alpha) - np.log(f3_alpha))

        # Crude SNR: band-limited (1-45 Hz) power vs out-of-band power.
        inband = (freqs >= 1.0) & (freqs < 45.0)
        sig_p = float(psd_mean[inband].sum()) + 1e-12
        noise_p = float(psd_mean[~inband].sum()) + 1e-12
        snr_db = 10.0 * np.log10(sig_p / noise_p)

        return EEGFeatures(
            relative_power=rel,
            frontal_alpha_asymmetry=faa,
            snr_db=snr_db,
            artifact_ratio=artifact_ratio,
        )
