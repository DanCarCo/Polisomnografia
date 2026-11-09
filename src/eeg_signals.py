"""Señales EEG/EOG sintéticas por época y extracción de los símbolos del HMM.

Cada símbolo observado ``v_k`` se materializa como una señal sintética de una
época, dominada por la banda que da nombre al símbolo:

=========  =====================================================================
Símbolo    Señal simulada
=========  =====================================================================
alfa       sinusoide dominante 9-11.5 Hz (~35 µV) + ruido blanco (EEG relajado)
theta      sinusoide dominante 4.5-7.5 Hz (~40 µV) + ruido blanco
delta      ondas lentas 0.6-3.5 Hz (50-90 µV) + ruido blanco (sueño profundo)
REMs       EEG mixto de bajo voltaje + ráfagas rápidas de EOG (5-11 por época)
=========  =====================================================================

``extract_symbol`` implementa la regla operativa inversa (frecuencia pico de la
densidad espectral de Welch y conteo de ráfagas en el EOG). Sirve como
**verificación de consistencia interna** de la definición de los símbolos; no
valida la fisiología (la señal es sintética y generada desde el propio símbolo).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

FS = 100.0            # frecuencia de muestreo [Hz]
EPOCH_S = 30.0        # segundos simulados por época mostrada/analizada
BLINK_PROB = 0.04     # probabilidad de un parpadeo aislado en épocas no-REM
BURST_THRESHOLD_UV = 60.0
MIN_BURSTS_REM = 3


def simulate_epoch(symbol: int, rng: np.random.Generator, fs: float = FS, dur_s: float = EPOCH_S,
                   noise_sd: float = 8.0):
    """Devuelve ``(t, eeg, eog)`` en µV para una época cuyo símbolo dominante es ``symbol``."""
    n = int(fs * dur_s)
    t = np.arange(n) / fs
    ph = lambda: rng.uniform(0, 2 * np.pi)
    eeg = noise_sd * rng.standard_normal(n)
    eog = 10.0 * rng.standard_normal(n)

    if symbol == 0:      # alfa
        eeg += 35 * np.sin(2 * np.pi * rng.uniform(9.0, 11.5) * t + ph())
        eeg += 8 * np.sin(2 * np.pi * rng.uniform(4.5, 6.5) * t + ph())
    elif symbol == 1:    # theta
        eeg += 40 * np.sin(2 * np.pi * rng.uniform(4.5, 7.5) * t + ph())
        eeg += 6 * np.sin(2 * np.pi * rng.uniform(9.0, 11.0) * t + ph())
    elif symbol == 2:    # delta
        eeg += rng.uniform(70, 90) * np.sin(2 * np.pi * rng.uniform(0.6, 1.5) * t + ph())
        eeg += rng.uniform(40, 55) * np.sin(2 * np.pi * rng.uniform(1.6, 3.5) * t + ph())
        eeg += 10 * np.sin(2 * np.pi * rng.uniform(4.5, 7.0) * t + ph())
    else:                # REMs: EEG mixto de bajo voltaje
        eeg += 12 * np.sin(2 * np.pi * rng.uniform(4.0, 6.0) * t + ph())
        eeg += 6 * np.sin(2 * np.pi * rng.uniform(8.0, 9.5) * t + ph())
        eeg += 4 * np.sin(2 * np.pi * rng.uniform(14.0, 18.0) * t + ph())

    def pulse(center, amp, width=0.12):
        return amp * -(t - center) / width * np.exp(-0.5 * ((t - center) / width) ** 2)

    if symbol == 3:
        for c in rng.uniform(1.0, dur_s - 1.0, size=rng.integers(5, 12)):
            eog += pulse(c, rng.choice([-1, 1]) * rng.uniform(220, 380))
    elif rng.random() < BLINK_PROB:
        eog += pulse(rng.uniform(1.0, dur_s - 1.0), rng.choice([-1, 1]) * rng.uniform(220, 380))
    return t, eeg, eog


def _count_bursts(eog: np.ndarray, fs: float) -> int:
    """Cuenta picos |EOG| > umbral separados al menos 0.4 s."""
    idx = np.flatnonzero(np.abs(eog) > BURST_THRESHOLD_UV)
    if len(idx) == 0:
        return 0
    return int(1 + np.count_nonzero(np.diff(idx) > 0.4 * fs))


def extract_symbol(eeg: np.ndarray, eog: np.ndarray, fs: float = FS) -> int:
    """Regla operativa: REMs si hay ≥ 3 ráfagas en el EOG; si no, banda de la frecuencia pico."""
    if _count_bursts(eog, fs) >= MIN_BURSTS_REM:
        return 3
    f, p = welch(eeg, fs=fs, nperseg=int(4 * fs))
    band = (f >= 0.5) & (f <= 13.0)
    peak = f[band][np.argmax(p[band])]
    if peak < 4.0:
        return 2
    return 1 if peak < 8.0 else 0


def extraction_consistency(n_per_symbol: int, rng: np.random.Generator) -> np.ndarray:
    """Matriz de confusión (símbolo generado × símbolo extraído) de la regla operativa."""
    cm = np.zeros((4, 4), dtype=int)
    for s in range(4):
        for _ in range(n_per_symbol):
            _, eeg, eog = simulate_epoch(s, rng)
            cm[s, extract_symbol(eeg, eog)] += 1
    return cm


def night_spectrogram(obs: np.ndarray, rng: np.random.Generator, fs: float = 64.0, dur_s: float = 12.0,
                      fmax: float = 30.0):
    """Densidad espectral por época (dB) de una noche, a partir de los símbolos observados.

    Devuelve ``(frecuencias, matriz F x T)`` para dibujar el espectrograma.
    """
    cols = []
    for sym in np.asarray(obs, dtype=int):
        _, eeg, _ = simulate_epoch(int(sym), rng, fs=fs, dur_s=dur_s)
        f, p = welch(eeg, fs=fs, nperseg=int(4 * fs))
        cols.append(p[f <= fmax])
        freqs = f[f <= fmax]
    return freqs, 10.0 * np.log10(np.array(cols).T + 1e-12)
