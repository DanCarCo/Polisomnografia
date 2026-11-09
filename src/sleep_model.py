"""Especificación formal del HMM de etapas del sueño y de sus variantes.

Modelo base (lambda = (S, V, A, B, pi)), tomado del enunciado de la actividad:

* S = {Despierto, Sueño Ligero, Sueño Profundo, REM}            (N = 4 estados ocultos)
* V = {Ondas Alpha, Ondas Theta, Ondas Delta, Mov. Oculares Rápidos}   (M = 4 símbolos)
* Época = 1 minuto; una noche completa = 480 épocas (8 h).

Correspondencia clínica (manual AASM): Despierto = W, Sueño Ligero = N1+N2,
Sueño Profundo = N3 (ondas lentas), REM = R. Los símbolos son rasgos discretos
por época: banda dominante del EEG (alfa 8-13 Hz, theta 4-8 Hz, delta 0.5-4 Hz)
o presencia de ráfagas de movimientos oculares rápidos en el EOG.

AVISO: son parámetros didácticos del enunciado, **no** estimados de registros
reales de polisomnografía. Ver sección de limitaciones del artículo.
"""
from __future__ import annotations

import numpy as np

from .emissions import CategoricalEmission, GaussianEmission, PoissonEmission, ProductEmission
from .hmm_core import HMM

STATE_NAMES = ("Despierto", "Sueño Ligero", "Sueño Profundo", "REM")
STATE_SHORT = ("W", "N1/N2", "N3", "REM")
OBS_NAMES = ("Ondas Alpha", "Ondas Theta", "Ondas Delta", "Mov. Oculares Rápidos")
OBS_SHORT = ("α", "θ", "δ", "REMs")

IDX_W, IDX_L, IDX_D, IDX_R = 0, 1, 2, 3

# ----------------------------------------------------------------- parámetros del enunciado
PI_NOM = np.array([0.50, 0.30, 0.10, 0.10])

A_NOM = np.array([
    [0.50, 0.40, 0.05, 0.05],   # Despierto
    [0.10, 0.60, 0.20, 0.10],   # Sueño Ligero
    [0.02, 0.28, 0.60, 0.10],   # Sueño Profundo
    [0.20, 0.20, 0.00, 0.60],   # REM  (cero estructural: REM no pasa directo a profundo)
])

B_NOM = np.array([
    [0.70, 0.20, 0.05, 0.05],   # Despierto
    [0.15, 0.65, 0.15, 0.05],   # Sueño Ligero
    [0.02, 0.13, 0.80, 0.05],   # Sueño Profundo
    [0.10, 0.10, 0.10, 0.70],   # REM
])

# Emisión continua (escenario E4). Rasgos por época:
#   x1 = frecuencia dominante del EEG [Hz]  ~ Normal(mu_j, sigma_j^2)
#   x2 = número de ráfagas de REM por minuto ~ Poisson(lambda_j)
# Fijados a priori con solapamiento moderado-alto (separación de medias de 1.5-1.75 sigma
# entre vigilia/ligero/profundo; ligero y REM casi indistinguibles en x1).
GAUSS_MEANS = np.array([[9.5], [6.5], [3.0], [6.0]])
GAUSS_SD = np.array([[2.0], [2.0], [2.0], [2.0]])
POISSON_RATES = np.array([[0.8], [0.6], [0.4], [3.0]])


# ----------------------------------------------------------------- constructores
def nominal_hmm(A=None, B=None, pi=None) -> HMM:
    """HMM categórico del enunciado (con posibles sustituciones de A, B o pi)."""
    return HMM(PI_NOM if pi is None else pi,
               A_NOM if A is None else A,
               CategoricalEmission(B_NOM if B is None else B),
               state_names=STATE_NAMES, obs_names=OBS_NAMES)


def gaussian_poisson_hmm(A=None, pi=None) -> HMM:
    """HMM con emisión continua mixta Gauss(frecuencia) x Poisson(ráfagas REM)."""
    emis = ProductEmission(GaussianEmission(GAUSS_MEANS, GAUSS_SD ** 2), PoissonEmission(POISSON_RATES))
    return HMM(PI_NOM if pi is None else pi, A_NOM if A is None else A, emis,
               state_names=STATE_NAMES, obs_names=("Frecuencia dominante [Hz]", "Ráfagas REM/min"))


# ----------------------------------------------------------------- transformaciones de parámetros
def blend_with_uniform(M: np.ndarray, eps: float) -> np.ndarray:
    """Mezcla convexa con la distribución uniforme por fila: (1-eps) M + eps/ncols.

    eps = 0 deja M intacta; eps = 1 produce filas uniformes (máxima entropía).
    Conserva la estocasticidad por filas. Se aplica igual a A y a B para que la
    comparación de sensibilidad sea justa (misma operación, misma intensidad).
    """
    M = np.asarray(M, dtype=float)
    return (1.0 - eps) * M + eps / M.shape[1]


def scale_persistence(A: np.ndarray, diag) -> np.ndarray:
    """Fija la probabilidad de permanencia a_ii = diag y reparte el resto
    conservando el patrón relativo de transiciones fuera de la diagonal.

    La duración de una racha en el estado i es geométrica con media 1/(1 - a_ii)
    épocas, de modo que ``diag`` controla directamente la duración media de las etapas.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    d = np.broadcast_to(np.asarray(diag, dtype=float), (n,))
    out = np.zeros_like(A)
    for i in range(n):
        off = A[i].copy()
        off[i] = 0.0
        s = off.sum()
        out[i] = off / s * (1.0 - d[i]) if s > 0 else off
        out[i, i] = d[i] if s > 0 else 1.0
    return out


def dirichlet_perturb(M: np.ndarray, kappa: float, rng: np.random.Generator, floor: float = 1e-3) -> np.ndarray:
    """Perturbación estocástica por fila: ``M'_i ~ Dirichlet(kappa * M_i)``.

    La media de la fila perturbada es ``M_i`` y su varianza decrece con la
    concentración ``kappa`` (Var[p] = p(1-p)/(kappa+1)). Los ceros se sustituyen por
    ``floor`` en la concentración para que la Dirichlet esté bien definida.
    """
    M = np.asarray(M, dtype=float)
    alpha = np.maximum(kappa * M, floor)
    out = np.vstack([rng.dirichlet(a) for a in alpha])
    return out


# ----------------------------------------------------------------- alteraciones simuladas
def fragmented_hmm(h: HMM, delta: float = 0.15) -> HMM:
    """Sueño fragmentado: desde cada etapa de sueño se añade masa ``delta`` a la
    transición hacia Despierto (más despertares nocturnos)."""
    A = h.A.copy()
    for i in (IDX_L, IDX_D, IDX_R):
        A[i] = (1.0 - delta) * A[i]
        A[i, IDX_W] += delta
    return HMM(h.pi, A, h.emission.copy(), h.state_names, h.obs_names)


def rem_scaled_hmm(h: HMM, factor: float) -> HMM:
    """Reduce la entrada a REM multiplicando por ``factor`` las transiciones hacia REM
    (y pi_REM) y renormalizando. ``factor = 1`` es el modelo sano; ``factor = 0``, ausencia de REM."""
    A = h.A.copy()
    for i in (IDX_W, IDX_L, IDX_D):
        A[i, IDX_R] *= factor
        A[i] /= A[i].sum()
    # La fila REM se conserva estocástica (si factor = 0 queda inalcanzable).
    pi = h.pi.copy()
    pi[IDX_R] *= factor
    pi /= pi.sum()
    return HMM(pi, A, h.emission.copy(), h.state_names, h.obs_names)


def no_rem_hmm(h: HMM) -> HMM:
    """Ausencia de fase REM: se anulan las transiciones hacia REM y la probabilidad inicial de REM."""
    return rem_scaled_hmm(h, 0.0)
