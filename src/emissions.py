"""Modelos de emisión para HMM.

Cada modelo de emisión implementa la misma interfaz mínima, de modo que los
algoritmos Forward/Backward, Viterbi y Baum-Welch (``hmm_core``) no dependen
del tipo de observación:

* ``log_likelihood(obs)`` -> log b_j(o_t) con forma ``(..., T, N)``.
* ``sample(states, rng)`` -> observaciones simuladas dados los estados.
* ``m_step(obs, gamma)``  -> reestimación de máxima verosimilitud (paso M de EM).

Distribuciones disponibles
--------------------------
``CategoricalEmission``  observaciones discretas (matriz B, filas que suman 1).
``GaussianEmission``     vector de rasgos continuos con covarianza diagonal.
``PoissonEmission``      conteos independientes (p. ej. movimientos oculares/min).
``ProductEmission``      bloques independientes (Gauss x Poisson) en un mismo vector.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln

_TINY = 1e-300


def safe_log(x: np.ndarray) -> np.ndarray:
    """Logaritmo natural que devuelve ``-inf`` (sin avisos) para ceros."""
    with np.errstate(divide="ignore"):
        return np.log(np.maximum(np.asarray(x, dtype=float), 0.0))


class CategoricalEmission:
    """Emisión discreta: ``B[j, k] = P(O_t = v_k | S_t = s_j)``."""

    obs_ndim = 1  # cada observación es un índice entero

    def __init__(self, B, tol: float = 1e-9):
        B = np.array(B, dtype=float)
        if B.ndim != 2:
            raise ValueError("B debe ser una matriz N x M.")
        if (B < 0).any() or not np.allclose(B.sum(axis=1), 1.0, atol=tol):
            raise ValueError("Cada fila de B debe ser una distribución (>=0 y suma 1).")
        self.B = B

    @property
    def n_states(self) -> int:
        return self.B.shape[0]

    @property
    def n_symbols(self) -> int:
        return self.B.shape[1]

    def log_likelihood(self, obs) -> np.ndarray:
        obs = np.asarray(obs, dtype=int)
        return np.moveaxis(safe_log(self.B)[:, obs], 0, -1)

    def sample(self, states, rng: np.random.Generator) -> np.ndarray:
        states = np.asarray(states, dtype=int)
        cdf = np.cumsum(self.B, axis=1)[states]          # (..., M)
        u = rng.random(states.shape)
        obs = (u[..., None] >= cdf).sum(axis=-1)
        return np.minimum(obs, self.n_symbols - 1)       # protege contra redondeo

    def m_step(self, obs, gamma) -> None:
        n, m = self.B.shape
        flat_o = np.asarray(obs, dtype=int).reshape(-1)
        flat_g = np.asarray(gamma).reshape(-1, n)
        counts = flat_g.T @ np.eye(m)[flat_o]            # (N, M) conteos esperados
        tot = counts.sum(axis=1, keepdims=True)
        ok = tot[:, 0] > 0
        newB = self.B.copy()
        newB[ok] = counts[ok] / tot[ok]
        self.B = newB

    def copy(self) -> "CategoricalEmission":
        return CategoricalEmission(self.B.copy())

    def params(self) -> dict:
        return {"B": self.B.copy()}


class GaussianEmission:
    """Emisión gaussiana multivariada con covarianza diagonal.

    ``b_j(o) = prod_d N(o_d; mu_jd, var_jd)``  (ecuación (1) del artículo con
    Sigma_j diagonal).
    """

    obs_ndim = 2  # cada observación es un vector de D rasgos

    def __init__(self, means, variances, min_var: float = 1e-6):
        self.means = np.array(means, dtype=float)
        self.vars = np.array(variances, dtype=float)
        if self.means.shape != self.vars.shape or self.means.ndim != 2:
            raise ValueError("means y variances deben ser N x D.")
        if (self.vars <= 0).any():
            raise ValueError("Las varianzas deben ser positivas.")
        self.min_var = min_var

    @property
    def n_states(self) -> int:
        return self.means.shape[0]

    @property
    def n_features(self) -> int:
        return self.means.shape[1]

    def log_likelihood(self, obs) -> np.ndarray:
        x = np.asarray(obs, dtype=float)[..., None, :]            # (..., T, 1, D)
        var = self.vars
        return -0.5 * (np.log(2 * np.pi * var).sum(-1) + ((x - self.means) ** 2 / var).sum(-1))

    def sample(self, states, rng: np.random.Generator) -> np.ndarray:
        states = np.asarray(states, dtype=int)
        z = rng.standard_normal(states.shape + (self.n_features,))
        return self.means[states] + np.sqrt(self.vars[states]) * z

    def m_step(self, obs, gamma) -> None:
        d = self.n_features
        x = np.asarray(obs, dtype=float).reshape(-1, d)
        g = np.asarray(gamma).reshape(-1, self.n_states)
        w = g.sum(axis=0)[:, None]
        ok = w[:, 0] > 0
        mu = self.means.copy()
        var = self.vars.copy()
        mu[ok] = (g.T @ x)[ok] / w[ok]
        second = (g.T @ (x ** 2))[ok] / w[ok]
        var[ok] = np.maximum(second - mu[ok] ** 2, self.min_var)
        self.means, self.vars = mu, var

    def copy(self) -> "GaussianEmission":
        return GaussianEmission(self.means.copy(), self.vars.copy(), self.min_var)

    def params(self) -> dict:
        return {"means": self.means.copy(), "vars": self.vars.copy()}


class PoissonEmission:
    """Conteos independientes ``O_d ~ Poisson(lambda_jd)`` dado el estado j."""

    obs_ndim = 2

    def __init__(self, rates, min_rate: float = 1e-6):
        self.rates = np.array(rates, dtype=float)
        if self.rates.ndim != 2 or (self.rates <= 0).any():
            raise ValueError("rates debe ser N x D con valores positivos.")
        self.min_rate = min_rate

    @property
    def n_states(self) -> int:
        return self.rates.shape[0]

    @property
    def n_features(self) -> int:
        return self.rates.shape[1]

    def log_likelihood(self, obs) -> np.ndarray:
        x = np.asarray(obs, dtype=float)[..., None, :]            # (..., T, 1, D)
        lam = self.rates
        return (x * np.log(lam) - lam - gammaln(x + 1.0)).sum(-1)

    def sample(self, states, rng: np.random.Generator) -> np.ndarray:
        states = np.asarray(states, dtype=int)
        return rng.poisson(self.rates[states]).astype(float)

    def m_step(self, obs, gamma) -> None:
        d = self.n_features
        x = np.asarray(obs, dtype=float).reshape(-1, d)
        g = np.asarray(gamma).reshape(-1, self.n_states)
        w = g.sum(axis=0)[:, None]
        ok = w[:, 0] > 0
        lam = self.rates.copy()
        lam[ok] = np.maximum((g.T @ x)[ok] / w[ok], self.min_rate)
        self.rates = lam

    def copy(self) -> "PoissonEmission":
        return PoissonEmission(self.rates.copy(), self.min_rate)

    def params(self) -> dict:
        return {"rates": self.rates.copy()}


class ProductEmission:
    """Producto de bloques de emisión independientes dado el estado oculto.

    El vector de observación se concatena en el orden de los bloques; p. ej.
    ``ProductEmission(Gaussian(1 rasgo), Poisson(1 conteo))`` -> ``D = 2``.
    """

    obs_ndim = 2

    def __init__(self, *blocks):
        if not blocks:
            raise ValueError("Se requiere al menos un bloque.")
        n = {b.n_states for b in blocks}
        if len(n) != 1:
            raise ValueError("Todos los bloques deben tener el mismo número de estados.")
        self.blocks = list(blocks)
        self.sizes = [b.n_features for b in blocks]
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)]).astype(int)

    @property
    def n_states(self) -> int:
        return self.blocks[0].n_states

    @property
    def n_features(self) -> int:
        return int(self.offsets[-1])

    def _slices(self, obs):
        obs = np.asarray(obs, dtype=float)
        for b, lo, hi in zip(self.blocks, self.offsets[:-1], self.offsets[1:]):
            yield b, obs[..., lo:hi]

    def log_likelihood(self, obs) -> np.ndarray:
        return sum(b.log_likelihood(x) for b, x in self._slices(obs))

    def sample(self, states, rng: np.random.Generator) -> np.ndarray:
        return np.concatenate([b.sample(states, rng) for b in self.blocks], axis=-1)

    def m_step(self, obs, gamma) -> None:
        for b, x in self._slices(obs):
            b.m_step(x, gamma)

    def copy(self) -> "ProductEmission":
        return ProductEmission(*[b.copy() for b in self.blocks])

    def params(self) -> dict:
        return {f"bloque{i}": b.params() for i, b in enumerate(self.blocks)}
