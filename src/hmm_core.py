"""Núcleo HMM (NumPy): simulación, Forward/Backward, Viterbi y Baum-Welch.

Notación del artículo: ``lambda = (S, V, A, B, pi)``.

* ``pi``  (N,)    distribución inicial,        ``pi_i = P(S_1 = s_i)``.
* ``A``   (N, N)  matriz de transición,        ``a_ij = P(S_t = s_j | S_{t-1} = s_i)``.
* ``emission``    modelo de emisión (matriz B si es categórico).

Decisiones de implementación (estabilidad numérica)
---------------------------------------------------
* Forward/Backward usan **escalado por época** (Rabiner, 1989) y la
  log-verosimilitud se obtiene como ``sum_t log c_t``; así no hay *underflow*
  con noches de 480 o más épocas.
* Viterbi opera en **log-espacio** (sumas en lugar de productos). La versión
  en dominio lineal (``viterbi_linear``) se conserva solo para demostrar el
  problema de *underflow* (véase ``experiments.numerical_stability``).
* Todos los métodos aceptan una secuencia ``(T,)`` o un lote ``(K, T)`` de
  secuencias de igual longitud; internamente se procesa por lotes (K, T, N).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .emissions import safe_log

_TINY = 1e-300


@dataclass
class ForwardBackward:
    """Resultado de Forward-Backward escalado para un lote de secuencias."""

    log_b: np.ndarray    # (K, T, N) log-verosimilitud de emisión
    b: np.ndarray        # (K, T, N) emisión desplazada por época (exp(log_b - m))
    alpha: np.ndarray    # (K, T, N) forward escalado
    beta: np.ndarray     # (K, T, N) backward escalado
    c: np.ndarray        # (K, T)    factores de escala
    gamma: np.ndarray    # (K, T, N) P(S_t = i | O_{1:T})
    loglik: np.ndarray   # (K,)      log P(O | lambda)


class HMM:
    """Modelo oculto de Markov de primer orden, homogéneo en el tiempo."""

    def __init__(self, pi, A, emission, state_names=None, obs_names=None, tol: float = 1e-9):
        self.pi = np.array(pi, dtype=float)
        self.A = np.array(A, dtype=float)
        self.emission = emission
        n = self.A.shape[0]
        if self.A.shape != (n, n) or self.pi.shape != (n,):
            raise ValueError("Dimensiones incompatibles entre pi y A.")
        if emission.n_states != n:
            raise ValueError("La emisión debe tener N estados.")
        if (self.A < 0).any() or not np.allclose(self.A.sum(axis=1), 1.0, atol=tol):
            raise ValueError("Cada fila de A debe sumar 1.")
        if (self.pi < 0).any() or not np.isclose(self.pi.sum(), 1.0, atol=tol):
            raise ValueError("pi debe sumar 1.")
        self.state_names = list(state_names) if state_names is not None else [f"S{i}" for i in range(n)]
        self.obs_names = list(obs_names) if obs_names is not None else None

    # ------------------------------------------------------------------ utilidades
    @property
    def n_states(self) -> int:
        return self.A.shape[0]

    @property
    def B(self) -> np.ndarray:
        """Matriz de emisión (solo para emisión categórica)."""
        return self.emission.B

    def copy(self) -> "HMM":
        return HMM(self.pi.copy(), self.A.copy(), self.emission.copy(), self.state_names, self.obs_names)

    def stationary_distribution(self) -> np.ndarray:
        """Distribución estacionaria: autovector izquierdo de A con autovalor 1."""
        w, v = np.linalg.eig(self.A.T)
        k = np.argmin(np.abs(w - 1.0))
        p = np.clip(np.real(v[:, k]) / np.real(v[:, k]).sum(), 0.0, None)   # signo y ruido numérico
        return p / p.sum()

    def _batch(self, obs):
        obs = np.asarray(obs)
        extra = obs.ndim - self.emission.obs_ndim
        if extra == 0:
            return obs[None], True
        if extra == 1:
            return obs, False
        raise ValueError("Forma de observaciones no válida.")

    # ------------------------------------------------------------------ simulación
    def sample(self, T: int, rng: np.random.Generator, initial_state: int | None = None):
        """Simula una secuencia de longitud T. Devuelve ``(estados, observaciones)``."""
        cum_a = np.cumsum(self.A, axis=1)
        cum_a[:, -1] = 1.0
        cum_pi = np.cumsum(self.pi)
        cum_pi[-1] = 1.0
        s = np.empty(T, dtype=int)
        s[0] = initial_state if initial_state is not None else np.searchsorted(cum_pi, rng.random(), side="right")
        u = rng.random(T)
        for t in range(1, T):
            s[t] = np.searchsorted(cum_a[s[t - 1]], u[t], side="right")
        return s, self.emission.sample(s, rng)

    # ------------------------------------------------------------------ Forward-Backward
    def forward_backward(self, obs) -> ForwardBackward:
        """Forward-Backward escalado (Problema 1: evaluación) para un lote."""
        obs_b, _ = self._batch(obs)
        log_b = self.emission.log_likelihood(obs_b)                    # (K, T, N)
        k, t_len, n = log_b.shape
        m = log_b.max(axis=2, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)
        b = np.exp(log_b - m)

        alpha = np.empty((k, t_len, n))
        c = np.empty((k, t_len))
        a = self.pi[None, :] * b[:, 0]
        c[:, 0] = np.maximum(a.sum(axis=1), _TINY)
        alpha[:, 0] = a / c[:, 0, None]
        for t in range(1, t_len):
            a = (alpha[:, t - 1] @ self.A) * b[:, t]
            c[:, t] = np.maximum(a.sum(axis=1), _TINY)
            alpha[:, t] = a / c[:, t, None]

        beta = np.empty((k, t_len, n))
        beta[:, -1] = 1.0
        for t in range(t_len - 2, -1, -1):
            beta[:, t] = ((b[:, t + 1] * beta[:, t + 1]) @ self.A.T) / c[:, t + 1, None]

        gamma = alpha * beta
        gamma /= gamma.sum(axis=2, keepdims=True)
        loglik = (np.log(c) + m[:, :, 0]).sum(axis=1)
        return ForwardBackward(log_b, b, alpha, beta, c, gamma, loglik)

    def log_likelihood(self, obs):
        """``log P(O | lambda)`` por secuencia (algoritmo Forward)."""
        obs_b, squeeze = self._batch(obs)
        ll = self.forward_backward(obs_b).loglik
        return float(ll[0]) if squeeze else ll

    def posterior_decode(self, obs):
        """Decodificación posterior: ``argmax_i P(S_t = i | O)`` por época.

        Devuelve ``(trayectoria, gamma)``. Maximiza la exactitud esperada por
        época, a diferencia de Viterbi, que maximiza la probabilidad conjunta.
        """
        obs_b, squeeze = self._batch(obs)
        gamma = self.forward_backward(obs_b).gamma
        path = gamma.argmax(axis=2)
        return (path[0], gamma[0]) if squeeze else (path, gamma)

    # ------------------------------------------------------------------ Viterbi
    def viterbi(self, obs):
        """Viterbi en log-espacio (Problema 2: decodificación).

        Devuelve ``(trayectoria, log P(trayectoria, O | lambda))``.
        """
        obs_b, squeeze = self._batch(obs)
        log_b = self.emission.log_likelihood(obs_b)
        k, t_len, n = log_b.shape
        log_a = safe_log(self.A)
        delta = safe_log(self.pi)[None, :] + log_b[:, 0]
        psi = np.zeros((k, t_len, n), dtype=np.int64)
        for t in range(1, t_len):
            scores = delta[:, :, None] + log_a[None, :, :]             # [k, i, j]
            psi[:, t] = scores.argmax(axis=1)
            delta = scores.max(axis=1) + log_b[:, t]
        path = np.empty((k, t_len), dtype=np.int64)
        path[:, -1] = delta.argmax(axis=1)
        rows = np.arange(k)
        for t in range(t_len - 1, 0, -1):
            path[:, t - 1] = psi[rows, t, path[:, t]]
        logp = delta.max(axis=1)
        return (path[0], float(logp[0])) if squeeze else (path, logp)

    def path_log_prob(self, obs, path) -> float:
        """``log P(trayectoria, O | lambda)`` de una trayectoria concreta (para comparar rutas)."""
        path = np.asarray(path, dtype=int)
        log_b = self.emission.log_likelihood(np.asarray(obs))              # (T, N)
        t_len = len(path)
        lp = safe_log(self.pi)[path[0]] + log_b[0, path[0]]
        lp += (safe_log(self.A)[path[:-1], path[1:]] + log_b[np.arange(1, t_len), path[1:]]).sum()
        return float(lp)

    # ------------------------------------------------------------------ Baum-Welch
    def fit(self, obs, n_iter: int = 200, tol: float = 1e-6, min_prob: float = 0.0):
        """Baum-Welch (EM) con varias secuencias de igual longitud (Problema 3).

        Modifica el modelo *in situ*. Devuelve el historial de log-verosimilitud
        total (no decreciente salvo error numérico). Las probabilidades
        exactamente cero se conservan (ceros estructurales de A).
        """
        obs_b, _ = self._batch(obs)
        history: list[float] = []
        for _ in range(n_iter):
            fb = self.forward_backward(obs_b)
            history.append(float(fb.loglik.sum()))
            n = self.n_states

            # Paso M: pi
            pi = fb.gamma[:, 0].mean(axis=0)
            # Paso M: A  (xi_t(i,j) = alpha_t(i) a_ij b_j(t+1) beta_{t+1}(j) / c_{t+1})
            left = fb.alpha[:, :-1].reshape(-1, n)
            right = (fb.b[:, 1:] * fb.beta[:, 1:] / fb.c[:, 1:, None]).reshape(-1, n)
            xi_sum = self.A * (left.T @ right)
            row = xi_sum.sum(axis=1, keepdims=True)
            A = np.where(row > 0, xi_sum / np.maximum(row, _TINY), self.A)
            if min_prob > 0:
                A = np.where(A > 0, np.maximum(A, min_prob), A)
                A /= A.sum(axis=1, keepdims=True)
            self.pi = pi / pi.sum()
            self.A = A
            # Paso M: emisión
            self.emission.m_step(obs_b, fb.gamma)

            if len(history) > 1 and abs(history[-1] - history[-2]) <= tol * (1.0 + abs(history[-2])):
                break
        return history


# ----------------------------------------------------------------------------------
# Viterbi en dominio lineal: SOLO para el estudio de estabilidad numérica.
# Reproduce el listado de la plantilla (productos de probabilidades sin escalar).
# ----------------------------------------------------------------------------------
def viterbi_linear(obs, A, B, pi):
    """Viterbi ingenuo (producto de probabilidades). Sufre *underflow* si T es grande."""
    obs = np.asarray(obs, dtype=int)
    n, t_len = A.shape[0], len(obs)
    delta = np.zeros((t_len, n))
    psi = np.zeros((t_len, n), dtype=int)
    delta[0] = pi * B[:, obs[0]]
    for t in range(1, t_len):
        tmp = delta[t - 1][:, None] * A
        psi[t] = tmp.argmax(axis=0)
        delta[t] = tmp.max(axis=0) * B[:, obs[t]]
    q = np.zeros(t_len, dtype=int)
    q[-1] = delta[-1].argmax()
    for t in range(t_len - 2, -1, -1):
        q[t] = psi[t + 1, q[t + 1]]
    return q, float(delta[-1].max())
