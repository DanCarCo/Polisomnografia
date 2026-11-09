"""Pruebas de validación del núcleo HMM.

La implementación se valida contra (i) enumeración exhaustiva de todas las
trayectorias en un HMM pequeño (verdad matemática exacta) y (ii) la librería
independiente ``hmmlearn`` cuando está instalada.
"""
import itertools

import numpy as np
import pytest

from src.emissions import CategoricalEmission, GaussianEmission, PoissonEmission, ProductEmission
from src.hmm_core import HMM, viterbi_linear


def _small_hmm():
    pi = [0.5, 0.3, 0.2]
    A = [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.3, 0.6]]
    B = [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]]
    return HMM(pi, A, CategoricalEmission(B))


def _brute_force(h: HMM, obs):
    """Enumera las N^T trayectorias: devuelve (log P(O), mejor trayectoria, log P* , marginales)."""
    n, t_len = h.n_states, len(obs)
    logb = h.emission.log_likelihood(np.asarray(obs))                  # (T, N)
    total, best, best_lp = 0.0, None, -np.inf
    marg = np.zeros((t_len, n))
    for path in itertools.product(range(n), repeat=t_len):
        lp = np.log(h.pi[path[0]]) + logb[0, path[0]]
        for t in range(1, t_len):
            lp += np.log(h.A[path[t - 1], path[t]]) + logb[t, path[t]]
        p = np.exp(lp)
        total += p
        for t, s in enumerate(path):
            marg[t, s] += p
        if lp > best_lp:
            best_lp, best = lp, np.array(path)
    return np.log(total), best, best_lp, marg / total


def test_forward_viterbi_posterior_vs_enumeracion():
    h = _small_hmm()
    obs = np.array([0, 1, 2, 2, 1, 0, 1])
    ll, best, best_lp, marg = _brute_force(h, obs)
    assert h.log_likelihood(obs) == pytest.approx(ll, abs=1e-10)
    path, lp = h.viterbi(obs)
    assert np.array_equal(path, best)
    assert lp == pytest.approx(best_lp, abs=1e-10)
    _, gamma = h.posterior_decode(obs)
    assert np.allclose(gamma, marg, atol=1e-10)
    assert np.allclose(gamma.sum(axis=1), 1.0)


def test_emision_producto_gauss_poisson_vs_enumeracion():
    emis = ProductEmission(GaussianEmission([[0.0], [2.0], [5.0]], [[1.0], [1.5], [0.5]]),
                           PoissonEmission([[0.5], [2.0], [6.0]]))
    h = HMM([0.4, 0.4, 0.2], [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]], emis)
    rng = np.random.default_rng(1)
    _, obs = h.sample(6, rng)
    ll, best, _, marg = _brute_force(h, obs)
    assert h.log_likelihood(obs) == pytest.approx(ll, abs=1e-9)
    assert np.array_equal(h.viterbi(obs)[0], best)
    assert np.allclose(h.posterior_decode(obs)[1], marg, atol=1e-9)


def test_lote_equivale_a_secuencias_individuales():
    h = _small_hmm()
    rng = np.random.default_rng(3)
    obs = np.stack([h.sample(50, rng)[1] for _ in range(5)])
    paths, lps = h.viterbi(obs)
    lls = h.log_likelihood(obs)
    for k in range(5):
        p1, lp1 = h.viterbi(obs[k])
        assert np.array_equal(paths[k], p1)
        assert lps[k] == pytest.approx(lp1)
        assert lls[k] == pytest.approx(h.log_likelihood(obs[k]))


def test_sin_underflow_en_secuencias_largas():
    h = _small_hmm()
    _, obs = h.sample(20000, np.random.default_rng(0))
    assert np.isfinite(h.log_likelihood(obs))
    path, lp = h.viterbi(obs)
    assert np.isfinite(lp) and len(path) == 20000


def test_viterbi_lineal_hace_underflow_y_el_log_no():
    h = _small_hmm()
    _, obs = h.sample(800, np.random.default_rng(0))
    _, p_lin = viterbi_linear(obs, h.A, h.B, h.pi)
    assert p_lin == 0.0                     # underflow del dominio lineal
    assert np.isfinite(h.viterbi(obs)[1])   # el log-espacio sigue funcionando


def test_baum_welch_log_verosimilitud_no_decreciente():
    true = _small_hmm()
    rng = np.random.default_rng(7)
    obs = np.stack([true.sample(200, rng)[1] for _ in range(10)])
    init = HMM([1 / 3] * 3, rng.dirichlet(np.ones(3), size=3),
               CategoricalEmission(rng.dirichlet(np.ones(3), size=3)))
    hist = init.fit(obs, n_iter=60, tol=0)
    assert np.all(np.diff(hist) >= -1e-8)
    assert hist[-1] > hist[0]
    assert np.allclose(init.A.sum(axis=1), 1) and np.allclose(init.B.sum(axis=1), 1)


def test_baum_welch_conserva_ceros_estructurales():
    pi = [0.5, 0.5]
    A = [[0.7, 0.3], [0.0, 1.0]]
    B = [[0.8, 0.2], [0.3, 0.7]]
    h = HMM(pi, A, CategoricalEmission(B))
    obs = np.stack([h.sample(100, np.random.default_rng(s))[1] for s in range(5)])
    h2 = HMM([0.5, 0.5], [[0.6, 0.4], [0.0, 1.0]], CategoricalEmission([[0.6, 0.4], [0.4, 0.6]]))
    h2.fit(obs, n_iter=30, tol=0)
    assert h2.A[1, 0] == 0.0


def test_simulacion_frecuencias_convergen_a_A():
    h = _small_hmm()
    s, _ = h.sample(200_000, np.random.default_rng(11))
    counts = np.zeros((3, 3))
    np.add.at(counts, (s[:-1], s[1:]), 1)
    est = counts / counts.sum(axis=1, keepdims=True)
    assert np.abs(est - h.A).max() < 0.01
    freq = np.bincount(s, minlength=3) / len(s)
    assert np.abs(freq - h.stationary_distribution()).max() < 0.01


def test_reproducibilidad_misma_semilla():
    h = _small_hmm()
    a = h.sample(300, np.random.default_rng(np.random.SeedSequence(42)))
    b = h.sample(300, np.random.default_rng(np.random.SeedSequence(42)))
    c = h.sample(300, np.random.default_rng(np.random.SeedSequence(43)))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


def test_matrices_invalidas_se_rechazan():
    with pytest.raises(ValueError):
        HMM([0.5, 0.5], [[0.9, 0.2], [0.5, 0.5]], CategoricalEmission([[0.5, 0.5], [0.5, 0.5]]))
    with pytest.raises(ValueError):
        CategoricalEmission([[0.5, 0.6], [0.5, 0.5]])


# --------------------------------------------------------------- contraste independiente
hmmlearn = pytest.importorskip("hmmlearn", reason="hmmlearn no instalado")


def _to_hmmlearn(h: HMM):
    from hmmlearn.hmm import CategoricalHMM
    m = CategoricalHMM(n_components=h.n_states, n_features=h.B.shape[1], init_params="", params="")
    m.startprob_, m.transmat_, m.emissionprob_ = h.pi, h.A, h.B
    return m


def test_contra_hmmlearn_forward_y_viterbi():
    h = _small_hmm()
    _, obs = h.sample(400, np.random.default_rng(5))
    m = _to_hmmlearn(h)
    X = obs.reshape(-1, 1)
    assert h.log_likelihood(obs) == pytest.approx(m.score(X), abs=1e-8)
    lp, path = m.decode(X, algorithm="viterbi")
    assert np.array_equal(h.viterbi(obs)[0], path)
    assert h.viterbi(obs)[1] == pytest.approx(lp, abs=1e-8)
    post = m.predict_proba(X)
    assert np.allclose(h.posterior_decode(obs)[1], post, atol=1e-8)
