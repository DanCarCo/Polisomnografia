"""Pruebas del modelo del sueño, las métricas clínicas, la estadística y la reproducibilidad."""
import numpy as np
import pandas as pd
import pytest

from src import clinical, experiments as ex, sleep_model as sm
from src.eeg_signals import extract_symbol, simulate_epoch
from src.utils_stats import (auc_from_scores, cohen_kappa, confusion_matrix, holm_adjust, macro_f1, summarize)


# ------------------------------------------------------------ especificación del modelo
def test_parametros_del_enunciado_son_estocasticos():
    assert sm.PI_NOM.sum() == pytest.approx(1.0)
    assert np.allclose(sm.A_NOM.sum(axis=1), 1.0)
    assert np.allclose(sm.B_NOM.sum(axis=1), 1.0)
    assert sm.A_NOM[sm.IDX_R, sm.IDX_D] == 0.0          # cero estructural REM -> profundo
    h = sm.nominal_hmm()
    assert h.n_states == 4 and h.B.shape == (4, 4)


def test_distribucion_estacionaria_satisface_pi_A_igual_pi():
    h = sm.nominal_hmm()
    p = h.stationary_distribution()
    assert p.sum() == pytest.approx(1.0)
    assert np.allclose(p @ h.A, p, atol=1e-10)


@pytest.mark.parametrize("eps", [0.0, 0.3, 0.8])
def test_mezcla_con_uniforme_conserva_filas(eps):
    for M in (sm.A_NOM, sm.B_NOM):
        out = sm.blend_with_uniform(M, eps)
        assert np.allclose(out.sum(axis=1), 1.0) and (out >= 0).all()
    assert np.allclose(sm.blend_with_uniform(sm.A_NOM, 0.0), sm.A_NOM)


def test_persistencia_fija_la_diagonal_y_conserva_el_patron():
    A2 = sm.scale_persistence(sm.A_NOM, 0.9)
    assert np.allclose(np.diag(A2), 0.9) and np.allclose(A2.sum(axis=1), 1.0)
    off, off2 = sm.A_NOM.copy(), A2.copy()
    np.fill_diagonal(off, 0)
    np.fill_diagonal(off2, 0)
    assert np.allclose(off2 / off2.sum(axis=1, keepdims=True), off / off.sum(axis=1, keepdims=True))
    assert A2[sm.IDX_R, sm.IDX_D] == 0.0


def test_dirichlet_conserva_estocasticidad_y_media():
    rng = np.random.default_rng(0)
    samples = np.stack([sm.dirichlet_perturb(sm.A_NOM, 50.0, rng) for _ in range(2000)])
    assert np.allclose(samples.sum(axis=2), 1.0)
    assert np.abs(samples.mean(axis=0) - sm.A_NOM).max() < 0.01


def test_modelos_patologicos_son_validos():
    nom = sm.nominal_hmm()
    frag, norem = sm.fragmented_hmm(nom, 0.15), sm.no_rem_hmm(nom)
    assert np.allclose(frag.A.sum(axis=1), 1.0) and np.allclose(norem.A.sum(axis=1), 1.0)
    assert (norem.A[[sm.IDX_W, sm.IDX_L, sm.IDX_D], sm.IDX_R] == 0).all() and norem.pi[sm.IDX_R] == 0
    assert frag.A[sm.IDX_L, sm.IDX_W] > nom.A[sm.IDX_L, sm.IDX_W]
    rng = np.random.default_rng(3)
    s, _ = norem.sample(2000, rng)
    assert (s == sm.IDX_R).sum() == 0                     # nunca aparece REM


# ------------------------------------------------------------ métricas clínicas
def test_metricas_clinicas_de_un_hipnograma_conocido():
    # 0=W, 1=L, 2=D, 3=R : 3 min despierto, luego L L D D R R R L W W L L R R R R
    hyp = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 1, 0, 0, 1, 1, 3, 3, 3, 3])
    m = clinical.sleep_metrics(hyp, epoch_min=1.0)
    assert m["tib_min"] == 19 and m["tst_min"] == 14
    assert m["sleep_efficiency"] == pytest.approx(100 * 14 / 19)
    assert m["sol_min"] == 3 and m["waso_min"] == 2
    assert m["awakenings"] == 1 and m["awakenings_ge3"] == 0
    assert m["rem_latency_min"] == 4                       # primera época REM = índice 7, inicio del sueño = 3
    assert m["min_R"] == 7 and m["min_D"] == 2
    assert m["cycles"] == 2                                # NREM->REM(3) y NREM->REM(4)
    assert [b[2] for b in clinical.bouts(hyp)][:3] == [3, 2, 2]


def test_hipnograma_sin_sueno():
    m = clinical.sleep_metrics(np.zeros(60, dtype=int))
    assert m["tst_min"] == 0 and m["sol_min"] == 60 and np.isnan(m["rem_latency_min"]) and m["cycles"] == 0


# ------------------------------------------------------------ estadística
def test_resumen_estadistico_y_t_de_student():
    x = np.arange(1, 101, dtype=float)
    s = summarize(x)
    assert s["n"] == 100 and s["mean"] == pytest.approx(50.5) and s["min"] == 1 and s["max"] == 100
    assert s["sd"] == pytest.approx(x.std(ddof=1))
    half = s["ci_high"] - s["mean"]
    assert half == pytest.approx(1.9842169515864165 * s["sd"] / 10, rel=1e-6)   # t_{0.975, 99}


def test_kappa_f1_y_holm():
    cm = np.array([[50, 0], [0, 50]])
    assert cohen_kappa(cm) == pytest.approx(1.0) and macro_f1(cm) == pytest.approx(1.0)
    assert cohen_kappa(np.array([[25, 25], [25, 25]])) == pytest.approx(0.0)
    y = np.array([0, 1, 2, 3, 3]); yh = np.array([0, 1, 2, 2, 3])
    assert confusion_matrix(y, yh, 4).sum() == 5
    adj = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_auc_casos_extremos():
    assert auc_from_scores([2, 3, 4], [0, 1]) == 1.0
    assert auc_from_scores([0, 1], [2, 3, 4]) == 0.0
    assert auc_from_scores([1, 1], [1, 1]) == 0.5


# ------------------------------------------------------------ señales sintéticas
def test_extraccion_de_simbolos_es_consistente():
    rng = np.random.default_rng(0)
    ok = 0
    for s in range(4):
        for _ in range(40):
            _, e, o = simulate_epoch(s, rng)
            ok += int(extract_symbol(e, o) == s)
    assert ok / 160 > 0.95


# ------------------------------------------------------------ reproducibilidad de los experimentos
def test_misma_semilla_mismo_resultado_y_semilla_distinta_difiere():
    scn = ex.build_scenarios()["E1"]
    a, _ = ex.run_scenario(scn, 6, master_seed=42)
    b, _ = ex.run_scenario(scn, 6, master_seed=42)
    c, _ = ex.run_scenario(scn, 6, master_seed=43)
    pd.testing.assert_frame_equal(a, b)                    # idéntico bit a bit
    assert not np.allclose(a["acc_viterbi"], c["acc_viterbi"])


def test_replicas_independientes_y_estables_al_anadir_replicas():
    scn = ex.build_scenarios()["E1"]
    small, _ = ex.run_scenario(scn, 4, master_seed=42)
    big, _ = ex.run_scenario(scn, 8, master_seed=42)
    assert np.allclose(small["acc_viterbi"], big["acc_viterbi"].iloc[:4])   # SeedSequence.spawn
    assert big["acc_viterbi"].nunique() > 4                                 # no es la misma noche repetida


def test_escenarios_producen_metricas_validas():
    for key, scn in ex.build_scenarios().items():
        df, extra = ex.run_scenario(scn, 3, master_seed=1)
        accs = df[["acc_viterbi", "acc_posterior", "acc_baseline"]]
        assert ((accs >= 0) & (accs <= 1)).all().all(), key
        assert extra["cm_viterbi"].sum() == 3 * scn.T


def test_barrido_persistencia_mejora_a_viterbi_frente_al_baseline():
    df = ex.run_sweep("persistence", [0.5, 0.95], 20, master_seed=5)
    g = df.groupby("level")[["acc_viterbi", "acc_baseline"]].mean()
    assert g.loc[0.95, "acc_viterbi"] - g.loc[0.95, "acc_baseline"] > 0.10
    assert g.loc[0.5, "acc_viterbi"] - g.loc[0.5, "acc_baseline"] < 0.08


def test_forward_distingue_modelos_patologicos():
    nom, norem = sm.nominal_hmm(), sm.no_rem_hmm(sm.nominal_hmm())
    rng = np.random.default_rng(2)
    obs_norem = np.stack([norem.sample(480, rng)[1] for _ in range(10)])
    obs_nom = np.stack([nom.sample(480, rng)[1] for _ in range(10)])
    assert (norem.log_likelihood(obs_norem) > nom.log_likelihood(obs_norem)).all()
    assert (nom.log_likelihood(obs_nom) > norem.log_likelihood(obs_nom)).all()
