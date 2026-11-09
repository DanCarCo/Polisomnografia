"""Experimentos Monte Carlo del HMM de etapas del sueño.

Reproducibilidad
----------------
Una única semilla maestra (``MASTER_SEED = 42``) genera, con
``numpy.random.SeedSequence``, flujos **independientes** para cada
experimento y cada réplica (``spawn``). Así no se repite la misma semilla en
cada réplica (lo que daría réplicas idénticas) y añadir réplicas no altera las
anteriores.

Experimentos
------------
``run_scenario``            escenarios E1-E5 (N réplicas por escenario).
``run_sweep``               sensibilidad a A, B y a la persistencia a_ii.
``run_initial_conditions``  efecto de la condición inicial (pi) en la decodificación.
``run_seed_sensitivity``    variabilidad de la media entre semillas maestras.
``run_learning``            Baum-Welch con noches de entrenamiento y de prueba.
``run_pathology``           alteraciones simuladas y clasificación con Forward.
``run_simulator_convergence`` convergencia de las frecuencias a A (ley de grandes números).
``run_numerical_stability`` Viterbi lineal (plantilla) vs log-espacio.
"""
from __future__ import annotations

import itertools
import zlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from . import sleep_model as sm
from .clinical import flag_alterations, reference_stats, sleep_metrics
from .emissions import CategoricalEmission, safe_log
from .hmm_core import HMM, viterbi_linear
from .utils_stats import (auc_bootstrap_ci, auc_from_scores, bootstrap_ci, cohen_kappa, compare_groups, confusion_matrix,
                          holm_adjust, macro_f1, recall_per_class, summarize)

MASTER_SEED = 42
N_STATES = 4
CLIN_KEYS = ("tst_min", "sleep_efficiency", "sol_min", "rem_latency_min", "waso_min", "awakenings",
             "awakenings_ge3", "pct_L_tst", "pct_D_tst", "pct_R_tst", "cycles", "shifts_per_hour")


# ===================================================================== semillas y simulación
def seed_sequence(master: int, tag: str) -> np.random.SeedSequence:
    """Flujo de semillas determinista para ``tag`` (crc32 es estable entre procesos)."""
    return np.random.SeedSequence([int(master), zlib.crc32(tag.encode("utf-8"))])


def simulate_nights(truth_getter: Callable[[np.random.Generator], HMM], T: int, seeds):
    """Simula una noche por semilla hija. Devuelve ``(estados, observaciones, modelos)``."""
    states, obs, models = [], [], []
    for ss in seeds:
        rng = np.random.default_rng(ss)
        truth = truth_getter(rng)
        s, o = truth.sample(T, rng)
        states.append(s)
        obs.append(o)
        models.append(truth)
    return np.stack(states), np.stack(obs), models


def memoryless_decode(h: HMM, obs) -> np.ndarray:
    """Clasificador Bayes sin memoria temporal (línea base sin HMM).

    ``argmax_j [ log pi*_j + log b_j(o_t) ]`` con pi* la distribución estacionaria:
    usa B y las frecuencias a priori de las etapas, pero ignora la dinámica A.
    """
    log_b = h.emission.log_likelihood(np.asarray(obs))
    return (log_b + safe_log(h.stationary_distribution())).argmax(axis=-1)


# ===================================================================== escenarios
@dataclass
class Scenario:
    key: str
    name: str
    description: str
    decoder: HMM                                    # modelo conocido por el decodificador
    truth_factory: Callable | None = None           # generador de la verdad (None = decoder)
    random_truth: bool = False                      # ¿la verdad cambia en cada réplica?
    T: int = 480
    epoch_min: float = 1.0


def build_scenarios(kappa_A: float = 20.0, eps_B: float = 0.30,
                    persistence=(0.80, 0.95, 0.95, 0.95)) -> dict[str, Scenario]:
    """Cinco escenarios definidos a priori (antes de ver resultados)."""
    nom = sm.nominal_hmm()
    truth_B = sm.nominal_hmm(B=sm.blend_with_uniform(sm.B_NOM, eps_B))
    cons = sm.nominal_hmm(A=sm.scale_persistence(sm.A_NOM, persistence))
    scs = [
        Scenario("E1", "Nominal",
                 "Matrices π, A y B del enunciado; el decodificador conoce el modelo verdadero.", nom),
        Scenario("E2", "Transición perturbada",
                 f"A verdadera ~ Dirichlet({kappa_A:g}·A) por réplica; el decodificador usa la A nominal.",
                 nom, truth_factory=lambda rng: sm.nominal_hmm(A=sm.dirichlet_perturb(sm.A_NOM, kappa_A, rng)),
                 random_truth=True),
        Scenario("E3", "Emisión degradada",
                 f"B verdadera = (1-{eps_B:g})·B + {eps_B:g}·U (mayor entropía); decodificador con B nominal.",
                 nom, truth_factory=lambda rng: truth_B),
        Scenario("E4", "Emisión continua",
                 "Rasgos continuos: frecuencia dominante ~ Normal y ráfagas REM ~ Poisson, con solapamiento.",
                 sm.gaussian_poisson_hmm()),
        Scenario("E5", "Sueño consolidado",
                 f"Mayor persistencia a_ii = {list(persistence)}; mismo patrón relativo de transiciones y B nominal.",
                 cons),
    ]
    return {s.key: s for s in scs}


def _night_metrics(s_true, s_pred) -> tuple[dict, np.ndarray]:
    cm = confusion_matrix(s_true, s_pred, N_STATES)
    rec = recall_per_class(cm)
    return ({"acc": float(np.trace(cm) / cm.sum()), "kappa": cohen_kappa(cm), "f1": macro_f1(cm),
             **{f"recall_{k}": rec[i] for i, k in enumerate(("W", "L", "D", "R"))}}, cm)


def run_scenario(scn: Scenario, n_reps: int, master_seed: int = MASTER_SEED):
    """Simula y decodifica ``n_reps`` noches. Devuelve ``(DataFrame por réplica, extras)``."""
    seeds = seed_sequence(master_seed, f"scenario:{scn.key}").spawn(n_reps)
    getter = scn.truth_factory if scn.truth_factory is not None else (lambda rng: scn.decoder)
    states, obs, truths = simulate_nights(getter, scn.T, seeds)

    vit, _ = scn.decoder.viterbi(obs)
    fb = scn.decoder.forward_backward(obs)          # una sola pasada: posteriores y log-verosimilitud
    post = fb.gamma.argmax(axis=2)
    loglik = fb.loglik
    base = memoryless_decode(scn.decoder, obs)
    oracle = None   # techo de exactitud: decodificador posterior con el modelo VERDADERO
    if scn.truth_factory is not None:
        if scn.random_truth:
            oracle = np.stack([truths[k].posterior_decode(obs[k])[0] for k in range(n_reps)])
        else:
            oracle = truths[0].posterior_decode(obs)[0]

    rows, cm_total, cm_base = [], np.zeros((N_STATES, N_STATES), dtype=np.int64), np.zeros((N_STATES, N_STATES), dtype=np.int64)
    for k in range(n_reps):
        mv, cm = _night_metrics(states[k], vit[k])
        mb, cmb = _night_metrics(states[k], base[k])
        cm_total += cm
        cm_base += cmb
        row = {"scenario": scn.key, "rep": k,
               "acc_viterbi": mv["acc"], "kappa_viterbi": mv["kappa"], "f1_viterbi": mv["f1"],
               "acc_posterior": float(np.mean(post[k] == states[k])),
               "acc_baseline": mb["acc"], "kappa_baseline": mb["kappa"],
               "acc_oracle": float(np.mean(oracle[k] == states[k])) if oracle is not None else np.nan,
               "gain_vs_baseline": mv["acc"] - mb["acc"],
               **{f"recall_{x}_viterbi": mv[f"recall_{x}"] for x in ("W", "L", "D", "R")},
               "loglik_per_epoch": float(loglik[k] / scn.T)}
        mt = sleep_metrics(states[k], scn.epoch_min)
        md = sleep_metrics(vit[k], scn.epoch_min)
        for key in CLIN_KEYS:
            row[f"true_{key}"] = mt[key]
            row[f"vit_{key}"] = md[key]
        rows.append(row)

    extras = {"cm_viterbi": cm_total, "cm_baseline": cm_base, "states": states, "obs": obs, "vit": vit,
              "post": post, "scenario": scn}
    return pd.DataFrame(rows), extras


def summarize_table(df: pd.DataFrame, group_cols, metric_cols, boot_rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Tabla larga: n, media, DE, mín, máx, mediana, EE, IC95 t y IC95 bootstrap por grupo y métrica."""
    rows = []
    for g, sub in df.groupby(list(group_cols), sort=False):
        g = g if isinstance(g, tuple) else (g,)
        for m in metric_cols:
            vals = sub[m].to_numpy(dtype=float)
            st = summarize(vals)
            lo = hi = np.nan
            if boot_rng is not None and st["n"] > 1 and st["sd"] > 0:
                lo, hi = bootstrap_ci(vals, boot_rng)
            rows.append({**dict(zip(group_cols, g)), "metric": m, **st, "boot_ci_low": lo, "boot_ci_high": hi})
    return pd.DataFrame(rows)


def compare_scenarios(df: pd.DataFrame, ref: str = "E1", metrics=("acc_viterbi", "kappa_viterbi")) -> pd.DataFrame:
    """Cada escenario frente a ``ref``: Welch, Mann-Whitney, d de Cohen y corrección de Holm."""
    rows = []
    for m in metrics:
        x_ref = df.loc[df.scenario == ref, m].to_numpy()
        sub = []
        for key in [k for k in df.scenario.unique() if k != ref]:
            x = df.loc[df.scenario == key, m].to_numpy()
            sub.append({"metric": m, "scenario": key, "reference": ref, **compare_groups(x, x_ref)})
        adj = holm_adjust([r["p_welch"] for r in sub])
        for r, p in zip(sub, adj):
            r["p_welch_holm"] = float(p)
        rows += sub
    kw = {m: stats.kruskal(*[g[m].to_numpy() for _, g in df.groupby("scenario")]) for m in metrics}
    out = pd.DataFrame(rows)
    out["kruskal_H_all"] = out["metric"].map(lambda m: float(kw[m].statistic))
    out["kruskal_p_all"] = out["metric"].map(lambda m: float(kw[m].pvalue))
    return out


def compare_decoders(df: pd.DataFrame) -> pd.DataFrame:
    """Comparación pareada (mismas noches) entre decodificadores dentro de cada escenario."""
    pairs = [("acc_viterbi", "acc_baseline"), ("acc_posterior", "acc_viterbi"), ("acc_posterior", "acc_oracle")]
    rows = []
    for key, sub in df.groupby("scenario", sort=False):
        for a, b in pairs:
            if sub[b].isna().all():
                continue
            d = (sub[a] - sub[b]).to_numpy()
            st = summarize(d)
            if np.allclose(d, 0):
                p_t = p_w = 1.0
            else:
                p_t = float(stats.ttest_1samp(d, 0.0).pvalue)
                p_w = float(stats.wilcoxon(d).pvalue)
            rows.append({"scenario": key, "a": a, "b": b, "mean_diff": st["mean"], "sd_diff": st["sd"],
                         "ci_low": st["ci_low"], "ci_high": st["ci_high"], "p_paired_t": p_t, "p_wilcoxon": p_w})
    return pd.DataFrame(rows)


# ===================================================================== barridos de sensibilidad
def run_sweep(factor: str, levels, n_reps: int, master_seed: int = MASTER_SEED, T: int = 480) -> pd.DataFrame:
    """Exactitud de los decodificadores al variar un factor.

    ``A_mix``/``B_mix``: la verdad usa (1-eps)·M + eps·U y el decodificador conoce la M nominal
    (desajuste del modelo); se reporta además el decodificador oráculo con la verdad.
    ``persistence``: la verdad y el decodificador comparten a_ii = level (modelo ajustado).
    """
    nom = sm.nominal_hmm()
    rows = []
    for lvl in levels:
        if factor == "A_mix":
            truth, decoder = sm.nominal_hmm(A=sm.blend_with_uniform(sm.A_NOM, lvl)), nom
        elif factor == "B_mix":
            truth, decoder = sm.nominal_hmm(B=sm.blend_with_uniform(sm.B_NOM, lvl)), nom
        elif factor == "persistence":
            truth = sm.nominal_hmm(A=sm.scale_persistence(sm.A_NOM, lvl))
            decoder = truth
        else:
            raise ValueError(factor)
        seeds = seed_sequence(master_seed, f"sweep:{factor}:{lvl:.4f}").spawn(n_reps)
        s, o, _ = simulate_nights(lambda rng: truth, T, seeds)
        vit = decoder.viterbi(o)[0]
        post = decoder.posterior_decode(o)[0]
        base = memoryless_decode(decoder, o)
        orc = truth.posterior_decode(o)[0]     # techo: posterior con el modelo verdadero
        for k in range(n_reps):
            rows.append({"factor": factor, "level": float(lvl), "rep": k,
                         "acc_viterbi": float(np.mean(vit[k] == s[k])),
                         "acc_posterior": float(np.mean(post[k] == s[k])),
                         "acc_baseline": float(np.mean(base[k] == s[k])),
                         "acc_oracle": float(np.mean(orc[k] == s[k]))})
    return pd.DataFrame(rows)


def run_initial_conditions(n_reps: int, master_seed: int = MASTER_SEED, T: int = 480, first: int = 10) -> pd.DataFrame:
    """Condición inicial: la noche empieza forzosamente en cada etapa y se decodifica con
    tres supuestos distintos sobre pi (nominal, uniforme, estacionaria)."""
    nom = sm.nominal_hmm()
    decoders = {"pi_nominal": nom,
                "pi_uniforme": sm.nominal_hmm(pi=np.full(N_STATES, 1 / N_STATES)),
                "pi_estacionaria": sm.nominal_hmm(pi=nom.stationary_distribution())}
    rows = []
    for s0 in range(N_STATES):
        truth = sm.nominal_hmm(pi=np.eye(N_STATES)[s0])
        seeds = seed_sequence(master_seed, f"initial:{s0}").spawn(n_reps)
        s, o, _ = simulate_nights(lambda rng: truth, T, seeds)
        for name, dec in decoders.items():
            vit = dec.viterbi(o)[0]
            ok = vit == s
            for k in range(n_reps):
                rows.append({"estado_inicial": sm.STATE_NAMES[s0], "decodificador": name, "rep": k,
                             "acc_primeras": float(ok[k, :first].mean()), "acc_resto": float(ok[k, first:].mean()),
                             "acc_total": float(ok[k].mean())})
    return pd.DataFrame(rows)


def run_seed_sensitivity(n_seeds: int, n_reps: int, T: int = 480, first_seed: int = MASTER_SEED) -> pd.DataFrame:
    """Repite E1 con ``n_seeds`` semillas maestras distintas (42, 43, ...)."""
    scn = build_scenarios()["E1"]
    rows = []
    for ms in range(first_seed, first_seed + n_seeds):
        df, _ = run_scenario(scn, n_reps, master_seed=ms)
        st = summarize(df["acc_viterbi"])
        rows.append({"master_seed": ms, "n": st["n"], "mean_acc_viterbi": st["mean"], "sd": st["sd"],
                     "ci_low": st["ci_low"], "ci_high": st["ci_high"]})
    return pd.DataFrame(rows)


def chain_properties() -> pd.DataFrame:
    """Propiedades teóricas de la cadena nominal (contraste con la simulación)."""
    A = sm.A_NOM
    h = sm.nominal_hmm()
    pi_star = h.stationary_distribution()
    eig = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
    rows = []
    for i, name in enumerate(sm.STATE_NAMES):
        rows.append({"estado": name, "pi_inicial": sm.PI_NOM[i], "a_ii": A[i, i],
                     "duracion_media_epocas": 1.0 / (1.0 - A[i, i]),
                     "distribucion_estacionaria": pi_star[i]})
    df = pd.DataFrame(rows)
    df["abs_lambda2"] = eig[1]
    df["tiempo_relajacion_epocas"] = 1.0 / (1.0 - eig[1])
    return df


# ===================================================================== aprendizaje (Baum-Welch)
def _permute_hmm(h: HMM, p) -> HMM:
    """Reordena los estados: el nuevo estado j es el antiguo p[j]."""
    p = list(p)
    return HMM(h.pi[p], h.A[np.ix_(p, p)], CategoricalEmission(h.B[p]), h.state_names, h.obs_names)


def align_to_truth(learned: HMM, truth: HMM) -> tuple[HMM, tuple]:
    """Alinea los estados aprendidos con los verdaderos por mínima distancia L1 entre filas de B
    (usa solo parámetros, no etiquetas de las noches de prueba)."""
    best, best_cost = None, np.inf
    for p in itertools.permutations(range(N_STATES)):
        cost = np.abs(learned.B[list(p)] - truth.B).sum()
        if cost < best_cost:
            best, best_cost = p, cost
    return _permute_hmm(learned, best), best


def _random_init(rng: np.random.Generator, n_symbols: int = 4) -> HMM:
    """Inicialización aleatoria (sin conocimiento previo): pi y filas de A y B ~ Dirichlet(1,...,1)."""
    return HMM(rng.dirichlet(np.ones(N_STATES)), rng.dirichlet(np.ones(N_STATES), size=N_STATES),
               CategoricalEmission(rng.dirichlet(np.ones(n_symbols), size=N_STATES)),
               sm.STATE_NAMES, sm.OBS_NAMES)


def _informed_init(rng: np.random.Generator, kappa: float = 10.0) -> HMM:
    """Inicialización informada: un a priori clínico aproximado, sin ceros duros.

    Cada fila ~ Dirichlet(kappa * fila_nominal + 0.1): media cercana al modelo nominal pero con
    ruido y sin imponer la estructura (no fija a priori ningún cero ni el orden de los estados).
    """
    dirich = lambda M: np.vstack([rng.dirichlet(kappa * row + 0.1) for row in M])
    return HMM(rng.dirichlet(kappa * sm.PI_NOM + 0.1), dirich(sm.A_NOM), CategoricalEmission(dirich(sm.B_NOM)),
               sm.STATE_NAMES, sm.OBS_NAMES)


INIT_MODES = {"aleatoria": _random_init, "informada": _informed_init}


def run_learning(n_reps: int, n_train_list=(1, 5, 20), n_test: int = 20, n_restarts: int = 3,
                 max_iter: int = 300, tol: float = 1e-6, master_seed: int = MASTER_SEED, T: int = 480,
                 init_modes=("aleatoria", "informada")):
    """Baum-Welch con dos tipos de inicialización; evaluación en noches de prueba no vistas.

    Las noches de entrenamiento y de prueba son disjuntas (partición por noche/secuencia). Cada
    reinicio ``q`` usa la misma inicialización para todos los tamaños de entrenamiento, y los
    estados aprendidos se alinean con los verdaderos solo con los parámetros (distancia entre B).

    Devuelve ``(DataFrame por réplica/tamaño/inicialización, historial de log-verosimilitud,
    modelos de la réplica 0 indexados por (inicialización, noches))``.
    """
    truth = sm.nominal_hmm()
    rows, hist_rows, example_models = [], [], {}
    for r in range(n_reps):
        seeds = seed_sequence(master_seed, f"learn:{r}").spawn(max(n_train_list) + n_test)
        s_all, o_all, _ = simulate_nights(lambda rng: truth, T, seeds)
        o_test, s_test = o_all[-n_test:], s_all[-n_test:]
        ll_true_test = float(np.mean(truth.log_likelihood(o_test))) / T
        acc_true = float(np.mean(truth.viterbi(o_test)[0] == s_test))
        acc_base = float(np.mean(memoryless_decode(truth, o_test) == s_test))
        for mode in init_modes:
            make_init = INIT_MODES[mode]
            for n_train in n_train_list:
                o_train = o_all[:n_train]
                best, best_ll, best_iters = None, -np.inf, 0
                for q in range(n_restarts):
                    m = make_init(np.random.default_rng(seed_sequence(master_seed, f"learn:{r}:init:{mode}:{q}")))
                    hist = m.fit(o_train, n_iter=max_iter, tol=tol)
                    final = float(m.log_likelihood(o_train).sum())
                    if r == 0 and n_train == max(n_train_list):
                        hist_rows += [{"init": mode, "n_train": n_train, "restart": q, "iter": i, "loglik": v}
                                      for i, v in enumerate(hist)]
                    if final > best_ll:
                        best, best_ll, best_iters = m, final, len(hist)
                aligned, perm = align_to_truth(best, truth)
                if r == 0:
                    example_models[(mode, n_train)] = aligned
                acc_l = float(np.mean(aligned.viterbi(o_test)[0] == s_test))
                rows.append({
                    "init": mode, "rep": r, "n_train": n_train, "iterations": best_iters,
                    "mae_A": float(np.abs(aligned.A - truth.A).mean()), "mae_B": float(np.abs(aligned.B - truth.B).mean()),
                    "max_err_A": float(np.abs(aligned.A - truth.A).max()), "max_err_B": float(np.abs(aligned.B - truth.B).max()),
                    "mae_pi": float(np.abs(aligned.pi - truth.pi).mean()),
                    "loglik_test_per_epoch_learned": float(np.mean(aligned.log_likelihood(o_test))) / T,
                    "loglik_test_per_epoch_true": ll_true_test,
                    "acc_test_learned": acc_l, "acc_test_true": acc_true, "acc_test_baseline": acc_base,
                    "perm": "".join(map(str, perm)),
                })
    return pd.DataFrame(rows), pd.DataFrame(hist_rows), example_models


# ===================================================================== patologías simuladas
PATHOLOGY_LABELS = ("Normal", "Sueño fragmentado", "Sin REM")


def pathology_models(frag_delta: float = 0.15) -> dict[str, HMM]:
    nom = sm.nominal_hmm()
    return {"Normal": nom, "Sueño fragmentado": sm.fragmented_hmm(nom, frag_delta), "Sin REM": sm.no_rem_hmm(nom)}


def run_pathology(n_per_class: int, n_reference: int, master_seed: int = MASTER_SEED, T: int = 480):
    """Clasificación de noches por máxima verosimilitud (Forward) entre tres modelos y alertas por z-score.

    Devuelve ``(noches, matriz de confusión, resumen por clase, estadísticos de referencia)``.
    """
    models = pathology_models()
    nom = models["Normal"]
    # Referencia independiente (flujo de semillas distinto) para los z-scores.
    s_ref, o_ref, _ = simulate_nights(lambda rng: nom, T, seed_sequence(master_seed, "patho:reference").spawn(n_reference))
    vit_ref = nom.viterbi(o_ref)[0]
    ref = reference_stats([sleep_metrics(v) for v in vit_ref])

    rows = []
    names = list(models)
    for label in names:
        truth = models[label]
        s, o, _ = simulate_nights(lambda rng: truth, T, seed_sequence(master_seed, f"patho:{label}").spawn(n_per_class))
        vit = nom.viterbi(o)[0]                         # herramienta de cribado: solo conoce el modelo normal
        ll = np.stack([models[n].log_likelihood(o) for n in names], axis=1)     # (n, 3)
        pred = ll.argmax(axis=1)
        for k in range(n_per_class):
            met = sleep_metrics(vit[k])
            flags = flag_alterations(met, ref)
            rows.append({"clase_real": label, "rep": k,
                         **{f"ll_{n}": ll[k, i] for i, n in enumerate(names)},
                         "clase_predicha_forward": names[pred[k]],
                         "n_alertas_z": len(flags), "alerta_REM_bajo": any(f["metric"] == "pct_R_tst" for f in flags),
                         "alerta_despertares": any(f["metric"] in ("awakenings", "waso_min", "sleep_efficiency") for f in flags),
                         **{f"vit_{key}": met[key] for key in ("sleep_efficiency", "waso_min", "awakenings", "pct_R_tst",
                                                              "pct_D_tst", "shifts_per_hour", "cycles")},
                         "score_normalidad": ll[k, 0] - max(ll[k, 1], ll[k, 2])})
    df = pd.DataFrame(rows)

    cm = pd.crosstab(df["clase_real"], df["clase_predicha_forward"]).reindex(index=names, columns=names, fill_value=0)
    normal = df.loc[df.clase_real == "Normal", "score_normalidad"]
    summ = []
    for label in names:
        sub = df[df.clase_real == label]
        sens = float((sub.clase_predicha_forward == label).mean())
        others = df[df.clase_real != label]
        spec = float((others.clase_predicha_forward != label).mean())
        auc = (auc_from_scores(normal, df.loc[df.clase_real == label, "score_normalidad"]) if label != "Normal" else np.nan)
        summ.append({"clase": label, "sensibilidad_forward": sens, "especificidad_forward": spec,
                     "auc_normal_vs_clase": auc,
                     "alerta_REM_bajo_%": 100 * float(sub.alerta_REM_bajo.mean()),
                     "alerta_despertares_%": 100 * float(sub.alerta_despertares.mean())})
    return df, cm, pd.DataFrame(summ), ref


def build_example_reports(master_seed: int = MASTER_SEED, T: int = 480, ref: dict | None = None):
    """Tres noches de ejemplo (una por clase) con su informe clínico simulado."""
    from .clinical import format_report
    models = pathology_models()
    nom = models["Normal"]
    out = {}
    for i, label in enumerate(PATHOLOGY_LABELS):
        rng = np.random.default_rng(seed_sequence(master_seed, f"report:{label}"))
        s, o = models[label].sample(T, rng)
        vit = nom.viterbi(o)[0]
        met = sleep_metrics(vit)
        ll = np.array([models[n].log_likelihood(o) for n in PATHOLOGY_LABELS])
        prob = np.exp(ll - ll.max())
        prob /= prob.sum()
        flags = flag_alterations(met, ref) if ref else []
        out[label] = format_report(f"Paciente simulado {i + 1}", met, flags,
                                   dict(zip(PATHOLOGY_LABELS, prob)), truth_label=label, ref=ref)
    return out


# ===================================================================== validación del simulador
def run_simulator_convergence(n_chains: int, lengths, master_seed: int = MASTER_SEED) -> pd.DataFrame:
    """Error de las frecuencias empíricas de transición/estacionarias frente a los valores teóricos."""
    nom = sm.nominal_hmm()
    pi_star = nom.stationary_distribution()
    rows = []
    longest = int(max(lengths))
    for c, ss in enumerate(seed_sequence(master_seed, "convergence").spawn(n_chains)):
        s, _ = nom.sample(longest, np.random.default_rng(ss))
        for T in lengths:
            sub = s[:T]
            counts = np.zeros((N_STATES, N_STATES))
            np.add.at(counts, (sub[:-1], sub[1:]), 1)
            tot = counts.sum(axis=1, keepdims=True)
            est = np.where(tot > 0, counts / np.maximum(tot, 1), 0.0)
            freq = np.bincount(sub, minlength=N_STATES) / T
            rows.append({"chain": c, "T": int(T), "frobenius_A": float(np.linalg.norm(est - nom.A)),
                         "max_abs_A": float(np.abs(est - nom.A).max()),
                         "tv_estacionaria": float(0.5 * np.abs(freq - pi_star).sum()),
                         "conteo_REM_a_Profundo": int(counts[sm.IDX_R, sm.IDX_D])})
    return pd.DataFrame(rows)


def run_numerical_stability(lengths, n_reps: int, master_seed: int = MASTER_SEED) -> pd.DataFrame:
    """Viterbi lineal (listado de la plantilla) frente a Viterbi en log-espacio.

    ``empate`` = las dos rutas difieren pero tienen la misma probabilidad conjunta (a 1e-9 en log):
    el *argmax* no es único y el desempate depende del redondeo, no de un error del algoritmo.
    """
    nom = sm.nominal_hmm()
    rows = []
    for T in lengths:
        seeds = seed_sequence(master_seed, f"stability:{T}").spawn(n_reps)
        s, o, _ = simulate_nights(lambda rng: nom, int(T), seeds)
        vit_log = nom.viterbi(o)[0]
        for k in range(n_reps):
            q_lin, p_lin = viterbi_linear(o[k], nom.A, nom.B, nom.pi)
            same = bool(np.array_equal(q_lin, vit_log[k]))
            tie = False
            if not same and p_lin > 0:
                tie = abs(nom.path_log_prob(o[k], q_lin) - nom.path_log_prob(o[k], vit_log[k])) < 1e-9
            rows.append({"T": int(T), "rep": k, "underflow_lineal": bool(p_lin == 0.0), "prob_lineal": p_lin,
                         "coincide_con_log": same, "diferencia_por_empate": tie,
                         "acc_lineal": float(np.mean(q_lin == s[k])), "acc_log": float(np.mean(vit_log[k] == s[k]))})
    return pd.DataFrame(rows)


def run_pathology_severity(n: int, master_seed: int = MASTER_SEED, T: int = 480, n_boot: int = 500,
                           frag_levels=(0.02, 0.04, 0.06, 0.08, 0.10, 0.15),
                           rem_levels=(0.75, 0.50, 0.25, 0.10, 0.0)) -> pd.DataFrame:
    """Poder de detección frente a la severidad de la alteración (AUC con IC95 % bootstrap).

    El puntaje es la razón de verosimilitudes Forward entre el modelo alterado *fijo* (delta = 0.15;
    sin REM) y el modelo normal; la verdad usa distintas severidades. Incluye un control nulo
    (otra muestra de noches normales): su AUC debe ser ≈ 0.5.
    """
    models = pathology_models()
    nom, alt = models["Normal"], {"fragmentacion": models["Sueño fragmentado"], "rem": models["Sin REM"]}

    def scores(truth: HMM, tag: str):
        s, o, _ = simulate_nights(lambda rng: truth, T, seed_sequence(master_seed, f"sev:{tag}").spawn(n))
        ll_nom = nom.log_likelihood(o)
        return {k: m.log_likelihood(o) - ll_nom for k, m in alt.items()}

    base = scores(nom, "normal")
    boot = np.random.default_rng(seed_sequence(master_seed, "sev:bootstrap"))
    rows = []

    def add(family, level, sc, key):
        lo, hi = auc_bootstrap_ci(sc[key], base[key], boot, n_boot)
        rows.append({"familia": family, "severidad": float(level), "auc": auc_from_scores(sc[key], base[key]),
                     "auc_ci_low": lo, "auc_ci_high": hi, "n": n})

    add("fragmentacion", 0.0, scores(nom, "null_frag"), "fragmentacion")        # control nulo
    for d in frag_levels:
        add("fragmentacion", d, scores(sm.fragmented_hmm(nom, d), f"frag:{d}"), "fragmentacion")
    add("rem", 1.0, scores(nom, "null_rem"), "rem")                              # control nulo
    for f in rem_levels:
        add("rem", f, scores(sm.rem_scaled_hmm(nom, f), f"rem:{f}"), "rem")
    return pd.DataFrame(rows)


def representative_night(df_e1: pd.DataFrame, extras_e1: dict) -> int:
    """Índice de la réplica de E1 cuya exactitud está más cerca de la mediana (sin cherry-picking)."""
    acc = df_e1["acc_viterbi"].to_numpy()
    return int(np.argmin(np.abs(acc - np.median(acc))))
