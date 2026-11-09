"""Métricas de clasificación y estadística descriptiva/inferencial.

Intervalos de confianza
-----------------------
* IC95 % de la media con la distribución *t de Student* (n-1 g.l.):
  ``x̄ ± t_{0.975, n-1} · s/√n``. Para n = 100 equivale a 1.984 (no 1.96).
* IC95 % *bootstrap percentil* como contraste no paramétrico.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


# ------------------------------------------------------------------ clasificación
def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Matriz de confusión: filas = estado real, columnas = estado estimado."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()), 1)
    return cm


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def cohen_kappa(cm: np.ndarray) -> float:
    """κ de Cohen (acuerdo corregido por azar; métrica estándar en estadificación del sueño)."""
    total = cm.sum()
    po = np.trace(cm) / total
    pe = float((cm.sum(axis=0) * cm.sum(axis=1)).sum()) / total ** 2
    return float((po - pe) / (1.0 - pe)) if pe < 1.0 else 1.0


def recall_per_class(cm: np.ndarray) -> np.ndarray:
    """Sensibilidad (recall) por clase; NaN si la clase no aparece."""
    tot = cm.sum(axis=1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, np.diag(cm) / tot, np.nan)


def macro_f1(cm: np.ndarray) -> float:
    """F1 macro promediado solo sobre las clases presentes en la verdad."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, np.nan)
    present = cm.sum(axis=1) > 0
    return float(np.nanmean(f1[present]))


# ------------------------------------------------------------------ descriptiva
def summarize(x, conf: float = 0.95) -> dict:
    """n, media, DE (ddof=1), mín, máx, mediana, error estándar e IC t de Student."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {k: np.nan for k in ("n", "mean", "sd", "min", "max", "median", "se", "ci_low", "ci_high")}
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    tcrit = float(stats.t.ppf(0.5 + conf / 2, n - 1)) if n > 1 else 0.0
    return {"n": n, "mean": mean, "sd": sd, "min": float(x.min()), "max": float(x.max()),
            "median": float(np.median(x)), "se": float(se),
            "ci_low": mean - tcrit * se, "ci_high": mean + tcrit * se}


def bootstrap_ci(x, rng: np.random.Generator, n_boot: int = 5000, conf: float = 0.95):
    """IC bootstrap percentil de la media."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * (1 - conf) / 2, 100 * (1 + conf) / 2])
    return float(lo), float(hi)


# ------------------------------------------------------------------ inferencial
def cohens_d(x, y) -> float:
    """d de Cohen con desviación típica combinada (grupos independientes)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return float((x.mean() - y.mean()) / sp) if sp > 0 else float("nan")


def holm_adjust(pvals) -> np.ndarray:
    """Corrección de Holm-Bonferroni (control de la tasa de error por familia)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def compare_groups(x, y) -> dict:
    """Diferencia de medias con IC95 % de Welch, prueba t de Welch, Mann-Whitney y d de Cohen."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    t, p_t = stats.ttest_ind(x, y, equal_var=False)
    u, p_u = stats.mannwhitneyu(x, y, alternative="two-sided")
    diff = float(x.mean() - y.mean())
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = np.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1)) if se > 0 else len(x) + len(y) - 2
    tcrit = stats.t.ppf(0.975, df)
    return {"mean_diff": diff, "diff_ci_low": float(diff - tcrit * se), "diff_ci_high": float(diff + tcrit * se),
            "welch_t": float(t), "p_welch": float(p_t), "p_mannwhitney": float(p_u), "cohens_d": cohens_d(x, y)}


def auc_from_scores(pos, neg) -> float:
    """Área bajo la curva ROC = P(score_pos > score_neg) + 0.5 P(empate) (equivale a U/(n1·n0))."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def auc_bootstrap_ci(pos, neg, rng: np.random.Generator, n_boot: int = 1000, conf: float = 0.95):
    """IC bootstrap percentil del AUC (remuestreo independiente de ambos grupos)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        vals[b] = auc_from_scores(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))])
    lo, hi = np.percentile(vals, [100 * (1 - conf) / 2, 100 * (1 + conf) / 2])
    return float(lo), float(hi)
