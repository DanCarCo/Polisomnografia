"""Métricas clínicas de un hipnograma y generación del informe simulado.

Un hipnograma es un vector de enteros ``h[t] in {0: Despierto, 1: Ligero,
2: Profundo, 3: REM}``, una entrada por época de ``epoch_min`` minutos.

Definiciones operativas (parametrizables):

* TIB: tiempo en cama = T · epoch_min.
* TST: tiempo total de sueño = épocas no-Despierto · epoch_min.
* Eficiencia del sueño (SE) = 100 · TST / TIB.
* Latencia de sueño (SOL): tiempo hasta la primera época no-Despierto.
* Latencia REM: desde el inicio del sueño hasta la primera época REM.
* WASO: minutos despierto después del inicio del sueño.
* Despertares: rachas de Despierto que empiezan después del inicio del sueño
  (``awakenings_ge3``: solo las de ≥ 3 min).
* Ciclo completo NREM→REM: episodio con ≥ 1 época de sueño NREM (Ligero/Profundo)
  seguido de una racha REM de ≥ ``min_rem_epochs`` épocas consecutivas.

AVISO: el informe es una salida educativa de una simulación; no es un
diagnóstico médico.
"""
from __future__ import annotations

import numpy as np

from .sleep_model import IDX_D, IDX_L, IDX_R, IDX_W, STATE_NAMES


def bouts(hyp) -> list[tuple[int, int, int]]:
    """Codificación por rachas: lista de ``(estado, inicio, longitud)``."""
    h = np.asarray(hyp)
    if len(h) == 0:
        return []
    change = np.flatnonzero(np.diff(h)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(h)]])
    return [(int(h[s]), int(s), int(e - s)) for s, e in zip(starts, ends)]


def count_cycles(hyp, min_rem_epochs: int = 3) -> int:
    seen_nrem, cycles = False, 0
    asleep = False
    for state, _, length in bouts(hyp):
        if state != IDX_W:
            asleep = True
        if not asleep:
            continue
        if state in (IDX_L, IDX_D):
            seen_nrem = True
        elif state == IDX_R and seen_nrem and length >= min_rem_epochs:
            cycles += 1
            seen_nrem = False
    return cycles


def sleep_metrics(hyp, epoch_min: float = 1.0, min_rem_epochs: int = 3) -> dict:
    h = np.asarray(hyp)
    t_len = len(h)
    tib = t_len * epoch_min
    asleep_idx = np.flatnonzero(h != IDX_W)
    tst = len(asleep_idx) * epoch_min
    onset = int(asleep_idx[0]) if len(asleep_idx) else None
    sol = onset * epoch_min if onset is not None else tib
    rem_idx = np.flatnonzero(h == IDX_R)
    rem_lat = (rem_idx[0] - onset) * epoch_min if (len(rem_idx) and onset is not None) else np.nan

    waso, n_awk, n_awk3 = 0.0, 0, 0
    if onset is not None:
        after = h[onset:]
        waso = float((after == IDX_W).sum() * epoch_min)
        for state, start, length in bouts(after):
            if state == IDX_W and start > 0:
                n_awk += 1
                n_awk3 += int(length >= 3)

    mins = {i: float((h == i).sum() * epoch_min) for i in range(4)}
    pct = lambda i: 100.0 * mins[i] / tst if tst > 0 else np.nan
    shifts = int(np.count_nonzero(np.diff(h)))
    return {
        "tib_min": float(tib), "tst_min": float(tst),
        "sleep_efficiency": 100.0 * tst / tib,
        "sol_min": float(sol), "rem_latency_min": float(rem_lat), "waso_min": waso,
        "awakenings": n_awk, "awakenings_ge3": n_awk3,
        "min_W": mins[IDX_W], "min_L": mins[IDX_L], "min_D": mins[IDX_D], "min_R": mins[IDX_R],
        "pct_W_tib": 100.0 * mins[IDX_W] / tib,
        "pct_L_tst": pct(IDX_L), "pct_D_tst": pct(IDX_D), "pct_R_tst": pct(IDX_R),
        "cycles": count_cycles(h, min_rem_epochs=min_rem_epochs),
        "shifts_per_hour": shifts / (tib / 60.0),
    }


# ------------------------------------------------------------------ referencia y alertas
FLAG_RULES = {
    # métrica: (dirección de alteración, etiqueta)
    "sleep_efficiency": ("low", "eficiencia del sueño reducida"),
    "sol_min": ("high", "latencia de sueño prolongada"),
    "waso_min": ("high", "vigilia intrasueño (WASO) aumentada"),
    "awakenings": ("high", "despertares nocturnos aumentados"),
    "pct_R_tst": ("low", "porcentaje de REM reducido"),
    "pct_D_tst": ("low", "porcentaje de sueño profundo reducido"),
    "shifts_per_hour": ("high", "alta frecuencia de cambios de etapa"),
}


def reference_stats(metric_dicts: list[dict]) -> dict:
    """Media y desviación típica por métrica a partir de noches del modelo de referencia."""
    out = {}
    for key in FLAG_RULES:
        v = np.array([m[key] for m in metric_dicts], dtype=float)
        v = v[np.isfinite(v)]
        out[key] = (float(v.mean()), float(v.std(ddof=1)))
    return out


def flag_alterations(metrics: dict, ref: dict, z_thr: float = 2.0) -> list[dict]:
    """Marca las métricas que se desvían ≥ z_thr desviaciones típicas del modelo de referencia."""
    flags = []
    for key, (direction, label) in FLAG_RULES.items():
        mu, sd = ref[key]
        if not np.isfinite(metrics[key]) or sd <= 0:
            continue
        z = (metrics[key] - mu) / sd
        if (direction == "low" and z <= -z_thr) or (direction == "high" and z >= z_thr):
            flags.append({"metric": key, "label": label, "value": metrics[key], "z": float(z)})
    return flags


def format_report(patient: str, metrics: dict, flags: list[dict], model_probs: dict | None = None,
                  truth_label: str | None = None, ref: dict | None = None) -> str:
    """Informe clínico simulado en Markdown (educativo, no diagnóstico)."""
    m = metrics
    lines = [f"# Informe de polisomnografía simulada — {patient}", ""]
    lines += ["> **Aviso:** resultado de una simulación con parámetros didácticos. "
              "No constituye un diagnóstico médico.", ""]
    lines += ["## Arquitectura del sueño (hipnograma decodificado con Viterbi)", "",
              "| Indicador | Valor |", "|---|---|",
              f"| Tiempo en cama (TIB) | {m['tib_min']:.0f} min |",
              f"| Tiempo total de sueño (TST) | {m['tst_min']:.0f} min |",
              f"| Eficiencia del sueño | {m['sleep_efficiency']:.1f} % |",
              f"| Latencia de sueño | {m['sol_min']:.0f} min |",
              f"| Latencia REM | {'—' if not np.isfinite(m['rem_latency_min']) else format(m['rem_latency_min'], '.0f') + ' min'} |",
              f"| Vigilia tras el inicio (WASO) | {m['waso_min']:.0f} min |",
              f"| Despertares (≥1 min / ≥3 min) | {m['awakenings']} / {m['awakenings_ge3']} |",
              f"| Ciclos NREM→REM completos | {m['cycles']} |",
              f"| Cambios de etapa por hora | {m['shifts_per_hour']:.1f} |", "",
              "| Etapa | Minutos | % |", "|---|---|---|",
              f"| {STATE_NAMES[IDX_W]} | {m['min_W']:.0f} | {m['pct_W_tib']:.1f} % del TIB |",
              f"| {STATE_NAMES[IDX_L]} | {m['min_L']:.0f} | {m['pct_L_tst']:.1f} % del TST |",
              f"| {STATE_NAMES[IDX_D]} | {m['min_D']:.0f} | {m['pct_D_tst']:.1f} % del TST |",
              f"| {STATE_NAMES[IDX_R]} | {m['min_R']:.0f} | {m['pct_R_tst']:.1f} % del TST |", ""]
    lines += ["## Alteraciones respecto al modelo de referencia (|z| ≥ 2)", ""]
    if flags:
        lines += [f"- **{f['label']}** (valor = {f['value']:.1f}, z = {f['z']:+.1f})" for f in flags]
    else:
        lines += ["- Sin desviaciones relevantes respecto a la distribución de referencia."]
    lines.append("")
    if model_probs:
        lines += ["## Probabilidad a posteriori de cada modelo (Forward, a priori uniforme)", "",
                  "| Modelo | P(modelo \\| observaciones) |", "|---|---|"]
        lines += [f"| {k} | {v:.4f} |" for k, v in model_probs.items()]
        best = max(model_probs, key=model_probs.get)
        lines += ["", f"**Diagnóstico probable (simulado):** {best}."]
    if truth_label:
        lines += ["", f"*Etiqueta verdadera de la simulación (solo para validar): {truth_label}.*"]
    return "\n".join(lines) + "\n"
