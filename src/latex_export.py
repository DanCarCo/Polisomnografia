"""Exporta tablas y macros de LaTeX a partir de los CSV de ``results/``.

Ninguna cifra del artículo se transcribe a mano: el texto usa macros generadas
aquí (``\\R{E1}{acc-viterbi}{mean}``, ``\\K{nombre}``, ...) y las tablas se
construyen directamente desde los resultados. Uso::

    python -m src.latex_export            # lee results/, escribe paper/auto/
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCEN_TEX = {"E1": "E1 Nominal", "E2": "E2 Transición perturbada", "E3": "E3 Emisión degradada",
            "E4": "E4 Emisión continua", "E5": "E5 Sueño consolidado"}
PCT_METRICS = re.compile(r"^(acc|recall|f1|gain|true_pct|vit_pct|true_sleep|vit_sleep)")


# ------------------------------------------------------------------ formato
def esc(s) -> str:
    """Escapa caracteres especiales de LaTeX."""
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def num(x, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    out = f"{x:.{nd}f}"
    if out.startswith("-"):
        out = out[1:] if float(out) == 0 else r"\ensuremath{-}" + out[1:]
    return out


def pval(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def ci(lo, hi, nd: int = 3) -> str:
    return f"[{num(lo, nd)}, {num(hi, nd)}]"


def _key(part) -> str:
    if isinstance(part, float):
        part = f"{part:g}"
    s = str(part).replace("_", "-").replace(" ", "").replace("ñ", "n").replace("á", "a").replace("é", "e") \
        .replace("í", "i").replace("ó", "o").replace("ú", "u").replace("Á", "A")
    return re.sub(r"[^A-Za-z0-9.\-]", "", s)


class Macros:
    """Acumula definiciones ``\\csname`` con claves normalizadas."""

    def __init__(self):
        self.lines: list[str] = []
        self.names: set[str] = set()

    def add(self, prefix: str, *parts, value: str) -> None:
        name = "@".join([prefix, *map(_key, parts)])
        self.names.add(name)
        self.lines.append(rf"\expandafter\def\csname {name}\endcsname{{{value}}}")

    def summary(self, prefix: str, df: pd.DataFrame, group_cols: list[str]) -> None:
        """Por fila: estadístico crudo (4 decimales) y variantes ``s1`` (1 decimal), ``s3`` (3 decimales)
        y ``spct`` (porcentaje con 1 decimal, solo para métricas de proporción)."""
        for _, r in df.iterrows():
            if pd.isna(r["n"]) or pd.isna(r["mean"]):      # métrica no aplicable al escenario (p. ej. techo)
                continue
            parts = [r[c] for c in group_cols] + [r["metric"]]
            self.add(prefix, *parts, "n", value=str(int(r["n"])))
            for st in ("mean", "sd", "min", "max", "ci_low", "ci_high"):
                v = float(r[st])
                self.add(prefix, *parts, st, value=num(v, 4))
                self.add(prefix, *parts, f"{st}3", value=num(v, 3))
                if st in ("mean", "sd"):
                    self.add(prefix, *parts, f"{st}1", value=num(v, 1))
                if PCT_METRICS.match(str(r["metric"])):
                    self.add(prefix, *parts, f"{st}pct", value=num(100 * v, 1))

    def dump(self, path: Path) -> None:
        head = ("% Archivo GENERADO por src/latex_export.py a partir de results/*.csv. No editar a mano.\n"
                "\\newcommand{\\R}[3]{\\csname R@#1@#2@#3\\endcsname}\n"
                "\\newcommand{\\Sw}[4]{\\csname S@#1@#2@#3@#4\\endcsname}\n"
                "\\newcommand{\\Lr}[4]{\\csname L@#1@#2@#3@#4\\endcsname}\n"
                "\\newcommand{\\Ic}[4]{\\csname I@#1@#2@#3@#4\\endcsname}\n"
                "\\newcommand{\\K}[1]{\\csname K@#1\\endcsname}\n")
        path.write_text(head + "\n".join(self.lines) + "\n", encoding="utf-8", newline="\n")


def _table(path: Path, header: list[str], rows: list[list[str]], colspec: str, small: bool = True) -> None:
    out = ["% Tabla GENERADA por src/latex_export.py. No editar a mano.", r"\begin{tabular}{" + colspec + "}", r"\toprule",
           " & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\", r"\midrule"]
    out += [" & ".join(r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")


def write_matrices(path: Path) -> None:
    """π, A y B del modelo nominal como ecuaciones. Los decimales van dentro de \\text{} porque
    babel-spanish convierte el punto decimal en coma en modo matemático."""
    from . import sleep_model as sm

    def mat(M, name):
        rows = [" & ".join(rf"\text{{{x:.2f}}}" for x in row) for row in M]
        return "\\begin{equation}\n\\mathbf{" + name + "}=\\begin{pmatrix}\n" + " \\\\\n".join(rows) + "\n\\end{pmatrix}\n\\end{equation}"

    pi = " & ".join(rf"\text{{{x:.2f}}}" for x in sm.PI_NOM)
    out = ["% Matrices GENERADAS por src/latex_export.py desde src/sleep_model.py. No editar a mano.",
           "\\begin{equation}\n\\boldsymbol{\\pi}=\\begin{pmatrix} " + pi + " \\end{pmatrix}\n\\end{equation}",
           mat(sm.A_NOM, "A"), mat(sm.B_NOM, "B")]
    path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")


# ------------------------------------------------------------------ exportación
def export_all(results: Path | str = ROOT / "results", auto: Path | str = ROOT / "paper" / "auto") -> list[str]:
    results, auto = Path(results), Path(auto)
    auto.mkdir(parents=True, exist_ok=True)
    rd = lambda n: pd.read_csv(results / n)
    m = Macros()

    # ---- escenarios
    summ, comp, dec = rd("escenarios_resumen.csv"), rd("escenarios_comparaciones.csv"), rd("escenarios_decodificadores_pareados.csv")
    keep = summ.metric.str.match(r"^(acc_|kappa_|f1_|gain_|recall_|true_|vit_)")
    m.summary("R", summ[keep], ["scenario"])
    s = lambda sc, met, st: float(summ[(summ.scenario == sc) & (summ.metric == met)][st].iloc[0])
    rows = []
    for sc in SCEN_TEX:
        rows.append([esc(SCEN_TEX[sc]), num(s(sc, "acc_viterbi", "mean")), num(s(sc, "acc_viterbi", "sd")),
                     num(s(sc, "acc_viterbi", "min")), num(s(sc, "acc_viterbi", "max")),
                     ci(s(sc, "acc_viterbi", "ci_low"), s(sc, "acc_viterbi", "ci_high")),
                     num(s(sc, "acc_baseline", "mean")), num(s(sc, "acc_posterior", "mean")), num(s(sc, "kappa_viterbi", "mean"))])
    _table(auto / "tabla_escenarios.tex", ["Escenario", "Media", "DE", "Mín", "Máx", "IC95\\,\\%", "Sin mem.", "Post.", r"$\kappa$"],
           rows, "lccccccc c".replace(" ", ""))

    rows = []
    for _, r in comp[comp.metric == "acc_viterbi"].iterrows():
        rows.append([esc(SCEN_TEX[r.scenario]), num(r.mean_diff), ci(r.diff_ci_low, r.diff_ci_high), num(r.cohens_d, 2),
                     pval(r.p_welch_holm), pval(r.p_mannwhitney)])
    _table(auto / "tabla_comparaciones.tex", ["Escenario vs E1", r"$\Delta$ media", "IC95\\,\\% de $\\Delta$", r"$d$ de Cohen",
                                              "$p$ Welch (Holm)", "$p$ Mann-Whitney"], rows, "lccccc")
    for _, r in comp.iterrows():
        m.add("K", "cmp", r.metric, r.scenario, "dpp", value=num(100 * float(r.mean_diff), 1))
        m.add("K", "cmp", r.metric, r.scenario, "lopp", value=num(100 * float(r.diff_ci_low), 1))
        m.add("K", "cmp", r.metric, r.scenario, "hipp", value=num(100 * float(r.diff_ci_high), 1))
        m.add("K", "cmp", r.metric, r.scenario, "d", value=num(float(r.cohens_d), 1))
        m.add("K", "cmp", r.metric, r.scenario, "pholm", value=pval(float(r.p_welch_holm)))
        m.add("K", "cmp", r.metric, r.scenario, "pmw", value=pval(float(r.p_mannwhitney)))
    m.add("K", "kruskalp", value=pval(float(comp.kruskal_p_all.iloc[0])))

    rows = []
    names = {("acc_viterbi", "acc_baseline"): "Viterbi $-$ sin memoria", ("acc_posterior", "acc_viterbi"): "Posterior $-$ Viterbi",
             ("acc_posterior", "acc_oracle"): "Posterior $-$ techo (modelo verdadero)"}
    for _, r in dec.iterrows():
        rows.append([esc(SCEN_TEX[r.scenario]), names[(r.a, r.b)], num(r.mean_diff), ci(r.ci_low, r.ci_high), pval(r.p_paired_t)])
        m.add("K", "dec", r.scenario, r.a, r.b, "mean", value=num(100 * float(r.mean_diff), 1))
        m.add("K", "dec", r.scenario, r.a, r.b, "lo", value=num(100 * float(r.ci_low), 1))
        m.add("K", "dec", r.scenario, r.a, r.b, "hi", value=num(100 * float(r.ci_high), 1))
        m.add("K", "dec", r.scenario, r.a, r.b, "p", value=pval(float(r.p_paired_t)))
    _table(auto / "tabla_decodificadores.tex", ["Escenario", "Contraste (pareado)", r"$\Delta$ exactitud", "IC95\\,\\%", "$p$ (t pareada)"],
           rows, "llccc")

    # ---- métricas clínicas (real vs Viterbi) en E1 y E5
    labels = [("sleep_efficiency", "Eficiencia del sueño (\\%)", 1), ("sol_min", "Latencia de sueño (min)", 1),
              ("rem_latency_min", "Latencia REM (min)", 1), ("waso_min", "WASO (min)", 1), ("awakenings", "Despertares (n)", 1),
              ("pct_D_tst", "Sueño profundo (\\% TST)", 1), ("pct_R_tst", "REM (\\% TST)", 1), ("cycles", "Ciclos NREM$\\to$REM (n)", 1),
              ("shifts_per_hour", "Cambios de etapa por hora", 1)]
    rows = []
    for key, lab, nd in labels:
        row = [lab]
        for sc in ("E1", "E5"):
            for pre in ("true", "vit"):
                row.append(f"{num(s(sc, f'{pre}_{key}', 'mean'), nd)} $\\pm$ {num(s(sc, f'{pre}_{key}', 'sd'), nd)}")
        rows.append(row)
    _table(auto / "tabla_clinica.tex", ["Indicador", "E1 real", "E1 Viterbi", "E5 real", "E5 Viterbi"], rows, "lcccc")

    # ---- sensibilidad
    sens = rd("sensibilidad_resumen.csv")
    for _, r in sens.iterrows():
        for st in ("mean", "sd", "ci_low", "ci_high"):
            m.add("S", r.factor, float(r.level), r.metric, st, value=num(float(r[st]), 4))
            m.add("S", r.factor, float(r.level), r.metric, f"{st}pct", value=num(100 * float(r[st]), 1))
    rows = []
    for fac, lab in (("A_mix", "Perturbar $\\mathbf{A}$"), ("B_mix", "Degradar $\\mathbf{B}$"), ("persistence", "Persistencia $a_{ii}$")):
        sub = sens[(sens.factor == fac) & (sens.metric == "acc_viterbi")].sort_values("level")
        for _, r in sub.iterrows():
            def g(met):
                v = sens[(sens.factor == fac) & (sens.level == r.level) & (sens.metric == met)]["mean"].iloc[0]
                return num(float(v))
            rows.append([lab, f"{r.level:g}", g("acc_viterbi") + " " + r"{\scriptsize " + ci(r.ci_low, r.ci_high) + "}",
                         g("acc_posterior"), g("acc_baseline"), g("acc_oracle")])
            lab = ""
    _table(auto / "tabla_sensibilidad.tex", ["Factor", "Nivel", "Viterbi (IC95\\,\\%)", "Posterior", "Sin memoria", "Techo"], rows, "llcccc")

    # ---- condición inicial
    ic = rd("condiciones_iniciales_resumen.csv")
    for _, r in ic.iterrows():
        for st in ("mean", "sd"):
            m.add("I", r.estado_inicial, r.decodificador, r.metric, st, value=num(float(r[st]), 4))
    for met, tag in (("acc_total", "tot"), ("acc_primeras", "first"), ("acc_resto", "rest")):
        v = ic[ic.metric == met]["mean"]
        m.add("K", "ic", tag, "minpct", value=num(100 * float(v.min()), 1))
        m.add("K", "ic", tag, "maxpct", value=num(100 * float(v.max()), 1))
    rows = []
    for est in ic.estado_inicial.unique():
        row = [esc(est)]
        for d in ("pi_nominal", "pi_uniforme", "pi_estacionaria"):
            for met in ("acc_primeras", "acc_resto"):
                v = ic[(ic.estado_inicial == est) & (ic.decodificador == d) & (ic.metric == met)]["mean"].iloc[0]
                row.append(num(float(v)))
        rows.append(row)
    _table(auto / "tabla_condicion_inicial.tex",
           ["Estado inicial", r"nom. 10", "nom. resto", "unif. 10", "unif. resto", "est. 10", "est. resto"], rows, "lcccccc")

    # ---- semillas y propiedades de la cadena
    sd = rd("sensibilidad_semillas.csv")
    m.add("K", "seedsd", value=num(float(sd.mean_acc_viterbi.std(ddof=1)), 4))
    m.add("K", "seedmean", value=num(float(sd.mean_acc_viterbi.mean()), 4))
    m.add("K", "seedminpct", value=num(100 * float(sd.mean_acc_viterbi.min()), 1))
    m.add("K", "seedmaxpct", value=num(100 * float(sd.mean_acc_viterbi.max()), 1))
    m.add("K", "seedn", value=str(len(sd)))
    m.add("K", "seedcover", value=num(100 * float(((sd.mean_acc_viterbi >= sd.ci_low.iloc[0]) & (sd.mean_acc_viterbi <= sd.ci_high.iloc[0])).mean()), 0))
    m.add("K", "seedsepred", value=num(float(sd["sd"].mean() / np.sqrt(sd["n"].iloc[0])), 4))
    cp = rd("propiedades_cadena.csv")
    rows = []
    for _, r in cp.iterrows():
        rows.append([esc(r.estado), num(r.pi_inicial, 2), num(r.a_ii, 2), num(r.duracion_media_epocas, 1), num(100 * r.distribucion_estacionaria, 1)])
        m.add("K", "stat", r.estado, value=num(100 * float(r.distribucion_estacionaria), 1))
        m.add("K", "dur", r.estado, value=num(float(r.duracion_media_epocas), 1))
    m.add("K", "lambdados", value=num(float(cp.abs_lambda2.iloc[0]), 3))
    m.add("K", "relaj", value=num(float(cp.tiempo_relajacion_epocas.iloc[0]), 1))
    _table(auto / "tabla_cadena.tex", ["Estado", r"$\pi_i$", "$a_{ii}$", "Duración media (épocas)", "Estacionaria (\\%)"], rows, "lcccc")

    # ---- matrices de confusión normalizadas por fila (en %)
    cmx = rd("matrices_confusion.csv")
    for (sc, dec_name, real), g in cmx.groupby(["scenario", "decodificador", "real"]):
        tot = g.conteo.sum()
        for _, r in g.iterrows():
            m.add("K", "conf", sc, dec_name, real, r.estimada, value=num(100 * float(r.conteo) / float(tot), 1))

    # ---- aprendizaje
    raw_learn = rd("aprendizaje_baum_welch.csv")
    for (mode, n), g in raw_learn.groupby(["init", "n_train"]):
        gap = (g.loglik_test_per_epoch_true - g.loglik_test_per_epoch_learned)
        m.add("K", "dll", mode, int(n), value=num(float(gap.mean()), 4))
        m.add("K", "dllmax", mode, int(n), value=num(float(gap.max()), 4))
        m.add("K", "nbad", mode, int(n), value=str(int((g.acc_test_learned < g.acc_test_baseline - 0.05).sum())))
    lr = rd("aprendizaje_resumen.csv")
    for _, r in lr.iterrows():
        for st in ("mean", "sd", "ci_low", "ci_high", "min", "max"):
            m.add("L", r.init, int(r.n_train), r.metric, st, value=num(float(r[st]), 4))
            m.add("L", r.init, int(r.n_train), r.metric, f"{st}3", value=num(float(r[st]), 3))
            m.add("L", r.init, int(r.n_train), r.metric, f"{st}pct", value=num(100 * float(r[st]), 1))
    rows = []
    for mode in ("aleatoria", "informada"):
        for n in sorted(lr.n_train.unique()):
            g = lambda met, st="mean": float(lr[(lr.init == mode) & (lr.n_train == n) & (lr.metric == met)][st].iloc[0])
            rows.append([mode.capitalize(), str(int(n)),
                         f"{num(g('acc_test_learned'))} {{\\scriptsize {ci(g('acc_test_learned', 'ci_low'), g('acc_test_learned', 'ci_high'))}}}",
                         num(g("loglik_test_per_epoch_learned") - g("loglik_test_per_epoch_true"), 4), num(g("mae_A")), num(g("mae_B")),
                         num(g("iterations"), 0)])
    g0 = lambda met: float(lr[(lr.init == "aleatoria") & (lr.n_train == 1) & (lr.metric == met)]["mean"].iloc[0])
    m.add("K", "accverdaderopct", value=num(100 * g0("acc_test_true"), 1))
    m.add("K", "accbasepruebapct", value=num(100 * g0("acc_test_baseline"), 1))
    _table(auto / "tabla_aprendizaje.tex", ["Inicialización", "Noches", "Exactitud en prueba (IC95\\,\\%)", r"$\Delta$ log-verosim./época",
                                            "MAE $\\mathbf{A}$", "MAE $\\mathbf{B}$", "Iter. EM"], rows, "llcccccc"[:7])

    # ---- patologías
    cm = rd("patologias_confusion.csv").set_index("clase_real")
    ps = rd("patologias_resumen.csv").set_index("clase")
    rows = []
    for real in cm.index:
        rows.append([esc(real)] + [str(int(cm.loc[real, c])) for c in cm.columns] +
                    [num(ps.loc[real, "sensibilidad_forward"], 2), num(ps.loc[real, "especificidad_forward"], 3),
                     num(ps.loc[real, "auc_normal_vs_clase"], 3)])
    _table(auto / "tabla_patologias.tex", ["Clase real"] + [esc(c) for c in cm.columns] + ["Sens.", "Espec.", "AUC"], rows, "lcccccc")
    for real in cm.index:
        m.add("K", "patsens", real, value=num(float(ps.loc[real, "sensibilidad_forward"]), 2))
    sev = rd("patologias_severidad.csv")
    for _, r in sev.iterrows():
        m.add("K", "sev", r.familia, float(r.severidad), "auc", value=num(float(r.auc), 3))
        m.add("K", "sev", r.familia, float(r.severidad), "lo", value=num(float(r.auc_ci_low), 3))
        m.add("K", "sev", r.familia, float(r.severidad), "hi", value=num(float(r.auc_ci_high), 3))
    ref = rd("patologias_referencia.csv")
    for _, r in ref.iterrows():
        m.add("K", "ref", r.metrica, "mean", value=num(float(r.media_ref), 1))
        m.add("K", "ref", r.metrica, "sd", value=num(float(r.sd_ref), 1))

    # ---- validación del simulador
    cv = rd("convergencia_simulador.csv").groupby("T").frobenius_A.mean()
    slope = np.polyfit(np.log(cv.index.to_numpy(float)), np.log(cv.to_numpy()), 1)[0]
    m.add("K", "convslope", value=num(float(slope), 2))
    m.add("K", "convfirst", value=num(float(cv.iloc[0]), 3))
    m.add("K", "convlast", value=num(float(cv.iloc[-1]), 4))
    m.add("K", "convTfirst", value=f"{int(cv.index[0]):,}".replace(",", r"\,"))
    m.add("K", "convTlast", value=f"{int(cv.index[-1]):,}".replace(",", r"\,"))
    st = rd("estabilidad_numerica.csv")
    g = st.groupby("T").agg(under=("underflow_lineal", "mean"), same=("coincide_con_log", "mean"), tie=("diferencia_por_empate", "mean"),
                            acc_lin=("acc_lineal", "mean"), acc_log=("acc_log", "mean"))
    rows = []
    for T, r in g.iterrows():
        rows.append([str(int(T)), num(100 * r.under, 0), num(100 * r.same, 0), num(100 * r.tie, 0), num(r.acc_lin), num(r.acc_log)])
        m.add("K", "stab", int(T), "under", value=num(100 * float(r.under), 0))
        m.add("K", "stab", int(T), "acclinpct", value=num(100 * float(r.acc_lin), 1))
        m.add("K", "stab", int(T), "acclogpct", value=num(100 * float(r.acc_log), 1))
        m.add("K", "stab", int(T), "tie", value=num(100 * float(r.tie), 0))
    _table(auto / "tabla_estabilidad.tex", ["$T$ (épocas)", "Underflow (\\%)", "Ruta idéntica (\\%)", "Dif. por empate (\\%)",
                                            "Exactitud lineal", "Exactitud log"], rows, "cccccc")
    ex = rd("consistencia_extraccion_senales.csv")
    m.add("K", "extractacc", value=num(100 * float(ex[ex.simbolo_generado == ex.simbolo_extraido].conteo.sum() / ex.conteo.sum()), 1))

    nr = results / "noche_representativa.json"
    if nr.exists():
        import json
        info = json.loads(nr.read_text(encoding="utf-8"))
        m.add("K", "nightacc", value=num(100 * float(info["exactitud"]), 1))
        m.add("K", "nightrep", value=str(int(info["replica"])))
    write_matrices(auto / "matrices.tex")
    m.dump(auto / "macros.tex")
    return sorted(m.names)


if __name__ == "__main__":
    names = export_all()
    print(f"{len(names)} macros escritas en {ROOT / 'paper' / 'auto'}")
