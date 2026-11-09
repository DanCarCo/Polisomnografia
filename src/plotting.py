"""Figuras de calidad de publicación (PNG, 300 dpi) del estudio.

Criterios de diseño
-------------------
* Colores de etapa impuestos por el enunciado (Azul = Despierto, Verde = Ligero,
  Morado = Profundo, Rojo = REM). Los tonos se validaron con ``validate_palette.js``
  (modo claro, todos los pares: ΔE normal mín. 20.6, contraste ≥ 3:1).
* Resto de entidades (escenarios, decodificadores) en orden categórico fijo.
* Líneas de 2 px, marcadores ≥ 8 px con anillo del color de la superficie, rejilla
  y ejes recesivos, texto en tintas neutras (nunca en el color de la serie),
  leyenda siempre que hay ≥ 2 series y ningún eje dual.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402
from scipy.signal import welch  # noqa: E402

from . import eeg_signals as eeg  # noqa: E402
from . import sleep_model as sm  # noqa: E402
from .clinical import bouts  # noqa: E402

# ------------------------------------------------------------------ paleta
STAGE_COLORS = ("#2a78d6", "#008300", "#8f2fa3", "#e34948")        # Despierto, Ligero, Profundo, REM
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SCEN_COLORS = {"E1": BLUE, "E2": ORANGE, "E3": AQUA, "E4": YELLOW, "E5": MAGENTA}
DEC_COLORS = {"viterbi": BLUE, "posterior": ORANGE, "baseline": "#898781", "oracle": AQUA}
DEC_LABELS = {"viterbi": "Viterbi", "posterior": "Posterior (Forward-Backward)",
              "baseline": "Sin memoria (solo B)", "oracle": "Techo (modelo verdadero)"}
SCEN_NAMES = {"E1": "Nominal", "E2": "Transición\nperturbada", "E3": "Emisión\ndegradada",
              "E4": "Emisión\ncontinua", "E5": "Sueño\nconsolidado"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE_RAMP = LinearSegmentedColormap.from_list("azul", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
HYP_POS = {sm.IDX_W: 3, sm.IDX_R: 2, sm.IDX_L: 1, sm.IDX_D: 0}     # convención de hipnograma (AASM)
DPI = 300


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": SURFACE,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "text.color": INK, "legend.frameon": False, "legend.fontsize": 8,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "figure.dpi": 100,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def _save(fig, path) -> str:
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return str(path)


def _panel_title(ax, text):
    ax.set_title(text, loc="left", pad=6)


def _legend_key(color, label, marker="o"):
    if marker == "s":      # identidad categórica de etapa: cuadrado sólido, sin línea
        return Line2D([0], [0], color=color, marker="s", markersize=8, lw=0, markeredgecolor=SURFACE, label=label)
    return Line2D([0], [0], color=color, marker=marker, markersize=6, lw=2, markeredgecolor=SURFACE, label=label)


# ================================================================== diagramas
def fig_flow(path):
    """Diagrama de flujo del diseño experimental."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)

    def box(cx, cy, w, h, text, fc="#f4f3ef", ec=AXIS, rounded=True):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.25" if rounded else "square,pad=0.02",
                                    fc=fc, ec=ec, lw=1.0))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=8, color=INK, linespacing=1.25)

    def arrow(p, q, rad=0.0, text=None, tx=None, ty=None):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.2, color=INK2,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
        if text:
            ax.text(tx, ty, text, fontsize=7.5, color=INK2, ha="center", va="center")

    box(5, 13.0, 7.4, 1.2, "F1–F3  Formulación y modelo formal\nλ = (S, V, A, B, π) · semilla maestra = 42", fc="#e9f1fc")
    box(5, 11.2, 7.4, 1.2, "F5  Generación de noches (480 épocas)\nSeedSequence.spawn → una semilla independiente por réplica")
    box(5, 9.4, 7.4, 1.2, "F4  Inferencia\nViterbi · Posterior (Forward-Backward) · clasificador sin memoria")
    box(5, 7.6, 7.4, 1.2, "Métricas por noche\nexactitud, κ de Cohen, F1 macro, métricas clínicas del hipnograma")
    ax.add_patch(plt.Polygon([[5, 6.55], [7.7, 5.5], [5, 4.45], [2.3, 5.5]], closed=True, fc="#fff4d6", ec=AXIS, lw=1.0))
    ax.text(5, 5.5, "¿r = N réplicas?", ha="center", va="center", fontsize=8)
    box(5, 3.3, 7.4, 1.2, "F6  Estadística: media, DE, mín, máx, IC95 % (t y bootstrap)\npruebas de Welch/Mann-Whitney con corrección de Holm")
    box(5, 1.5, 7.4, 1.2, "Estudios adicionales: sensibilidad (A, B, a_ii), condición inicial,\nBaum-Welch, patologías simuladas → CSV, figuras, informe", fc="#e6f4ec")
    for y0, y1 in [(12.4, 11.8), (10.6, 10.0), (8.8, 8.2), (7.0, 6.55), (4.45, 3.9), (2.7, 2.1)]:
        arrow((5, y0), (5, y1))
    ax.text(5.15, 3.98, "sí", fontsize=7.5, color=INK2, ha="left")
    ax.plot([2.3, 0.7, 0.7], [5.5, 5.5, 11.2], color=INK2, lw=1.2)
    arrow((0.7, 11.2), (1.3, 11.2))
    ax.text(0.55, 8.4, "no (r ← r + 1)", fontsize=7.5, color=INK2, rotation=90, ha="right", va="center")
    return _save(fig, path)


def fig_model(path, h):
    """Grafo de transiciones entre etapas (anchura de arista ∝ probabilidad)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.8, 5.9))
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-2.95, 2.95)
    pos = {sm.IDX_W: (0.0, 1.65), sm.IDX_L: (2.5, 0.0), sm.IDX_D: (0.0, -1.65), sm.IDX_R: (-2.5, 0.0)}
    out_angle = {sm.IDX_W: 90, sm.IDX_L: 0, sm.IDX_D: 270, sm.IDX_R: 180}     # hacia fuera del centro
    r, A, n = 0.44, h.A, h.n_states
    circles = {i: Circle(pos[i], r, fc=STAGE_COLORS[i], ec=SURFACE, lw=2.0, zorder=3) for i in range(n)}
    for c in circles.values():
        ax.add_patch(c)
    for i in range(n):
        for j in range(n):
            if i == j or A[i, j] <= 0:
                continue
            p0, p2 = np.array(pos[i]), np.array(pos[j])
            rad = 0.26
            ax.add_patch(FancyArrowPatch(tuple(p0), tuple(p2), connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                         mutation_scale=8, lw=0.6 + 4.0 * A[i, j], color=MUTED, alpha=0.9,
                                         patchA=circles[i], patchB=circles[j], shrinkA=1.5, shrinkB=1.5, zorder=1))
            ctrl = (p0 + p2) / 2 + rad * np.array([p2[1] - p0[1], -(p2[0] - p0[0])])    # punto de control de arc3
            tt = 0.30                                                                    # etiqueta cerca del origen
            lab = (1 - tt) ** 2 * p0 + 2 * (1 - tt) * tt * ctrl + tt ** 2 * p2
            ax.text(lab[0], lab[1], f"{A[i, j]:.2f}", fontsize=7.2, color=INK, ha="center", va="center", zorder=5,
                    bbox=dict(fc=SURFACE, ec="none", pad=0.7, alpha=0.92))
    for i in range(n):
        x, y = pos[i]
        ang = out_angle[i]
        lr = 0.27
        cx, cy = x + 1.32 * r * np.cos(np.radians(ang)), y + 1.32 * r * np.sin(np.radians(ang))
        ax.add_patch(Arc((cx, cy), 2 * lr, 2 * lr, theta1=ang + 215, theta2=ang + 505, color=MUTED,
                         lw=0.6 + 4.0 * A[i, i], zorder=1))
        tx, ty = x + 2.6 * r * np.cos(np.radians(ang)), y + 2.6 * r * np.sin(np.radians(ang))
        ax.text(tx, ty, f"{A[i, i]:.2f}", fontsize=7.2, ha="center", va="center", color=INK, zorder=5)
        ax.text(x, y, sm.STATE_SHORT[i], color="#ffffff", fontsize=8.5, fontweight="bold", ha="center", va="center", zorder=6)
    names = list(sm.STATE_NAMES)
    names[sm.IDX_R] = "Sueño REM"
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=STAGE_COLORS[i], markeredgecolor=SURFACE, markersize=9,
                      label=f"{sm.STATE_SHORT[i]} = {names[i]} (π = {h.pi[i]:.2f})") for i in range(n)]
    ax.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.02), fontsize=7.6, handletextpad=0.3,
              columnspacing=1.4)
    fig.text(0.5, 0.045, "Grosor ∝ probabilidad a_ij · el bucle es la permanencia a_ii", ha="center", fontsize=7.2, color=MUTED)
    return _save(fig, path)


def _heatmap(ax, M, rows, cols, title, fmt="{:.2f}", vmax=None):
    vmax = vmax if vmax is not None else max(1.0, float(np.nanmax(M)))
    ax.imshow(M, cmap=BLUE_RAMP, vmin=0, vmax=vmax)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=8,
                    color="#ffffff" if M[i, j] > 0.55 * vmax else INK)
    ax.set_xticks(range(len(cols)), cols, fontsize=7.5)
    ax.set_yticks(range(len(rows)), rows, fontsize=7.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    _panel_title(ax, title)


def fig_matrices(path, A, B):
    """Mapas de calor de la matriz de transición A y de emisión B."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    _heatmap(axes[0], A, sm.STATE_SHORT, sm.STATE_SHORT, "(a) Transición A (fila = estado actual)")
    _heatmap(axes[1], B, sm.STATE_SHORT, sm.OBS_SHORT, "(b) Emisión B (fila = estado oculto)")
    axes[0].set_xlabel("Estado siguiente")
    axes[1].set_xlabel("Símbolo observado")
    fig.tight_layout(w_pad=2.0)
    return _save(fig, path)


def fig_learned(path, truth, models):
    """A y B verdaderas frente a las aprendidas por Baum-Welch (réplica 0, estados alineados)."""
    apply_style()
    n = max(k[1] for k in models)
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.9))
    cols = [("(%s) %s", None)] + [(f"{mode}", (mode, n)) for mode in ("aleatoria", "informada")]
    _heatmap(axes[0, 0], truth.A, sm.STATE_SHORT, sm.STATE_SHORT, "(a) A verdadera")
    _heatmap(axes[1, 0], truth.B, sm.STATE_SHORT, sm.OBS_SHORT, "(d) B verdadera")
    for j, (mode, key) in enumerate(cols[1:], start=1):
        letter_a, letter_b = ("b", "e") if j == 1 else ("c", "f")
        _heatmap(axes[0, j], models[key].A, sm.STATE_SHORT, sm.STATE_SHORT, f"({letter_a}) A, inic. {mode}")
        _heatmap(axes[1, j], models[key].B, sm.STATE_SHORT, sm.OBS_SHORT, f"({letter_b}) B, inic. {mode}")
    for ax in axes[:, 1:].ravel():
        ax.set_yticklabels([])
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    return _save(fig, path)


# ================================================================== noche representativa
def _draw_hyp(ax, h, dt):
    segs = bouts(h)
    prev = None
    for state, start, length in segs:
        y = HYP_POS[state]
        ax.hlines(y, start * dt, (start + length) * dt, color=STAGE_COLORS[state], lw=3.0, capstyle="butt", zorder=3)
        if prev is not None:
            ax.vlines(start * dt, min(prev, y), max(prev, y), color=AXIS, lw=0.8, zorder=2)
        prev = y
    ax.set_yticks([3, 2, 1, 0], [sm.STATE_NAMES[sm.IDX_W], sm.STATE_NAMES[sm.IDX_R], sm.STATE_NAMES[sm.IDX_L],
                                 sm.STATE_NAMES[sm.IDX_D]], fontsize=7.8)
    ax.set_ylim(-0.9, 3.5)
    ax.grid(axis="x", visible=False)


def fig_night(path, states, vit, freqs, spec, epoch_min=1.0, acc=None):
    """Hipnograma real, hipnograma Viterbi y espectrograma simulado de una noche."""
    apply_style()
    T = len(states)
    dt = epoch_min / 60.0
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 6.4), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1.05]})
    _draw_hyp(axes[0], states, dt)
    _panel_title(axes[0], "(a) Hipnograma real (estados ocultos simulados)")
    _draw_hyp(axes[1], vit, dt)
    wrong = np.flatnonzero(np.asarray(states) != np.asarray(vit))
    axes[1].plot(wrong * dt + dt / 2, np.full(len(wrong), -0.62), "|", color=INK2, ms=5, mew=0.7)
    ttl = "(b) Hipnograma decodificado con Viterbi (barras inferiores = épocas erróneas)"
    if acc is not None:
        ttl = f"(b) Viterbi · exactitud {100 * acc:.1f} % (barras inferiores = épocas erróneas)"
    _panel_title(axes[1], ttl)
    im = axes[2].imshow(spec, aspect="auto", origin="lower", extent=[0, T * dt, freqs[0], freqs[-1]], cmap=BLUE_RAMP,
                        interpolation="nearest")
    axes[2].grid(False)
    for f0 in (4, 8, 13):
        axes[2].axhline(f0, color="#ffffff", lw=0.6, alpha=0.8)
    for lab, y in (("δ", 2.0), ("θ", 6.0), ("α", 10.5), ("β", 20.0)):
        axes[2].text(0.06, y, lab, color=INK, fontsize=8, va="center",
                     bbox=dict(fc="#ffffff", ec="none", alpha=0.75, pad=0.8))
    axes[2].set_ylabel("Frecuencia (Hz)")
    axes[2].set_xlabel("Tiempo desde el inicio del registro (h)")
    _panel_title(axes[2], "(c) Espectrograma EEG sintético (densidad espectral por época, dB)")
    cax = axes[2].inset_axes([1.012, 0.05, 0.015, 0.9])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2)
    axes[2].set_xlim(0, T * dt)
    handles = [_legend_key(STAGE_COLORS[i], sm.STATE_NAMES[i], marker="s") for i in range(4)]
    axes[0].legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.22), columnspacing=1.4,
                   handletextpad=0.4)
    fig.tight_layout(h_pad=1.1)
    return _save(fig, path)


def fig_eeg(path, rng):
    """EEG, EOG y espectro de una época típica de cada etapa (señales sintéticas)."""
    apply_style()
    fig = plt.figure(figsize=(6.8, 5.6))
    gs = fig.add_gridspec(4, 3, width_ratios=[2.2, 2.2, 1.35], hspace=0.35, wspace=0.28)
    sym_of_stage = {0: 0, 1: 1, 2: 2, 3: 3}
    for r in range(4):
        t, e, o = eeg.simulate_epoch(sym_of_stage[r], rng, dur_s=30.0)
        m_eeg, m_eog = t < 3.0, t < 10.0
        col = STAGE_COLORS[r]
        a1 = fig.add_subplot(gs[r, 0])
        a1.plot(t[m_eeg], e[m_eeg], color=col, lw=0.9)
        a1.set_ylim(-190, 190)
        a2 = fig.add_subplot(gs[r, 1], sharey=a1)
        a2.plot(t[m_eog], o[m_eog], color=col, lw=0.9)
        a3 = fig.add_subplot(gs[r, 2])
        f, p = welch(e, fs=eeg.FS, nperseg=int(4 * eeg.FS))
        k = f <= 30
        a3.semilogy(f[k], p[k], color=col, lw=1.6)
        for f0 in (4, 8, 13):
            a3.axvline(f0, color=AXIS, lw=0.6)
        a3.set_ylim(1e-1, 1e4)
        a1.set_ylabel(f"{sm.STATE_NAMES[r]}\n(µV)", fontsize=8, color=INK)
        if r == 0:
            _panel_title(a1, "EEG (3 s)")
            _panel_title(a2, "EOG (10 s)")
            _panel_title(a3, "PSD (µV²/Hz)")
        if r < 3:
            for a in (a1, a2, a3):
                a.set_xticklabels([])
        else:
            a1.set_xlabel("Tiempo (s)")
            a2.set_xlabel("Tiempo (s)")
            a3.set_xlabel("Frecuencia (Hz)")
        a2.tick_params(labelleft=False)
    return _save(fig, path)


# ================================================================== escenarios
def _jitter(rng, n, w=0.12):
    return rng.uniform(-w, w, n)


def fig_scenarios(path, df, summ):
    apply_style()
    keys = list(df.scenario.unique())
    rng = np.random.default_rng(7)             # solo para el desplazamiento visual de los puntos
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw={"width_ratios": [1.1, 1.1]})
    n_nights = int((df.scenario == keys[0]).sum())
    ax = axes[0]
    for i, k in enumerate(keys):
        v = df.loc[df.scenario == k, "acc_viterbi"].to_numpy()
        ax.boxplot(v, positions=[i], widths=0.42, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=SCEN_COLORS[k], alpha=0.22, edgecolor="none"),
                   medianprops=dict(color=SCEN_COLORS[k], lw=2.0), whiskerprops=dict(color=AXIS, lw=0.9),
                   capprops=dict(color=AXIS, lw=0.9))
        ax.scatter(i + _jitter(rng, len(v)), v, s=7, color=SCEN_COLORS[k], alpha=0.55, linewidths=0, zorder=3)
    ax.set_xticks(range(len(keys)), [f"{k}\n{SCEN_NAMES[k]}" for k in keys], fontsize=6.6)
    ax.set_ylabel("Exactitud de Viterbi por noche")
    ax.set_ylim(0.4, 1.0)
    _panel_title(ax, f"(a) Distribución (N = {n_nights} noches)")
    ax = axes[1]
    offs = {"baseline": -0.24, "viterbi": 0.0, "posterior": 0.24}
    for dec in ("baseline", "viterbi", "posterior"):
        mean, lo, hi = [], [], []
        for k in keys:
            r = summ[(summ.scenario == k) & (summ.metric == f"acc_{dec}")].iloc[0]
            mean.append(r["mean"]); lo.append(r["ci_low"]); hi.append(r["ci_high"])
        x = np.arange(len(keys)) + offs[dec]
        mean, lo, hi = map(np.array, (mean, lo, hi))
        ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt="none", ecolor=DEC_COLORS[dec], elinewidth=2.0, capsize=0, zorder=2)
        ax.plot(x, mean, "o", color=DEC_COLORS[dec], ms=7.5, mec=SURFACE, mew=1.5, zorder=3, label=DEC_LABELS[dec])
    ax.set_xticks(range(len(keys)), [f"{k}\n{SCEN_NAMES[k]}" for k in keys], fontsize=6.6)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Exactitud media e IC95 %")
    ax.legend(loc="upper left", handletextpad=0.3, borderaxespad=0.2)
    _panel_title(ax, "(b) Decodificadores por escenario")
    fig.tight_layout(w_pad=1.6)
    return _save(fig, path)


def fig_confusion(path, cms: list, titles: list):
    apply_style()
    fig, axes = plt.subplots(1, len(cms), figsize=(3.4 * len(cms), 3.2))
    axes = np.atleast_1d(axes)
    for ax, cm, ttl in zip(axes, cms, titles):
        M = cm / cm.sum(axis=1, keepdims=True)
        _heatmap(ax, M, sm.STATE_SHORT, sm.STATE_SHORT, ttl, fmt="{:.2f}", vmax=1.0)
        ax.set_xlabel("Etapa decodificada")
    axes[0].set_ylabel("Etapa real")
    fig.tight_layout(w_pad=1.6)
    return _save(fig, path)


def fig_sweeps(path, sweep_df):
    apply_style()
    specs = [("A_mix", "(a) Perturbar A: (1−ε)·A + ε·U", "Intensidad ε"),
             ("B_mix", "(b) Degradar B: (1−ε)·B + ε·U", "Intensidad ε"),
             ("persistence", "(c) Persistencia a_ii (modelo ajustado)", "Probabilidad de permanencia a_ii")]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.2), sharey=True)
    from scipy import stats as st
    for ax, (fac, ttl, xl) in zip(axes, specs):
        sub = sweep_df[sweep_df.factor == fac]
        levels = np.sort(sub.level.unique())
        for dec in ("baseline", "oracle", "posterior", "viterbi"):
            if fac == "persistence" and dec == "oracle":
                continue
            col = f"acc_{dec}"
            g = sub.groupby("level")[col]
            m, s, n = g.mean().loc[levels].to_numpy(), g.std(ddof=1).loc[levels].to_numpy(), g.count().loc[levels].to_numpy()
            ci = st.t.ppf(0.975, n - 1) * s / np.sqrt(n)
            ax.fill_between(levels, m - ci, m + ci, color=DEC_COLORS[dec], alpha=0.14, lw=0)
            ax.plot(levels, m, color=DEC_COLORS[dec], lw=2.0)
        ax.set_xlabel(xl)
        _panel_title(ax, ttl)
        ax.set_ylim(0.3, 1.0)
    axes[0].set_ylabel("Exactitud media (banda = IC95 %)")
    handles = [_legend_key(DEC_COLORS[d], DEC_LABELS[d], marker="") for d in ("viterbi", "posterior", "baseline", "oracle")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07), handlelength=1.6, columnspacing=1.4)
    fig.tight_layout(w_pad=1.0)
    return _save(fig, path)


# ================================================================== aprendizaje
def fig_learning(path, learn_df, hist_df):
    from scipy import stats as st
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    mode_col = {"aleatoria": BLUE, "informada": ORANGE}
    mode_lab = {"aleatoria": "Inicialización aleatoria", "informada": "Inicialización informada"}
    sizes = sorted(learn_df.n_train.unique())
    xs = np.arange(len(sizes))

    ax = axes[0, 0]
    for (mode, q), g in hist_df.groupby(["init", "restart"]):
        ax.plot(g["iter"], g["loglik"], color=mode_col[mode], lw=1.6, alpha=0.9)
    ax.set_ylim(hist_df.loglik.max() - 900, hist_df.loglik.max() + 40)
    ax.set_xlabel("Iteración EM")
    ax.set_ylabel("log P(O | λ) de entrenamiento")
    ax.legend(handles=[_legend_key(mode_col[m], mode_lab[m], marker="") for m in mode_col], loc="lower right",
              handlelength=1.4, fontsize=7.2)
    _panel_title(ax, "(a) Convergencia (20 noches, 3 reinicios)")

    def mean_ci(mode, col):
        m, lo, hi = [], [], []
        for n in sizes:
            v = learn_df.loc[(learn_df.init == mode) & (learn_df.n_train == n), col].to_numpy()
            s = v.std(ddof=1) / np.sqrt(len(v))
            tcrit = st.t.ppf(0.975, len(v) - 1)
            m.append(v.mean()); lo.append(v.mean() - tcrit * s); hi.append(v.mean() + tcrit * s)
        m, lo, hi = map(np.array, (m, lo, hi))
        return m, m - lo, hi - m

    def series(ax, col, ylabel, title, refs=False):
        for mode in mode_col:
            m, dl, dh = mean_ci(mode, col)
            ax.errorbar(xs, m, yerr=[dl, dh], fmt="none", ecolor=mode_col[mode], elinewidth=2.0, capsize=0)
            ax.plot(xs, m, "o-", color=mode_col[mode], ms=7.5, mec=SURFACE, mew=1.5, label=mode_lab[mode])
        if refs:
            ax.axhline(learn_df.acc_test_true.mean(), color=AQUA, lw=2.0, label="Modelo verdadero")
            ax.axhline(learn_df.acc_test_baseline.mean(), color=DEC_COLORS["baseline"], lw=2.0, label="Sin memoria")
        ax.set_xticks(xs, [str(n) for n in sizes])
        ax.set_xlabel("Noches de entrenamiento")
        ax.set_ylabel(ylabel)
        _panel_title(ax, title)

    series(axes[0, 1], "acc_test_learned", "Exactitud en noches de prueba", "(b) Exactitud en prueba", refs=True)
    axes[0, 1].set_ylim(0.35, 0.8)
    axes[0, 1].legend(loc="lower right", handlelength=1.4, fontsize=6.8)
    series(axes[1, 0], "mae_A", "Error absoluto medio en A", "(c) Recuperación de A")
    series(axes[1, 1], "mae_B", "Error absoluto medio en B", "(d) Recuperación de B")
    for ax in (axes[1, 0], axes[1, 1]):
        ax.set_ylim(bottom=0)
    fig.tight_layout(h_pad=1.4, w_pad=1.2)
    return _save(fig, path)


# ================================================================== patologías
def fig_pathology(path, patho_df, sev_df):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.2), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    ax = axes[0]
    cls_col = {"Normal": BLUE, "Sueño fragmentado": ORANGE, "Sin REM": AQUA}
    for c, col in cls_col.items():
        g = patho_df[patho_df.clase_real == c]
        ax.scatter(g["ll_Sueño fragmentado"] - g["ll_Normal"], g["ll_Sin REM"] - g["ll_Normal"], s=12, color=col,
                   alpha=0.7, linewidths=0.5, edgecolors=SURFACE, label=c)
    ax.axhline(0, color=AXIS, lw=0.9)
    ax.axvline(0, color=AXIS, lw=0.9)
    ax.set_xlabel("log-razón: fragmentado vs normal")
    ax.set_ylabel("log-razón: sin REM vs normal")
    ax.legend(loc="lower right", handletextpad=0.2, borderaxespad=0.1, fontsize=7.2)
    _panel_title(ax, "(a) Log-razón (Forward)")
    for ax, fam, xl, xf, ttl in ((axes[1], "fragmentacion", "Fragmentación δ", lambda s: s, "(b) Fragmentación"),
                                 (axes[2], "rem", "Reducción de la entrada a REM", lambda s: 1 - s, "(c) Pérdida de REM")):
        g = sev_df[sev_df.familia == fam].copy()
        g["x"] = xf(g["severidad"])
        g = g.sort_values("x")
        ax.fill_between(g.x, g.auc_ci_low, g.auc_ci_high, color=BLUE, alpha=0.14, lw=0)
        ax.plot(g.x, g.auc, "o-", color=BLUE, ms=7, mec=SURFACE, mew=1.4)
        ax.axhline(0.5, color=MUTED, lw=1.0)
        ax.text(g.x.max(), 0.505, "azar", color=MUTED, fontsize=7, ha="right", va="bottom")
        ax.set_ylim(0.4, 1.02)
        ax.set_xlabel(xl)
        ax.set_ylabel("AUC (IC95 % bootstrap)" if fam == "fragmentacion" else "")
        _panel_title(ax, ttl)
    fig.tight_layout(w_pad=1.2)
    return _save(fig, path)


# ================================================================== validación
def fig_validation(path, conv_df, stab_df):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), gridspec_kw={"width_ratios": [1, 1.3, 1.1]})
    ax = axes[0]
    g = conv_df.groupby("T")["frobenius_A"]
    Ts = np.array(sorted(conv_df["T"].unique()), dtype=float)
    m, lo, hi = g.mean().loc[Ts].to_numpy(), g.min().loc[Ts].to_numpy(), g.max().loc[Ts].to_numpy()
    ax.fill_between(Ts, lo, hi, color=BLUE, alpha=0.14, lw=0)
    ax.loglog(Ts, m, "o-", color=BLUE, ms=6.5, mec=SURFACE, mew=1.3, label="‖Â − A‖_F (media, rango)")
    ref = m[0] * np.sqrt(Ts[0] / Ts)
    ax.loglog(Ts, ref, color=MUTED, lw=1.4, label="∝ T^(−1/2)")
    ax.set_xlabel("Longitud de la cadena T")
    ax.set_ylabel("Error de las frecuencias de transición")
    ax.legend(loc="upper right", handlelength=1.4, fontsize=7.2)
    _panel_title(ax, "(a) Convergencia del simulador")
    sub = stab_df.groupby("T").agg(under=("underflow_lineal", "mean"), acc_lin=("acc_lineal", "mean"), acc_log=("acc_log", "mean"))
    ax = axes[1]
    xs = np.arange(len(sub))
    ax.bar(xs, 100 * sub["under"], width=0.55, color=ORANGE)
    ax.set_xticks(xs, [str(int(t)) for t in sub.index], fontsize=6.6)
    ax.set_xlabel("Longitud T (épocas)")
    ax.set_ylabel("% de noches con underflow")
    ax.set_ylim(0, 105)
    _panel_title(ax, "(b) Viterbi lineal (plantilla)")
    ax = axes[2]
    ax.plot(xs, sub["acc_log"], "o-", color=BLUE, ms=6.5, mec=SURFACE, mew=1.3, label="Viterbi en log-espacio")
    ax.plot(xs, sub["acc_lin"], "o-", color=ORANGE, ms=6.5, mec=SURFACE, mew=1.3, label="Viterbi lineal")
    ax.set_xticks(xs, [str(int(t)) for t in sub.index], fontsize=6.6)
    ax.set_xlabel("Longitud T (épocas)")
    ax.set_ylabel("Exactitud")
    ax.set_ylim(0.3, 0.85)
    ax.legend(loc="lower left", handlelength=1.4)
    _panel_title(ax, "(c) Efecto sobre la exactitud")
    fig.tight_layout(w_pad=1.2)
    return _save(fig, path)
