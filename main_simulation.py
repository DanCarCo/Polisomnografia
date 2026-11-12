#!/usr/bin/env python
"""Ejecuta todo el estudio: simulación, análisis estadístico, CSV, figuras e informes.

Uso
---
    python main_simulation.py                # estudio completo (semilla 42, N = 100 noches/escenario)
    python main_simulation.py --quick        # versión rápida para verificar la instalación (~1 min)
    python main_simulation.py --seed 7       # otra semilla maestra
    python main_simulation.py --no-figures   # solo tablas CSV

Salidas
-------
    results/   CSV crudos y resumidos, informes clínicos simulados, entorno y sumas SHA-256
    paper/figures/   figuras PNG a 300 dpi usadas en el artículo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import eeg_signals as eeg  # noqa: E402
from src import experiments as ex  # noqa: E402
from src import sleep_model as sm  # noqa: E402
from src.utils_stats import summarize  # noqa: E402

CSV_KW = dict(index=False, float_format="%.10g", lineterminator="\n")
DEPENDENCIES = ("numpy", "scipy", "pandas", "matplotlib", "hmmlearn", "pytest")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV con saltos de línea LF en cualquier sistema operativo (hashes SHA-256 portables)."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        df.to_csv(fh, **CSV_KW)


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f} s] {msg}", flush=True)


def write_environment(outdir: Path, args) -> None:
    versions = {}
    for pkg in DEPENDENCIES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    env = {"python": platform.python_version(), "implementacion": platform.python_implementation(),
           "sistema": f"{platform.system()} {platform.release()}", "librerias": versions,
           "semilla_maestra": args.seed, "repeticiones_por_escenario": args.reps,
           "modo_rapido": bool(args.quick)}
    (outdir / "environment.json").write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def write_checksums(outdir: Path) -> None:
    """SHA-256 de los CSV, informes y resumen clave (el entorno y las figuras dependen de versiones)."""
    files = sorted(list(outdir.glob("*.csv")) + list((outdir / "informes").glob("*.md")) + [outdir / "resumen_clave.json"])
    lines = []
    for f in files:
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        lines.append(f"{digest}  {f.relative_to(outdir).as_posix()}")
    (outdir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=ex.MASTER_SEED, help="semilla maestra (por defecto 42)")
    ap.add_argument("--reps", type=int, default=100, help="noches simuladas por escenario")
    ap.add_argument("--learn-reps", type=int, default=10, help="réplicas del experimento Baum-Welch")
    ap.add_argument("--quick", action="store_true", help="versión reducida para comprobar la instalación")
    ap.add_argument("--outdir", default=str(ROOT / "results"))
    ap.add_argument("--figdir", default=str(ROOT / "paper" / "figures"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    quick = args.quick
    if quick:
        args.reps, args.learn_reps = min(args.reps, 20), min(args.learn_reps, 2)
    n_seeds = 5 if quick else 20
    conv_chains, conv_lengths = (3, [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5]) if quick else (10, [10 ** k for k in range(2, 7)])
    stab_reps = 10 if quick else 50
    patho_n, patho_ref, sev_n, sev_boot = (30, 60, 30, 100) if quick else (100, 300, 100, 500)
    sweep_reps = args.reps

    outdir, figdir = Path(args.outdir), Path(args.figdir)
    (outdir / "informes").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seed = args.seed
    boot_rng = np.random.default_rng(ex.seed_sequence(seed, "bootstrap"))
    key_numbers: dict = {}

    def save(df: pd.DataFrame, name: str) -> None:
        write_csv(df, outdir / name)

    write_environment(outdir, args)
    log(f"Semilla maestra = {seed}; N = {args.reps} noches por escenario; modo rápido = {quick}", t0)

    # ------------------------------------------------------------ propiedades teóricas
    save(ex.chain_properties(), "propiedades_cadena.csv")

    # ------------------------------------------------------------ escenarios E1-E5
    scenarios = ex.build_scenarios()
    frames, extras = [], {}
    for key, scn in scenarios.items():
        df, ext = ex.run_scenario(scn, args.reps, master_seed=seed)
        frames.append(df)
        extras[key] = ext
        log(f"{key} {scn.name}: exactitud Viterbi = {df.acc_viterbi.mean():.4f}", t0)
    df_all = pd.concat(frames, ignore_index=True)
    save(df_all, "escenarios_por_replica.csv")

    metrics = (["acc_viterbi", "acc_posterior", "acc_baseline", "acc_oracle", "kappa_viterbi", "f1_viterbi",
                "kappa_baseline", "gain_vs_baseline", "loglik_per_epoch"]
               + [f"recall_{x}_viterbi" for x in ("W", "L", "D", "R")]
               + [f"{p}_{k}" for k in ex.CLIN_KEYS for p in ("true", "vit")])
    summ = ex.summarize_table(df_all, ["scenario"], metrics, boot_rng)
    save(summ, "escenarios_resumen.csv")
    save(ex.compare_scenarios(df_all), "escenarios_comparaciones.csv")
    save(ex.compare_decoders(df_all), "escenarios_decodificadores_pareados.csv")

    cm_rows = []
    for key, ext in extras.items():
        for dec, cm in (("viterbi", ext["cm_viterbi"]), ("baseline", ext["cm_baseline"])):
            for i in range(4):
                for j in range(4):
                    cm_rows.append({"scenario": key, "decodificador": dec, "real": sm.STATE_NAMES[i],
                                    "estimada": sm.STATE_NAMES[j], "conteo": int(cm[i, j])})
    save(pd.DataFrame(cm_rows), "matrices_confusion.csv")
    key_numbers["escenarios"] = {
        k: {"acc_viterbi": float(df_all[df_all.scenario == k].acc_viterbi.mean()),
            "acc_baseline": float(df_all[df_all.scenario == k].acc_baseline.mean())} for k in scenarios}

    # ------------------------------------------------------------ barridos de sensibilidad
    levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    sweeps = pd.concat([ex.run_sweep("A_mix", levels, sweep_reps, seed),
                        ex.run_sweep("B_mix", levels, sweep_reps, seed),
                        ex.run_sweep("persistence", [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98], sweep_reps, seed)], ignore_index=True)
    save(sweeps, "sensibilidad_por_replica.csv")
    save(ex.summarize_table(sweeps, ["factor", "level"], ["acc_viterbi", "acc_posterior", "acc_baseline", "acc_oracle"], boot_rng),
         "sensibilidad_resumen.csv")
    log("Barridos de sensibilidad (A, B, persistencia) completados", t0)

    # ------------------------------------------------------------ condición inicial y semillas
    ic = ex.run_initial_conditions(sweep_reps, seed)
    save(ic, "condiciones_iniciales_por_replica.csv")
    save(ex.summarize_table(ic, ["estado_inicial", "decodificador"], ["acc_primeras", "acc_resto", "acc_total"]),
         "condiciones_iniciales_resumen.csv")
    seeds_df = ex.run_seed_sensitivity(n_seeds, args.reps, first_seed=seed)
    save(seeds_df, "sensibilidad_semillas.csv")
    key_numbers["entre_semillas"] = {"sd_medias": float(seeds_df.mean_acc_viterbi.std(ddof=1)),
                                     "media_de_medias": float(seeds_df.mean_acc_viterbi.mean())}
    log("Condición inicial y sensibilidad a la semilla completadas", t0)

    # ------------------------------------------------------------ aprendizaje (Baum-Welch)
    learn_df, hist_df, learned_models = ex.run_learning(args.learn_reps, tol=1e-5, master_seed=seed)
    save(learn_df, "aprendizaje_baum_welch.csv")
    save(hist_df, "aprendizaje_historial_ll.csv")
    save(ex.summarize_table(learn_df, ["init", "n_train"], ["acc_test_learned", "acc_test_true", "acc_test_baseline", "mae_A", "mae_B",
                                                    "mae_pi", "iterations", "loglik_test_per_epoch_learned",
                                                    "loglik_test_per_epoch_true"], boot_rng), "aprendizaje_resumen.csv")
    log("Baum-Welch completado", t0)

    # ------------------------------------------------------------ patologías simuladas
    patho_df, patho_cm, patho_sum, ref = ex.run_pathology(patho_n, patho_ref, seed)
    save(patho_df, "patologias_por_noche.csv")
    write_csv(patho_cm.reset_index(), outdir / "patologias_confusion.csv")
    save(patho_sum, "patologias_resumen.csv")
    save(pd.DataFrame([{"metrica": k, "media_ref": v[0], "sd_ref": v[1]} for k, v in ref.items()]), "patologias_referencia.csv")
    sev_df = ex.run_pathology_severity(sev_n, seed, n_boot=sev_boot)
    save(sev_df, "patologias_severidad.csv")
    for name, text in ex.build_example_reports(seed, ref=ref).items():
        slug = name.lower().replace(" ", "_").replace("ñ", "n")
        (outdir / "informes" / f"informe_{slug}.md").write_text(text, encoding="utf-8", newline="\n")
    log("Patologías simuladas completadas", t0)

    # ------------------------------------------------------------ validación del simulador
    conv_df = ex.run_simulator_convergence(conv_chains, conv_lengths, seed)
    save(conv_df, "convergencia_simulador.csv")
    stab_df = ex.run_numerical_stability([50, 100, 200, 300, 400, 480, 600, 1000], stab_reps, seed)
    save(stab_df, "estabilidad_numerica.csv")
    cm_eeg = eeg.extraction_consistency(50 if quick else 250, np.random.default_rng(ex.seed_sequence(seed, "extraction")))
    save(pd.DataFrame([{"simbolo_generado": sm.OBS_NAMES[i], "simbolo_extraido": sm.OBS_NAMES[j], "conteo": int(cm_eeg[i, j])}
                       for i in range(4) for j in range(4)]), "consistencia_extraccion_senales.csv")
    key_numbers["extraccion_exactitud"] = float(np.trace(cm_eeg) / cm_eeg.sum())
    log("Validación del simulador completada", t0)

    (outdir / "resumen_clave.json").write_text(json.dumps(key_numbers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    # ------------------------------------------------------------ figuras
    if not args.no_figures:
        from src import plotting as pl

        figdir.mkdir(parents=True, exist_ok=True)
        nom = sm.nominal_hmm()
        e1 = extras["E1"]
        idx = ex.representative_night(df_all[df_all.scenario == "E1"], e1)
        states, obs, vit = e1["states"][idx], e1["obs"][idx], e1["vit"][idx]
        freqs, spec = eeg.night_spectrogram(obs, np.random.default_rng(ex.seed_sequence(seed, "spectrogram")))
        acc_night = float(np.mean(states == vit))
        pl.fig_flow(figdir / "fig01_flujo_metodologico.png")
        pl.fig_model(figdir / "fig02_grafo_hmm.png", nom)
        pl.fig_matrices(figdir / "fig08_matrices_A_B.png", nom.A, nom.B)
        pl.fig_night(figdir / "fig03_hipnograma_noche.png", states, vit, freqs, spec, acc=acc_night)
        pl.fig_eeg(figdir / "fig10_senales_eeg.png", np.random.default_rng(ex.seed_sequence(seed, "fig_eeg")))
        pl.fig_scenarios(figdir / "fig04_escenarios.png", df_all, summ)
        pl.fig_confusion(figdir / "fig09_matrices_confusion.png", [extras["E1"]["cm_viterbi"], extras["E5"]["cm_viterbi"]],
                         ["(a) E1 nominal (Viterbi)", "(b) E5 sueño consolidado (Viterbi)"])
        pl.fig_sweeps(figdir / "fig05_sensibilidad.png", sweeps)
        pl.fig_learning(figdir / "fig06_baum_welch.png", learn_df, hist_df)
        pl.fig_learned(figdir / "fig11_matrices_aprendidas.png", nom, learned_models)
        pl.fig_pathology(figdir / "fig07_patologias.png", patho_df, sev_df)
        pl.fig_validation(figdir / "fig12_validacion_simulador.png", conv_df, stab_df)
        (outdir / "noche_representativa.json").write_text(
            json.dumps({"escenario": "E1", "replica": int(idx), "criterio": "exactitud de Viterbi más cercana a la mediana",
                        "exactitud": acc_night}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        log(f"Figuras guardadas en {figdir}", t0)

    if not args.no_figures:
        from src import latex_export

        names = latex_export.export_all(outdir, figdir.parent / "auto")
        log(f"Tablas y {len(names)} macros de LaTeX exportadas a {figdir.parent / 'auto'}", t0)

    write_checksums(outdir)
    log("Listo. Sumas SHA-256 en results/SHA256SUMS.txt", t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
