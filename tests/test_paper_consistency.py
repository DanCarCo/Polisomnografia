"""Consistencia del manuscrito LaTeX sin necesidad de compilarlo.

Comprueba que citas, referencias cruzadas, figuras, tablas automáticas y macros
de resultados existan, que la bibliografía siga el orden de primera cita (estilo
IEEE) y que haya al menos 10 referencias de los últimos 5 años.
"""
import re
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[1] / "paper"
MAIN = PAPER / "main.tex"
pytestmark = pytest.mark.skipif(not MAIN.exists(), reason="paper/main.tex aún no existe")


def _tex() -> str:
    text = MAIN.read_text(encoding="utf-8")
    return "\n".join(line.split("%")[0] if not line.lstrip().startswith("\\%") else line
                     for line in re.sub(r"\\%", "<PCT>", text).splitlines()).replace("<PCT>", "%")


def _bib(text):
    body = text[text.index(r"\begin{thebibliography}"):text.index(r"\end{thebibliography}")]
    items = re.split(r"\\bibitem\{", body)[1:]
    return [(it.split("}")[0], it.split("}", 1)[1]) for it in items]


def test_citas_y_bibliografia_coinciden_y_siguen_el_orden_de_primera_cita():
    text = _tex()
    bib = _bib(text)
    keys = [k for k, _ in bib]
    assert len(keys) == len(set(keys)), "claves duplicadas en la bibliografía"
    main = text[:text.index(r"\begin{thebibliography}")]
    cited = []
    for group in re.findall(r"\\cite\{([^}]*)\}", main):
        for k in (x.strip() for x in group.split(",")):
            assert k in keys, f"cita sin entrada en la bibliografía: {k}"
            if k not in cited:
                cited.append(k)
    appendix = text[text.index(r"\end{thebibliography}"):]
    for group in re.findall(r"\\cite\{([^}]*)\}", appendix):
        for k in (x.strip() for x in group.split(",")):
            assert k in keys
            if k not in cited:
                cited.append(k)
    assert set(keys) == set(cited), f"entradas sin citar: {set(keys) - set(cited)}"
    assert keys == cited, "la bibliografía debe ir en orden de primera cita (IEEE)"


def test_al_menos_diez_referencias_de_los_ultimos_cinco_anios():
    recent = [k for k, body in _bib(_tex()) if (m := re.search(r"\b(20\d\d)\b", body)) and int(m.group(1)) >= 2021]
    assert len(recent) >= 10, f"solo {len(recent)} referencias de 2021 en adelante"


def test_referencias_cruzadas_y_etiquetas():
    text = _tex()
    labels = re.findall(r"\\label\{([^}]*)\}", text) + re.findall(r"label=\{([^}]*)\}", text)  # incluye lstlisting
    assert len(labels) == len(set(labels)), "etiquetas duplicadas"
    for ref in re.findall(r"\\(?:ref|eqref|autoref)\{([^}]*)\}", text):
        assert ref in labels, f"\\ref sin \\label: {ref}"


def test_figuras_y_tablas_existen():
    text = _tex()
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", text):
        assert (PAPER / "figures" / name).exists() or (PAPER / name).exists(), f"falta la figura {name}"
    for name in re.findall(r"\\input\{([^}]*)\}", text):
        assert (PAPER / name).exists() or (PAPER / f"{name}.tex").exists(), f"falta {name}"


def test_macros_de_resultados_usadas_estan_definidas():
    text = _tex()
    macros = (PAPER / "auto" / "macros.tex").read_text(encoding="utf-8")
    defined = set(re.findall(r"\\csname ([^\\]*)\\endcsname\{", macros))
    pat = {"R": ("R", 3), "Sw": ("S", 4), "Lr": ("L", 4), "Ic": ("I", 4), "K": ("K", 1)}   # comando -> (prefijo, nº de argumentos)
    for cmd, (prefix, n) in pat.items():
        for m in re.finditer(rf"\\{cmd}((?:\{{[^}}]*\}}){{{n}}})(?![A-Za-z])", text):
            args = re.findall(r"\{([^}]*)\}", m.group(1))
            name = f"{prefix}@" + "@".join(args)
            assert name in defined, f"macro de resultado no definida: \\{cmd}{m.group(1)}  ({name})"


def test_entornos_balanceados():
    text = _tex()
    begins = re.findall(r"\\begin\{([^}]*)\}", text)
    ends = re.findall(r"\\end\{([^}]*)\}", text)
    for env in set(begins) | set(ends):
        assert begins.count(env) == ends.count(env), f"entorno desbalanceado: {env}"
    assert text.count("{") - text.count(r"\{") == text.count("}") - text.count(r"\}"), "llaves desbalanceadas"
