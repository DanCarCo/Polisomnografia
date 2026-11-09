# Informe de polisomnografía simulada — Paciente simulado 1

> **Aviso:** resultado de una simulación con parámetros didácticos. No constituye un diagnóstico médico.

## Arquitectura del sueño (hipnograma decodificado con Viterbi)

| Indicador | Valor |
|---|---|
| Tiempo en cama (TIB) | 480 min |
| Tiempo total de sueño (TST) | 426 min |
| Eficiencia del sueño | 88.8 % |
| Latencia de sueño | 0 min |
| Latencia REM | 8 min |
| Vigilia tras el inicio (WASO) | 54 min |
| Despertares (≥1 min / ≥3 min) | 22 / 10 |
| Ciclos NREM→REM completos | 15 |
| Cambios de etapa por hora | 15.8 |

| Etapa | Minutos | % |
|---|---|---|
| Despierto | 54 | 11.2 % del TIB |
| Sueño Ligero | 264 | 62.0 % del TST |
| Sueño Profundo | 83 | 19.5 % del TST |
| REM | 79 | 18.5 % del TST |

## Alteraciones respecto al modelo de referencia (|z| ≥ 2)

- Sin desviaciones relevantes respecto a la distribución de referencia.

## Probabilidad a posteriori de cada modelo (Forward, a priori uniforme)

| Modelo | P(modelo \| observaciones) |
|---|---|
| Normal | 0.9990 |
| Sueño fragmentado | 0.0010 |
| Sin REM | 0.0000 |

**Diagnóstico probable (simulado):** Normal.

*Etiqueta verdadera de la simulación (solo para validar): Normal.*
