# Informe de polisomnografía simulada — Paciente simulado 3

> **Aviso:** resultado de una simulación con parámetros didácticos. No constituye un diagnóstico médico.

## Arquitectura del sueño (hipnograma decodificado con Viterbi)

| Indicador | Valor |
|---|---|
| Tiempo en cama (TIB) | 480 min |
| Tiempo total de sueño (TST) | 442 min |
| Eficiencia del sueño | 92.1 % |
| Latencia de sueño | 2 min |
| Latencia REM | 35 min |
| Vigilia tras el inicio (WASO) | 36 min |
| Despertares (≥1 min / ≥3 min) | 13 / 4 |
| Ciclos NREM→REM completos | 2 |
| Cambios de etapa por hora | 15.4 |

| Etapa | Minutos | % |
|---|---|---|
| Despierto | 38 | 7.9 % del TIB |
| Sueño Ligero | 228 | 51.6 % del TST |
| Sueño Profundo | 199 | 45.0 % del TST |
| REM | 15 | 3.4 % del TST |

## Alteraciones respecto al modelo de referencia (|z| ≥ 2)

- **porcentaje de REM reducido** (valor = 3.4, z = -4.5)

## Probabilidad a posteriori de cada modelo (Forward, a priori uniforme)

| Modelo | P(modelo \| observaciones) |
|---|---|
| Normal | 0.0000 |
| Sueño fragmentado | 0.0000 |
| Sin REM | 1.0000 |

**Diagnóstico probable (simulado):** Sin REM.

*Etiqueta verdadera de la simulación (solo para validar): Sin REM.*
