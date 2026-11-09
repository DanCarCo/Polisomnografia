# Informe de polisomnografía simulada — Paciente simulado 2

> **Aviso:** resultado de una simulación con parámetros didácticos. No constituye un diagnóstico médico.

## Arquitectura del sueño (hipnograma decodificado con Viterbi)

| Indicador | Valor |
|---|---|
| Tiempo en cama (TIB) | 480 min |
| Tiempo total de sueño (TST) | 351 min |
| Eficiencia del sueño | 73.1 % |
| Latencia de sueño | 1 min |
| Latencia REM | 7 min |
| Vigilia tras el inicio (WASO) | 128 min |
| Despertares (≥1 min / ≥3 min) | 44 / 23 |
| Ciclos NREM→REM completos | 10 |
| Cambios de etapa por hora | 20.1 |

| Etapa | Minutos | % |
|---|---|---|
| Despierto | 129 | 26.9 % del TIB |
| Sueño Ligero | 231 | 65.8 % del TST |
| Sueño Profundo | 62 | 17.7 % del TST |
| REM | 58 | 16.5 % del TST |

## Alteraciones respecto al modelo de referencia (|z| ≥ 2)

- **eficiencia del sueño reducida** (valor = 73.1, z = -4.6)
- **vigilia intrasueño (WASO) aumentada** (valor = 128.0, z = +4.6)
- **despertares nocturnos aumentados** (valor = 44.0, z = +4.6)
- **porcentaje de sueño profundo reducido** (valor = 17.7, z = -2.4)
- **alta frecuencia de cambios de etapa** (valor = 20.1, z = +2.6)

## Probabilidad a posteriori de cada modelo (Forward, a priori uniforme)

| Modelo | P(modelo \| observaciones) |
|---|---|
| Normal | 0.0000 |
| Sueño fragmentado | 1.0000 |
| Sin REM | 0.0000 |

**Diagnóstico probable (simulado):** Sueño fragmentado.

*Etiqueta verdadera de la simulación (solo para validar): Sueño fragmentado.*
