# Mapeo Basel III FRTB — Outputs del Pipeline

## Referencia: BCBS "Minimum capital requirements for market risk" (enero 2019)

---

## Internal Models Approach (IMA) — Requerimientos cubiertos

| Requerimiento FRTB | Output del pipeline | Archivo |
|---|---|---|
| Expected Shortfall 97.5% (Art. 325bb) | `es_975` | `reports/expected_shortfall.json` |
| Backtesting VaR 99% (Art. 325bf) | Kupiec + Christoffersen | `reports/var_backtest.json` |
| Ventana mínima 250 días (Art. 325bf) | `window_days=250` | `src/validation/var_backtesting.py` |
| Stress scenarios (Art. 325bk) | 6 escenarios históricos + MC | `reports/stress_scenarios/` |
| Horizonte de liquidez por activo (Art. 325bd) | `LIQUIDITY_HORIZONS` | `src/validation/expected_shortfall.py` |
| Traffic light system | Zonas verde/amarilla/roja | `reports/var_backtest.json` |
| P&L attribution test | Pendiente (v1.1) | — |

---

## SR 11-7 (Federal Reserve) — Requerimientos cubiertos

| Pilar SR 11-7 | Implementación | Archivo |
|---|---|---|
| Conceptual soundness | ADRs documentados | `docs/decisions/ADRs.md` |
| Ongoing monitoring | PSI diario + alertas | `src/monitoring/drift.py` |
| Outcomes analysis | Backtesting estadístico | `src/validation/var_backtesting.py` |
| Model card | Auto-generada en HTML | `reports/model_card.html` |
| Audit trail | Log inmutable con hashes | `reports/audit_trail.jsonl` |
| Limitaciones documentadas | Sección 5 del Model Card | `src/governance/model_card.py` |

---

## EU AI Act — Annex III (High-risk AI system)

Este sistema califica como **alto riesgo** bajo Annex III (sistemas de IA usados en
servicios financieros para evaluación de riesgo).

| Artículo EU AI Act | Requerimiento | Cumplimiento |
|---|---|---|
| Art. 9 | Sistema de gestión de riesgos | ✅ SR 11-7 + backtesting |
| Art. 10 | Gobernanza de datos | ✅ Data quality checks |
| Art. 11 | Documentación técnica | ✅ Model Card + ADRs |
| Art. 12 | Registro de logs | ✅ Audit trail inmutable |
| Art. 13 | Transparencia | ✅ Explainability (attention weights) |
| Art. 14 | Supervisión humana | ⚠️ Parcial — requiere proceso aprobación |
| Art. 15 | Exactitud y robustez | ✅ Stress testing + backtesting |

---

## Notas metodológicas

### Por qué ES 97.5% y no VaR 99%

Basel II usaba VaR 99% como métrica principal. Basel III (FRTB) migró a
**Expected Shortfall 97.5%** por dos razones:

1. El ES captura la magnitud de las pérdidas en la cola (no solo el umbral)
2. El ES es sub-aditivo (cumple coherencia de medidas de riesgo — Artzner et al., 1999)
   mientras que el VaR en general no lo es

Matemáticamente: `ES_0.975 = E[Loss | Loss > VaR_0.975]`

### Por qué t-Student y no Normal

Los retornos financieros tienen kurtosis > 3 (colas más pesadas que la normal).
Asumir normalidad subestima el riesgo en eventos extremos (tail underestimation).

La distribución t-Student con grados de libertad bajos (df ≈ 4-6) ajusta
mucho mejor las colas observadas empíricamente en datos de mercado.

### Scaling sqrt-of-time

Basel III permite escalar el VaR/ES de 1 día a horizontes mayores usando
la raíz cuadrada del tiempo, bajo el supuesto de retornos i.i.d.:

`VaR_T = VaR_1 × sqrt(T)`

Este supuesto es una aproximación — en la práctica los retornos tienen
autocorrelación en volatilidad (efecto ARCH), lo que el TFT captura
explícitamente.
