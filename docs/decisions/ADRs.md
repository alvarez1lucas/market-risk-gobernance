# Architectural Decision Records (ADRs)
## Market Risk Deep Learning Suite

---

## ADR-001: Elección de TFT como modelo champion

**Fecha:** 2025-01-01  
**Estado:** Aprobado  
**Decisores:** Risk Engineering Team

### Contexto
Se evaluaron 3 arquitecturas para forecasting de VaR:
- GARCH(1,1) con distribución t-Student (benchmark regulatorio)
- LSTM con mecanismo de atención (Bahdanau)
- Temporal Fusion Transformer (Lim et al., 2021)

### Decisión
**Temporal Fusion Transformer (TFT)** como modelo champion, LSTM como challenger,
GARCH como benchmark regulatorio obligatorio.

### Justificación

| Criterio | GARCH | LSTM+Att | TFT |
|---|---|---|---|
| Producción de cuantiles directos | ✅ (paramétrico) | ✅ (pinball loss) | ✅ (multi-cuantil nativo) |
| Manejo de múltiples horizontes | ❌ | Parcial | ✅ nativo |
| Explainability (SR 11-7) | ✅ parámetros | ✅ attention weights | ✅ attention + variable importance |
| Variables conocidas en el futuro | ❌ | ❌ | ✅ |
| Rendimiento empírico en crisis | Pobre (normal implícita) | Bueno | Mejor |
| Aceptación regulatoria | Alta (benchmark) | Media | Alta (bien fundamentado) |

### Consecuencias
- El pipeline siempre compara TFT vs GARCH en backtesting
- Si el TFT no pasa Kupiec, se promueve el LSTM como champion alternativo
- El GARCH permanece en producción como fallback regulatorio

---

## ADR-002: Distribución t-Student en lugar de Normal

**Fecha:** 2025-01-05  
**Estado:** Aprobado

### Contexto
Los retornos financieros exhiben leptocurtosis (colas más pesadas que la normal).
Basel III implícitamente asume esto al requerir ES en lugar de VaR.

### Decisión
Usar distribución **t-Student con grados de libertad estimados** en todos los modelos
donde se asume una distribución paramétrica (GARCH, Historical Simulation fallback).

### Evidencia
- Kurtosis empírica del SPX (2000-2024): ~8.3 vs 3.0 de la normal
- El colapso de LTCM (1998) ocurrió porque sus modelos asumían normalidad
- Basel III migró a ES 97.5% precisamente para capturar mejor la cola

### Consecuencias
- GARCH se estima con `dist='t'` en arch library
- Monte Carlo usa t-Student con df=5 (calibrado empíricamente)
- QQ-plot vs t-Student incluido en notebook 01_eda

---

## ADR-003: Fuentes de datos gratuitas (yfinance + FRED)

**Fecha:** 2025-01-10  
**Estado:** Aprobado

### Contexto
El proyecto debe ser reproducible sin licencias de pago (Bloomberg, Refinitiv).

### Decisión
**yfinance** para precios de activos + **FRED API** (clave gratuita) para macro.
Datos sintéticos calibrados para stress scenarios.

### Limitaciones aceptadas
- yfinance puede tener gaps en datos históricos — se aplica forward fill
- FRED es mensual/semanal para algunas series — se interpola a diario
- Datos pre-2000 con menor calidad — período de entrenamiento desde 2000

### Consecuencias
- `src/data/ingest.py` incluye fallback a datos sintéticos si FRED no disponible
- Data quality checks automáticos en cada ejecución
- README documenta cómo obtener FRED API key gratis

---

## ADR-004: Pinball Loss para estimación de cuantiles

**Fecha:** 2025-01-15  
**Estado:** Aprobado

### Contexto
Para estimar VaR directamente sin asumir distribución, se necesita una función de
pérdida que optimice cuantiles específicos.

### Decisión
**Quantile Loss (Pinball Loss)** para entrenar el LSTM y como objetivo secundario del TFT.

```
L(q, y, ŷ) = q·max(y-ŷ, 0) + (1-q)·max(ŷ-y, 0)
```

### Justificación
- Sin supuestos distribucionales — más robusto en colas
- Multi-cuantil simultáneo — un solo forward pass produce [1%, 2.5%, 5%, 50%, 95%, 97.5%, 99%]
- Cumple SR 11-7: el modelo predice explícitamente lo que se valida en backtesting

---

## ADR-005: Arquitectura de 4 repos (governance como submódulo)

**Fecha:** 2025-01-20  
**Estado:** Aprobado

### Contexto
El portfolio tiene 3 proyectos de riesgo más una capa transversal de governance.

### Decisión
```
github.com/[usuario]/
├── credit-risk-model-validation/    ← Proyecto 1
├── market-risk-deep-learning/       ← Proyecto 2 (este repo)
└── ai-governance-framework/         ← Governance transversal
    ├── submodules/credit-risk       → apunta a Proyecto 1
    └── submodules/market-risk       → apunta a Proyecto 2
```

### Justificación
- Governance como submódulo evita duplicación de código
- Cada proyecto es auto-contenido y desplegable de forma independiente
- El repo de governance demuestra capacidad de integración end-to-end
- Los hiring managers pueden ver el governance repo como "prueba de madurez"

### Consecuencias
- `src/governance/` en este repo es un subset del governance central
- El Model Risk Register vive en el repo de governance, no aquí
- CI/CD del governance repo valida los outputs de este repo como checks
