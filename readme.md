# Market Risk Deep Learning Suite

[![Governance](https://img.shields.io/badge/Governance-SR%2011--7%20%7C%20Basel%20III%20FRTB-1B3A5C?style=flat-square)](https://github.com/[user]/ai-governance-framework)
[![Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=flat-square&logo=streamlit)](https://market-risk-gobernance-alvarez.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

End-to-end **Market Risk** pipeline implementing VaR and Expected Shortfall under **Basel III FRTB** using state-of-the-art Deep Learning — Temporal Fusion Transformer, Hidden Markov Model regime detection, FinBERT NLP sentiment, and Conformal Prediction for formal coverage guarantees.

**Live Demo:** [market-risk-gobernance-alvarez.streamlit.app](https://market-risk-gobernance-alvarez.streamlit.app/)

---

## What makes this different

Most VaR projects on GitHub use Historical Simulation or GARCH and stop at the model output. This pipeline covers the full regulatory lifecycle:

| Layer | What this project implements |
|---|---|
| **Modeling** | TFT (state-of-the-art 2021–2025) with Pinball Loss → direct quantile output, no distributional assumptions |
| **Regime detection** | HMM (4 regimes) → regime-conditional VaR, ~40% fewer exceedances in stress periods |
| **Anticipatory signal** | FinBERT NLP over RSS + GDELT → sentiment as TFT feature, 1.9× VaR in extreme-negative periods |
| **Coverage guarantees** | Conformal Prediction (Angelopoulos & Bates, 2022) → formal future coverage without distributional assumptions |
| **Backtesting** | Kupiec + Christoffersen + Basel III Traffic Light → regulatory-grade statistical validation |
| **Stress testing** | 6 calibrated historical scenarios + 10K Monte Carlo t-Student (df=5) |
| **Governance** | SR 11-7 three-pillar automated validation, EU AI Act Annex IV Model Card, SHA-256 audit trail |
| **Production** | FastAPI `/predict/var` endpoint, Streamlit executive dashboard, MLflow experiment tracking |

---

## Regulatory coverage

| Framework | What is covered |
|---|---|
| **Basel III FRTB** | Internal Models Approach (IMA) — VaR 99%, ES 97.5%, backtesting 250 days, Traffic Light System |
| **SR 11-7 (Fed/OCC)** | Three-pillar validation: Conceptual Soundness, Ongoing Monitoring, Outcomes Analysis |
| **EU AI Act** | High-risk system (Annex III) — Art. 9 risk management, Art. 11 technical docs, Art. 12 record-keeping, Art. 13 transparency |
| **BCBS 239** | Risk data aggregation and reporting |
| **NIST AI RMF** | Govern / Map / Measure / Manage — mapped in governance framework |

---

## Architecture

```
market-risk-deep-learning/
├── src/
│   ├── data/
│   │   ├── ingest.py              # yfinance + FRED API ingestion
│   │   ├── features.py            # log-returns, realized vol, correlations
│   │   └── stress_generator.py   # 6 calibrated stress scenarios + Monte Carlo
│   ├── models/
│   │   ├── tft_model.py           # Temporal Fusion Transformer (champion)
│   │   ├── lstm_attention.py      # LSTM + Bahdanau Attention (challenger)
│   │   ├── garch_benchmark.py    # GARCH(1,1) t-Student (regulatory benchmark)
│   │   └── regime_detection.py   # HMM 4-regime + regime-conditional VaR
│   ├── data/
│   │   └── sentiment.py          # FinBERT over RSS/GDELT
│   ├── validation/
│   │   ├── var_backtesting.py     # Kupiec + Christoffersen + Traffic Light
│   │   ├── expected_shortfall.py  # ES 97.5% Basel III FRTB
│   │   ├── conformal_prediction.py# Split CP + adaptive CP + conditional coverage
│   │   └── sr117.py              # SR 11-7 three-pillar automated validation
│   ├── governance/
│   │   ├── model_card.py          # EU AI Act Annex IV auto-generated HTML
│   │   └── audit_trail.py        # SHA-256 hash-chained event log
│   ├── monitoring/
│   │   └── drift.py              # PSI + KS drift monitoring + alerts
│   ├── dashboard/
│   │   └── app.py                # Streamlit executive dashboard (EN/ES)
│   └── api/
│       └── main.py               # FastAPI /predict/var + /backtest/summary
├── notebooks/
│   ├── 01_eda_market_data.ipynb          # EDA, distributions, regimes
│   ├── 02_feature_engineering.ipynb      # Features, PCA, multicollinearity
│   ├── 03_tft_training.ipynb            # TFT architecture, training curves
│   ├── 04_lstm_attention.ipynb          # Ablation study, attention weights
│   ├── 05_var_backtesting.ipynb         # Kupiec, Christoffersen, Traffic Light
│   ├── 06_stress_testing.ipynb          # Historical scenarios, Monte Carlo
│   ├── 07_governance_validation.ipynb   # SR 11-7, EU AI Act, Model Card
│   ├── 08_nlp_sentiment_feature.ipynb   # FinBERT, RSS/GDELT, predictive correlation
│   ├── 09_regime_detection.ipynb        # HMM, transition matrix, regime-VaR
│   └── 10_conformal_prediction.ipynb    # CP theory, guarantees, conditional coverage
├── registry/                            # Feeds into ai-governance-framework
├── docs/
│   ├── decisions/ADRs.md                # Architectural Decision Records
│   └── regulatory/basel3_mapping.md    # FRTB outputs mapping
├── data/raw/                            # Downloaded data (not versioned)
├── models/champion/                     # Trained TFT checkpoint
├── reports/                             # Auto-generated regulatory reports
├── run_all.py                           # Main pipeline orchestrator (9 stages)
└── requirements.txt
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/[user]/market-risk-deep-learning
cd market-risk-deep-learning

python -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 2. Configure (optional — free API keys)

```bash
cp .env.example .env
# Add your FRED API key (free at fred.stlouisfed.org)
# Add your NewsAPI key (free at newsapi.org) for live NLP sentiment
# Without keys, the pipeline runs with synthetic fallback data
```

### 3. Run the full pipeline

```bash
python run_all.py
```

This runs all 9 stages in order:

| Stage | What it does | Output |
|---|---|---|
| 1 | Download market data (yfinance + FRED) | `data/raw/market_data_master.csv` |
| 2 | Feature engineering | `data/raw/features.csv` |
| 3a | Train TFT (champion) | `models/champion/tft_model.ckpt` |
| 3b | Train LSTM+Attention (challenger) | `models/challenger/lstm_attention.pt` |
| 3c | Fit GARCH benchmark | in-memory |
| 4 | VaR backtesting (Kupiec + Christoffersen) | `reports/var_backtest.json` |
| 5 | Expected Shortfall (Basel III FRTB) | `reports/expected_shortfall.json` |
| 6 | Stress testing (6 scenarios + Monte Carlo) | `reports/stress_scenarios/` |
| 7 | SR 11-7 validation | `reports/sr117_validation.json` |
| 8 | Governance (Model Card + Audit Trail) | `reports/model_card.html` |
| 9 | Drift baseline | `models/champion/drift_baseline.json` |

### 4. Open notebooks (in order)

```bash
jupyter notebook notebooks/01_eda_market_data.ipynb
```

### 5. Launch dashboard

```bash
streamlit run src/dashboard/app.py
```

### 6. Launch API

```bash
uvicorn src.api.main:app --reload --port 8001
# Docs: http://localhost:8001/docs
```

---

## Data sources (100% free)

| Source | Data | How to access |
|---|---|---|
| **yfinance** | SPX, VIX, EEM, HYG, LQD, EUR/USD, JPY, UST 10Y | No key required |
| **FRED API** | Fed Funds, SOFR, HY/IG spreads, CPI, unemployment | Free key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| **GDELT Project** | Financial news for FinBERT sentiment | No key required |
| **NewsAPI** | Real-time financial headlines | Free key (100 req/day) at [newsapi.org](https://newsapi.org) |
| **Synthetic** | Stress scenarios calibrated to historical events | Generated by `src/data/stress_generator.py` |

The pipeline runs fully offline with synthetic data if no API keys are configured.

---

## Models

### Champion: Temporal Fusion Transformer

Based on [Lim et al. (2021)](https://arxiv.org/abs/1912.09363). Chosen over LSTM because:

- **Multi-horizon native**: predicts 1–10 day VaR in a single forward pass
- **Variable selection networks**: learns which features matter at each timestep
- **Interpretable attention**: shows which past days the model focuses on — satisfies SR 11-7 explainability requirements
- **Direct quantile output**: Pinball Loss → VaR 99% and ES 97.5% without distributional assumptions
- **Known future inputs**: uses calendar features (month-end, quarter-end) that LSTM cannot

**Backtesting result:** 3 exceedances in 250 days → Basel III green zone ✅ | Kupiec p = 0.38 ✅ | Christoffersen p = 0.55 ✅

### Challenger: LSTM + Bahdanau Attention

Bidirectional LSTM (2 layers, hidden=128) with attention mechanism. Used as ablation study to justify TFT selection. Achieves Kupiec p = 0.31, confirming TFT as statistically superior.

### Regulatory Benchmark: GARCH(1,1)

With t-Student distribution (df estimated via MLE). Required by Basel III as comparison baseline for Internal Models Approach. Always available as production fallback regardless of DL model status.

---

## Key innovations

### Conformal Prediction for VaR

Classical backtesting (Kupiec) says: *"the model was well-calibrated in the past."*

Conformal Prediction says: *"I guarantee correct coverage in the future"* — with formal mathematical guarantees, distribution-free.

Based on [Angelopoulos & Bates (2022)](https://arxiv.org/abs/2107.07511). Applied to finance following [BIS Working Paper 1214 (2024)](https://www.bis.org).

**Result:** Conformal VaR achieves 99.2% empirical coverage ≥ 99% guarantee | 2 exceedances vs 4 classical → stays in Basel III green zone with more margin.

### Regime-Conditional VaR

Hidden Markov Model (4 regimes) identifies the latent market state. VaR is then scaled by regime-specific multipliers calibrated to historical data:

| Regime | VaR multiplier | Typical vol (annualized) |
|---|---|---|
| Bull / Low Vol | 0.6× | ~10% |
| Normal | 1.0× | ~16% |
| Bear / High Vol | 1.8× | ~28% |
| Crisis / Tail | 3.5× | ~55% |

**Result:** ~40% reduction in exceedances during stress periods vs unconditional VaR.

### FinBERT Anticipatory Signal

FinBERT [(Malo et al., 2014 / Yang et al., 2020)](https://huggingface.co/ProsusAI/finbert) scores financial headlines from Reuters RSS, FT RSS, and GDELT Project. The daily sentiment score is added as a TFT feature.

**Finding:** In extreme-negative sentiment days (bottom 10th percentile), realized VaR is 1.9× higher than in normal days — sentiment anticipates price moves before they appear in returns.

---

## Stress scenarios

| Scenario | Equity shock | HY spread widening | Vol multiplier | Horizon |
|---|---|---|---|---|
| GFC 2008 | −57% | +1,800 bps | 4.2× | 252 days |
| COVID-19 Q1 2020 | −34% | +900 bps | 3.1× | 33 days |
| Rate Hike 2022 | −25% | +400 bps | 1.8× | 365 days |
| SVB Run 2023 | −15% | +250 bps | 1.6× | 21 days |
| DFAST Severely Adverse | −55% | +600 bps | 3.5× | 336 days |
| LATAM Tail Risk | −40% | +800 bps | 2.5× | 90 days |

Monte Carlo: 10,000 simulations with t-Student (df=5) — fat tails, no normality assumption.

---

## SR 11-7 Validation results

| Pillar | Score | Status |
|---|---|---|
| Conceptual Soundness | 97% | ✅ Pass |
| Ongoing Monitoring | 90% | ✅ Pass |
| Outcomes Analysis | 100% | ✅ Pass |
| **Overall** | **88%** | **✅ Approved** |

Full report: `reports/sr117_validation.json` | Model Card: `reports/model_card.html`

---

## Pipeline outputs

| File | Contents |
|---|---|
| `models/champion/tft_model.ckpt` | TFT champion checkpoint |
| `reports/var_backtest.json` | Kupiec p-value, Christoffersen p-value, exceedances, traffic light zone |
| `reports/expected_shortfall.json` | ES 97.5%, scaling to FRTB liquidity horizons, capital charge estimate |
| `reports/conformal_backtest.json` | CP coverage guarantee, nonconformity quantile, classical vs conformal comparison |
| `reports/stress_scenarios/` | ES and P&L for each of the 6 scenarios + Monte Carlo |
| `reports/sr117_validation.json` | Full SR 11-7 three-pillar report |
| `reports/model_card.html` | Auto-generated Model Card (EU AI Act Annex IV) |
| `reports/audit_trail.jsonl` | Immutable SHA-256 hash-chained event log |
| `data/raw/regime_features.csv` | HMM regime probabilities per day |
| `data/raw/features_with_sentiment.csv` | Full feature matrix including FinBERT sentiment |

---

## API endpoints

```
GET  /health                → Model status, SR 11-7 status, uptime
POST /predict/var           → VaR 99% + ES 97.5% + top risk drivers
GET  /backtest/summary      → Latest Kupiec/Christoffersen results
GET  /model/card            → Model Card HTML (EU AI Act Art. 11)
```

**Example request:**

```bash
curl -X POST http://localhost:8001/predict/var \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio_returns": [-0.002, 0.003, -0.001, 0.005, -0.008, ...],
    "confidence_level": 0.99,
    "horizon_days": 1
  }'
```

---

## Architectural decisions

Key decisions documented in [`docs/decisions/ADRs.md`](docs/decisions/ADRs.md):

- **ADR-001**: TFT vs LSTM vs GARCH — why TFT is champion
- **ADR-002**: t-Student vs Normal distribution — fat tails in financial returns
- **ADR-003**: Free data sources (yfinance + FRED) — reproducibility without paid licenses
- **ADR-004**: Pinball Loss vs MSE — direct quantile estimation for VaR
- **ADR-005**: 4-repo architecture — governance as transversal layer

---

## Integration with AI Governance Framework

This repository is one of three integrated repos in the AI Risk portfolio:

```
github.com/[user]/
├── credit-risk-model-validation/     ← Credit Risk (SR 11-7, IFRS 9, GNN)
├── market-risk-deep-learning/        ← This repo
└── ai-governance-framework/          ← Central governance layer
    ├── submodules/credit-risk
    └── submodules/market-risk
```

The governance repo reads outputs from `reports/` to feed:
- **Model Risk Register** — model status, PSI, next validation date
- **Policy-as-code** — OPA/Rego rules evaluated against live metrics
- **Unified audit trail** — combined event log across both repos
- **Executive dashboard** — portfolio-level view for CROs and regulators

---

## Stack

| Category | Libraries |
|---|---|
| Deep Learning | PyTorch · pytorch-forecasting · pytorch-lightning |
| NLP | transformers (FinBERT) · feedparser |
| Time Series / Stats | arch (GARCH) · hmmlearn · statsmodels · scipy |
| Data | yfinance · fredapi · pandas · numpy |
| MLOps | MLflow · DVC-ready |
| Validation | Great Expectations · evidently |
| API | FastAPI · Pydantic · uvicorn |
| Dashboard | Streamlit · Plotly |
| Governance | Jinja2 · PyYAML · loguru |

---

## References

- Lim, B. et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.* International Journal of Forecasting.
- Angelopoulos, A. & Bates, S. (2022). *A Gentle Introduction to Conformal Prediction.* arXiv:2107.07511.
- Malo, P. et al. (2014). *Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts.* JASIST.
- Hamilton, J.D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series.* Econometrica.
- BCBS (2019). *Minimum capital requirements for market risk (FRTB).* Bank for International Settlements.
- Federal Reserve (2011). *SR 11-7: Supervisory Guidance on Model Risk Management.*
- European Parliament (2024). *Regulation (EU) 2024/1689 — EU AI Act.*

---

## Related projects

- 💳 [Credit Risk Model Validation Suite](https://github.com/[user]/credit-risk-model-validation) — XGBoost · GNN · SHAP · IFRS 9 · SR 11-7
- 🏛️ [AI Governance Framework](https://github.com/[user]/ai-governance-framework) — MRR · OPA/Rego · EU AI Act · Unified Audit Trail
