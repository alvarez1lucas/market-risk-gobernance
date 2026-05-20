# Notebooks — Orden de ejecución

Correr `python run_all.py` primero. Luego abrir en orden:

| Notebook | Descripción | Prerrequisito |
|---|---|---|
| 01_eda_market_data.ipynb | EDA de series financieras, distribuciones, regímenes | Stage 1-2 |
| 02_feature_engineering.ipynb | Log-returns, vol realizada, correlaciones, PCA | Stage 1-2 |
| 03_tft_training.ipynb | Entrenamiento TFT, learning curves, attention weights | Stage 3 |
| 04_lstm_attention.ipynb | LSTM vs TFT ablation study, attention visualization | Stage 3 |
| 05_var_backtesting.ipynb | Backtesting regulatorio, traffic light, análisis de errores | Stage 4 |
| 06_stress_testing.ipynb | Escenarios históricos, Monte Carlo, distribución de pérdidas | Stage 6 |
| 07_governance_validation.ipynb | SR 11-7 completo, Model Card, EU AI Act checklist | Stage 7-8 |

Celdas marcadas ✏️ requieren análisis propio del usuario.
