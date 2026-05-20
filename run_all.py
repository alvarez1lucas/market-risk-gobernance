"""
run_all.py — Pipeline principal Market Risk Deep Learning Suite
Orquesta todos los stages en orden correcto.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

import mlflow

# Fix Windows console encoding issues for emoji/unicode logging messages
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure report directory exists before the FileHandler opens the log file
Path("reports").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"reports/pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"),
    ],
)
log = logging.getLogger(__name__)


def run_pipeline():
    log.info("=" * 60)
    log.info("Market Risk DL Suite — Pipeline start")
    log.info("=" * 60)

    mlflow.set_experiment("market-risk-dl")

    with mlflow.start_run(run_name=f"pipeline_{datetime.now():%Y%m%d_%H%M}"):

        # ── Stage 1: Ingesta de datos ──────────────────────────────
        log.info("[Stage 1] Descargando datos — yfinance + FRED")
        from src.data.ingest import MarketDataIngestor
        ingestor = MarketDataIngestor(start="2000-01-01", end="2024-12-31")
        raw_data = ingestor.run()
        log.info(f"  Activos descargados: {list(raw_data.keys())}")

        # ── Stage 2: Feature engineering ──────────────────────────
        log.info("[Stage 2] Feature engineering — log-returns, vol realizada")
        from src.data.features import FeatureEngine
        features = FeatureEngine(raw_data).run()
        mlflow.log_metric("n_features", features.shape[1])
        log.info(f"  Features generadas: {features.shape[1]} columnas, {features.shape[0]} observaciones")

        # ── Stage 3: Entrenamiento de modelos ─────────────────────
        log.info("[Stage 3a] Entrenando Temporal Fusion Transformer")
        from src.models.tft_model import TFTRiskModel
        tft = TFTRiskModel(features)
        tft_results = tft.train()
        mlflow.log_metrics({"tft_val_loss": tft_results["val_loss"], "tft_mae": tft_results["mae"]})

        log.info("[Stage 3b] Entrenando LSTM con atención")
        from src.models.lstm_attention import LSTMAttentionModel
        lstm = LSTMAttentionModel(features)
        lstm_results = lstm.train()
        mlflow.log_metrics({"lstm_val_loss": lstm_results["val_loss"], "lstm_mae": lstm_results["mae"]})

        log.info("[Stage 3c] Calibrando benchmark GARCH(1,1)")
        from src.models.garch_benchmark import GARCHBenchmark
        garch = GARCHBenchmark(features)
        garch_results = garch.fit()

        # ── Stage 4: Backtesting regulatorio ──────────────────────
        log.info("[Stage 4] Backtesting VaR — Kupiec + Christoffersen")
        from src.validation.var_backtesting import VaRBacktester
        backtester = VaRBacktester(
            model_predictions=tft_results["predictions"],
            realized_returns=features["log_return_SPX"],
        )
        backtest_report = backtester.run()
        mlflow.log_metrics({
            "kupiec_pval": backtest_report.kupiec_pval,
            "christoffersen_pval": backtest_report.christoffersen_pval or 0.0,
            "exceedances_250d": backtest_report.n_exceedances,
        })
        christoffersen_text = (
            f"{backtest_report.christoffersen_pval:.3f}"
            if backtest_report.christoffersen_pval is not None
            else "N/A"
        )
        log.info(f"  Kupiec p={backtest_report.kupiec_pval:.3f} | "
                 f"Christoffersen p={christoffersen_text} | "
                 f"Exceedances: {backtest_report.n_exceedances}")

        # ── Stage 5: Expected Shortfall ───────────────────────────
        log.info("[Stage 5] Expected Shortfall 97.5% — Basel III FRTB")
        from src.validation.expected_shortfall import ExpectedShortfallCalculator
        es_calc = ExpectedShortfallCalculator(tft_results["predictions"])
        es_report = es_calc.run()
        mlflow.log_metric("expected_shortfall_975", es_report["es_975"])

        # ── Stage 6: Stress testing ───────────────────────────────
        log.info("[Stage 6] Stress testing — COVID, GFC 2008, +200bps")
        from src.data.stress_generator import StressScenarioGenerator
        stress = StressScenarioGenerator(features)
        stress_report = stress.run_all_scenarios()

        # ── Stage 7: Validación SR 11-7 ───────────────────────────
        log.info("[Stage 7] Validación SR 11-7 completa")
        from src.validation.sr117 import SR117Validator
        validator = SR117Validator(
            model=tft,
            backtest_report=backtest_report,
            stress_report=stress_report,
        )
        sr117_report = validator.validate()
        sr117_report.save("reports/sr117_validation.json")

        # ── Stage 8: Governance ───────────────────────────────────
        log.info("[Stage 8] Generando Model Card + Audit Trail")
        from src.governance.model_card import ModelCardGenerator
        card = ModelCardGenerator(
            model_name="Market VaR — TFT v1.0",
            backtest=backtest_report,
            stress=stress_report,
            sr117=sr117_report,
        )
        card.generate("reports/model_card.html")

        from src.governance.audit_trail import AuditTrail
        audit = AuditTrail()
        audit.log_event("pipeline_completed", {
            "kupiec_pval": backtest_report.kupiec_pval,
            "es_975": es_report["es_975"],
            "sr117_status": sr117_report.overall_status,
        })

        # ── Stage 9: Drift baseline ───────────────────────────────
        log.info("[Stage 9] Estableciendo baseline de drift monitoring")
        from src.monitoring.drift import DriftMonitor
        drift_monitor = DriftMonitor(features)
        drift_monitor.save_baseline("models/champion/drift_baseline.json")

        # ── Selección de champion ─────────────────────────────────
        log.info("[Champion selection] Comparando TFT vs LSTM vs GARCH")
        champion = _select_champion(tft_results, lstm_results, garch_results, backtest_report)
        log.info(f"  Champion: {champion}")
        mlflow.log_param("champion_model", champion)

    log.info("=" * 60)
    log.info("Pipeline completado exitosamente")
    log.info("  Dashboard: streamlit run src/dashboard/app.py")
    log.info("  API:       uvicorn src.api.main:app --reload --port 8001")
    log.info("=" * 60)


def _select_champion(tft, lstm, garch, backtest) -> str:
    """
    Selecciona el modelo champion basado en:
    1. Aprobación de backtesting regulatorio (Kupiec p > 0.05)
    2. Menor MAE en validación
    3. Menor ES (más conservador en stress)
    """
    if backtest.kupiec_pval < 0.05:
        raise ValueError("Ningún modelo aprueba backtesting regulatorio — pipeline detenido")

    scores = {
        "TFT": tft["mae"],
        "LSTM_Attention": lstm["mae"],
        "GARCH": garch["mae"],
    }
    return min(scores, key=scores.get)


if __name__ == "__main__":
    # Crear dirs necesarios si no existen
    for d in ["reports/figures", "reports/stress_scenarios", "models/champion", "models/challenger"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    run_pipeline()
