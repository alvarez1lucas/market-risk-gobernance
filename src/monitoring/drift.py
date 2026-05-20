"""
src/monitoring/drift.py — Drift monitoring para modelos de market risk
Detecta cambios de régimen en la distribución de retornos.
Métricas: PSI, KS test, correlación rolling.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy import stats
from loguru import logger
from typing import Dict, Optional


class DriftMonitor:
    """
    Monitor de drift para series temporales financieras.
    
    A diferencia del drift en ML clásico, en mercados el drift puede ser:
    - Cambio de régimen (crisis → recuperación)
    - Structural break (cambio en política monetaria)
    - Data quality issue (ticker suspendido, corporate action)
    
    Se monitorea PSI sobre distribución de retornos y correlaciones.
    """

    PSI_WARNING  = 0.10   # Requiere investigación
    PSI_CRITICAL = 0.20   # Requiere re-validación del modelo

    def __init__(self, features: pd.DataFrame, window_days: int = 63):
        self.features = features
        self.window = window_days
        self.baseline: Optional[Dict] = None

    def save_baseline(self, path: str = "models/champion/drift_baseline.json"):
        """Guarda la distribución de referencia del período de entrenamiento."""
        return_cols = [c for c in self.features.columns if c.startswith("log_return_")]
        baseline = {}

        for col in return_cols:
            series = self.features[col].dropna()
            baseline[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "skew": float(series.skew()),
                "kurt": float(series.kurtosis()),
                "percentiles": {
                    str(p): float(np.percentile(series, p))
                    for p in [1, 5, 25, 50, 75, 95, 99]
                },
            }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "baseline_period_start": str(self.features.index[0].date()),
                "baseline_period_end": str(self.features.index[-1].date()),
                "n_observations": len(self.features),
                "series": baseline,
            }, f, indent=2)

        self.baseline = baseline
        logger.info(f"Baseline de drift guardado: {len(return_cols)} series")

    def calculate_psi(self, reference: np.ndarray, current: np.ndarray,
                      n_bins: int = 10) -> float:
        """
        Population Stability Index (PSI).
        Mide cuánto cambió la distribución de retornos.
        PSI < 0.10: sin cambio significativo
        PSI 0.10-0.20: cambio moderado — monitorear
        PSI > 0.20: cambio severo — re-validar modelo
        """
        # Bins basados en la distribución de referencia
        bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bins[0]  = -np.inf
        bins[-1] =  np.inf

        ref_counts = np.histogram(reference, bins=bins)[0]
        cur_counts = np.histogram(current,   bins=bins)[0]

        # Proporciones con suavizado para evitar log(0)
        ref_pct = (ref_counts + 0.5) / (len(reference) + 0.5 * n_bins)
        cur_pct = (cur_counts + 0.5) / (len(current)   + 0.5 * n_bins)

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def run_daily_check(self, new_data: pd.Series, series_name: str) -> Dict:
        """
        Chequeo diario de drift sobre una serie de retornos.
        Compara la ventana reciente contra el baseline.
        """
        if self.baseline is None:
            logger.warning("Baseline no cargado — correr save_baseline() primero")
            return {}

        baseline_series = self.baseline.get(series_name, {})
        if not baseline_series:
            return {}

        # Reconstruir distribución de referencia aproximada desde percentiles
        pcts = baseline_series["percentiles"]
        ref_approx = np.array([float(v) for v in pcts.values()])

        # Serie actual (últimos window_days)
        current = new_data.dropna().values[-self.window:]

        psi = self.calculate_psi(ref_approx, current)
        ks_stat, ks_pval = stats.ks_2samp(ref_approx, current)

        # Determinar severidad
        if psi > self.PSI_CRITICAL:
            severity = "CRITICAL"
            action   = "Re-validación inmediata requerida"
        elif psi > self.PSI_WARNING:
            severity = "WARNING"
            action   = "Monitoreo reforzado — investigar causa"
        else:
            severity = "OK"
            action   = "Sin acción requerida"

        result = {
            "series": series_name,
            "psi": psi,
            "severity": severity,
            "action": action,
            "ks_statistic": float(ks_stat),
            "ks_pval": float(ks_pval),
            "current_mean": float(np.mean(current)),
            "baseline_mean": baseline_series["mean"],
            "current_std": float(np.std(current)),
            "baseline_std": baseline_series["std"],
        }

        log_fn = logger.warning if severity != "OK" else logger.info
        log_fn(f"Drift [{series_name}]: PSI={psi:.3f} → {severity}")

        return result
