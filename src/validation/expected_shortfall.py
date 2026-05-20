"""
src/validation/expected_shortfall.py — Expected Shortfall bajo Basel III FRTB
El ES al 97.5% reemplaza al VaR 99% como métrica principal bajo FRTB.

Referencia: BCBS "Minimum capital requirements for market risk" (2019)
"""

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger
import json
from pathlib import Path
from typing import Dict


class ExpectedShortfallCalculator:
    """
    Calcula Expected Shortfall (CVaR) bajo el marco FRTB de Basel III.
    
    El ES es más conservador que el VaR porque considera la magnitud de las
    pérdidas en la cola, no solo si se superó el umbral.
    
    ES_α = E[Loss | Loss > VaR_α]
    
    Basel III FRTB usa ES al 97.5% (en lugar del VaR 99% de Basel II).
    Horizonte de liquidez varía por clase de activo (10 días para equities).
    """

    # Horizontes de liquidez por clase de activo (Basel III FRTB Tabla 2)
    LIQUIDITY_HORIZONS = {
        "equity_large_cap": 10,
        "equity_small_cap": 20,
        "fx_liquid": 10,
        "credit_ig": 20,
        "credit_hy": 40,
        "rates_sovereign": 10,
    }

    def __init__(self, var_predictions: np.ndarray,
                 realized_returns: np.ndarray = None,
                 confidence_level: float = 0.975):
        self.var_predictions = np.array(var_predictions)
        self.realized_returns = realized_returns
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level  # 0.025 para ES 97.5%

    def run(self) -> Dict:
        """Calcula ES y métricas regulatorias asociadas."""

        results = {}

        # ES histórico sobre las predicciones del modelo
        es_model = self._calculate_es_from_distribution(self.var_predictions)
        results["es_975"] = float(es_model)

        # ES condicional (si hay retornos realizados)
        if self.realized_returns is not None:
            es_realized = self._calculate_conditional_es(
                self.var_predictions, self.realized_returns
            )
            results["es_conditional"] = float(es_realized)
            results["es_ratio"] = float(es_model / es_realized) if es_realized != 0 else None

        # ES escalado a distintos horizontes de liquidez
        results["es_by_horizon"] = self._scale_to_horizons(es_model)

        # Capital requirement estimado bajo FRTB
        results["capital_requirement"] = self._estimate_capital_requirement(es_model)

        logger.info(f"  ES 97.5% (1-day): {es_model:.4f} | "
                    f"10-day: {results['es_by_horizon']['10d']:.4f}")

        # Guardar
        Path("reports").mkdir(exist_ok=True)
        with open("reports/expected_shortfall.json", "w") as f:
            json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v)
                       for k, v in results.items()}, f, indent=2)

        return results

    def _calculate_es_from_distribution(self, distribution: np.ndarray) -> float:
        """ES como media de la distribución por debajo del cuantil alpha."""
        threshold = np.percentile(distribution, self.alpha * 100)
        tail = distribution[distribution <= threshold]
        return float(tail.mean()) if len(tail) > 0 else float(threshold)

    def _calculate_conditional_es(self, var_pred: np.ndarray,
                                  realized: np.ndarray) -> float:
        """
        ES condicional: media de pérdidas realizadas que superaron el VaR predicho.
        Mide qué tan pesada es la cola cuando el modelo falla.
        """
        exceedance_mask = realized < var_pred
        if exceedance_mask.sum() == 0:
            return 0.0
        return float(realized[exceedance_mask].mean())

    def _scale_to_horizons(self, es_1day: float) -> Dict:
        """
        Escala el ES de 1 día a múltiples horizontes de liquidez.
        
        Basel III usa la raíz cuadrada del tiempo para escalar volatilidad,
        pero con ajustes por la estructura de correlación intertemporal.
        Simplificación: ES_T = ES_1 × sqrt(T) bajo supuesto de i.i.d.
        """
        horizons = [1, 5, 10, 20, 40, 60, 250]
        return {
            f"{h}d": float(es_1day * np.sqrt(h))
            for h in horizons
        }

    def _estimate_capital_requirement(self, es_975: float) -> Dict:
        """
        Estimación del capital regulatorio bajo FRTB Internal Models Approach.
        
        FRTB IMA Capital = max(ES_t, m_c × ES_avg_60d)
        donde m_c ≥ 1.5 (multiplicador supervisory) + backtesting add-on.
        
        Nota: Esta es una aproximación pedagógica. El cálculo real requiere
        datos de sensibilidades y correlaciones por risk bucket.
        """
        m_c = 1.5  # Multiplicador mínimo Basel III

        # ES a 10 días (horizonte estándar para equities)
        es_10d = es_975 * np.sqrt(10)

        # Capital de mercado estimado
        capital_estimate = m_c * abs(es_10d)

        return {
            "es_10d": float(es_10d),
            "multiplier_mc": m_c,
            "capital_charge_estimate": float(capital_estimate),
            "note": "Aproximación pedagógica — cálculo real requiere risk bucketing FRTB",
        }
