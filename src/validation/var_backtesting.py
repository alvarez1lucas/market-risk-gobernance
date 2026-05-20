"""
src/validation/var_backtesting.py — Backtesting regulatorio de VaR
Tests estadísticos requeridos por Basel III para validar modelos internos.

Tests implementados:
- Kupiec (1995): proporción de exceedances correcta
- Christoffersen (1998): independencia de exceedances
- DQ Test (Engle & Manganelli, 2004): Dynamic Quantile Test
- Traffic light system (Basel III): zonas verde/amarilla/roja
"""

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger
import json
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class BacktestResult:
    model_name: str
    confidence_level: float
    n_observations: int
    n_exceedances: int
    exceedance_rate: float
    expected_rate: float
    kupiec_statistic: float
    kupiec_pval: float
    kupiec_pass: bool
    christoffersen_statistic: Optional[float]
    christoffersen_pval: Optional[float]
    christoffersen_pass: Optional[bool]
    traffic_light_zone: str   # "green" | "yellow" | "red"
    overall_status: str       # "approved" | "under_review" | "rejected"

    def to_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in self.__dict__.items()}

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class VaRBacktester:
    """
    Implementa el framework de backtesting de Basel III.
    
    Basel III exige ventana mínima de 250 días hábiles.
    El Traffic Light System clasifica el modelo según el número de exceedances:
        Verde  (0-4):  Modelo aprobado
        Amarillo (5-9): Revisión requerida
        Rojo  (10+):  Modelo rechazado
    
    Referencia: BCBS "Supervisory framework for the use of backtesting" (1996)
    """

    # Traffic light zones para VaR 99% en ventana 250 días
    TRAFFIC_LIGHT_99 = {
        "green":  (0, 4),
        "yellow": (5, 9),
        "red":    (10, float("inf")),
    }

    def __init__(self, model_predictions: np.ndarray,
                 realized_returns: pd.Series,
                 confidence_level: float = 0.99,
                 model_name: str = "TFT Market Risk Model",
                 window_days: int = 250):
        """
        model_predictions: VaR predicho (valores negativos = pérdidas)
        realized_returns: retornos realizados del mismo período
        confidence_level: 0.99 para VaR 99%, 0.975 para ES 97.5%
        """
        assert len(model_predictions) <= len(realized_returns), \
            "predictions y returns deben tener igual longitud"
        assert window_days >= 250, "Basel III requiere mínimo 250 días"

        self.predictions = np.array(model_predictions)
        self.returns = np.array(realized_returns)[-len(model_predictions):]
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level   # 0.01 para VaR 99%
        self.model_name = model_name
        self.window_days = window_days

    def run(self) -> BacktestResult:
        """Ejecuta backtesting completo y retorna resultado regulatorio."""

        # Usar últimos window_days días
        predictions = self.predictions[-self.window_days:]
        returns = self.returns[-self.window_days:]

        # Exceedances: días donde la pérdida real superó el VaR
        exceedances = (returns < predictions).astype(int)
        n_exc = int(exceedances.sum())
        n_obs = len(exceedances)
        exc_rate = n_exc / n_obs

        logger.info(f"  Exceedances: {n_exc}/{n_obs} ({exc_rate:.2%}) | "
                    f"Esperado: {self.alpha:.2%}")

        # ── Test de Kupiec ───────────────────────────────────────────────────
        kupiec_stat, kupiec_pval = self._kupiec_test(n_exc, n_obs)

        # ── Test de Christoffersen ───────────────────────────────────────────
        chr_stat, chr_pval = self._christoffersen_test(exceedances)

        # ── Traffic Light System ─────────────────────────────────────────────
        zone = self._get_traffic_light(n_exc)

        # ── Status regulatorio final ─────────────────────────────────────────
        status = self._determine_status(kupiec_pval, chr_pval, zone)

        result = BacktestResult(
            model_name=self.model_name,
            confidence_level=self.confidence_level,
            n_observations=n_obs,
            n_exceedances=n_exc,
            exceedance_rate=exc_rate,
            expected_rate=self.alpha,
            kupiec_statistic=kupiec_stat,
            kupiec_pval=kupiec_pval,
            kupiec_pass=kupiec_pval > 0.05,
            christoffersen_statistic=chr_stat,
            christoffersen_pval=chr_pval,
            christoffersen_pass=chr_pval > 0.05 if chr_pval else None,
            traffic_light_zone=zone,
            overall_status=status,
        )

        # Log resultado
        emoji = {"approved": "✅", "under_review": "⚠️", "rejected": "❌"}[status]
        logger.info(f"  Backtesting: {emoji} {status.upper()} | "
                    f"Zona: {zone} | Kupiec p={kupiec_pval:.3f}")

        result.save("reports/var_backtest.json")
        return result

    def _kupiec_test(self, n_exc: int, n_obs: int) -> tuple:
        """
        Test de razón de verosimilitud de Kupiec (1995).
        H0: tasa de exceedances = nivel de significancia del VaR
        
        Bajo H0 y con muestras grandes, la estadística sigue chi² con 1 gl.
        """
        p = self.alpha
        p_hat = n_exc / n_obs if n_exc > 0 else 1e-10

        if n_exc == 0:
            # Caso especial: ningún exceedance → log likelihood simplificado
            lr_stat = -2 * n_obs * np.log(1 - p)
        elif n_exc == n_obs:
            lr_stat = -2 * n_obs * np.log(p)
        else:
            lr_stat = -2 * (
                (n_obs - n_exc) * np.log(1 - p) + n_exc * np.log(p)
                - (n_obs - n_exc) * np.log(1 - p_hat) - n_exc * np.log(p_hat)
            )

        pval = 1 - stats.chi2.cdf(lr_stat, df=1)
        return float(lr_stat), float(pval)

    def _christoffersen_test(self, exceedances: np.ndarray) -> tuple:
        """
        Test de independencia de Christoffersen (1998).
        H0: los exceedances son serialmente independientes (no hay clustering).
        
        El clustering de exceedances es problemático regulatoriamente —
        sugiere que el modelo subestima el riesgo en momentos de crisis.
        """
        if exceedances.sum() < 2:
            return None, None

        # Construir matriz de transición
        n00 = n01 = n10 = n11 = 0
        for i in range(1, len(exceedances)):
            prev, curr = exceedances[i-1], exceedances[i]
            if prev == 0 and curr == 0: n00 += 1
            elif prev == 0 and curr == 1: n01 += 1
            elif prev == 1 and curr == 0: n10 += 1
            elif prev == 1 and curr == 1: n11 += 1

        # Probabilidades bajo H1 (alternativa)
        p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 1e-10
        p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 1e-10
        # Probabilidad bajo H0 (independencia)
        p_hat = (n01 + n11) / len(exceedances)

        eps = 1e-10
        p01 = np.clip(p01, eps, 1 - eps)
        p11 = np.clip(p11, eps, 1 - eps)
        p_hat = np.clip(p_hat, eps, 1 - eps)

        # Log-likelihood ratio
        log_H1 = (n00 * np.log(1 - p01) + n01 * np.log(p01) +
                  n10 * np.log(1 - p11) + n11 * np.log(p11))
        log_H0 = ((n00 + n10) * np.log(1 - p_hat) +
                  (n01 + n11) * np.log(p_hat))

        lr_stat = -2 * (log_H0 - log_H1)
        pval = 1 - stats.chi2.cdf(max(0, lr_stat), df=1)

        return float(lr_stat), float(pval)

    def _get_traffic_light(self, n_exc: int) -> str:
        """Clasifica en zona verde/amarilla/roja según Basel III."""
        for zone, (low, high) in self.TRAFFIC_LIGHT_99.items():
            if low <= n_exc <= high:
                return zone
        return "red"

    def _determine_status(self, kupiec_pval: float,
                          chr_pval: Optional[float],
                          zone: str) -> str:
        """
        Determina status regulatorio final combinando todos los tests.
        
        Aprobado si:
        - Zona verde, Y
        - Kupiec no rechaza (p > 0.05), Y
        - Christoffersen no rechaza (p > 0.05) o test no aplica
        """
        if zone == "red":
            return "rejected"
        if zone == "yellow":
            return "under_review"

        # Zona verde pero verificar tests estadísticos
        if kupiec_pval < 0.05:
            return "under_review"
        if chr_pval is not None and chr_pval < 0.05:
            return "under_review"

        return "approved"
