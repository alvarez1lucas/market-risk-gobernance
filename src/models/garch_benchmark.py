"""
src/models/garch_benchmark.py — GARCH(1,1) como benchmark regulatorio
Modelo clásico exigido por Basel III como punto de comparación.
El regulador exige demostrar que el modelo DL supera al benchmark histórico.
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning("arch no instalado — GARCH usará implementación manual")


class GARCHBenchmark:
    """
    GARCH(1,1) con distribución t-Student.
    Basel III requiere que los modelos internos sean comparados
    contra este benchmark para justificar el uso de modelos avanzados.
    """

    def __init__(self, features: pd.DataFrame, target_col: str = "log_return_SPX"):
        self.returns = features[target_col].dropna()
        self.model = None
        self.fitted = None

    def fit(self) -> Dict:
        """Ajusta el GARCH y retorna pronósticos de volatilidad."""
        if ARCH_AVAILABLE:
            return self._fit_arch()
        else:
            return self._fit_manual()

    def _fit_arch(self) -> Dict:
        """GARCH(1,1) con distribución t-Student via arch library."""
        train_size = int(len(self.returns) * 0.85)
        train = self.returns.iloc[:train_size]

        self.model = arch_model(
            train * 100,  # arch espera retornos en porcentaje
            vol="Garch",
            p=1, q=1,
            dist="t",     # t-Student — colas más pesadas que normal
        )
        self.fitted = self.model.fit(disp="off", show_warning=False)

        # Forecasts en ventana de validación (expanding window)
        val_returns = self.returns.iloc[train_size:]
        var_99_forecasts = []
        es_975_forecasts = []

        for i in range(len(val_returns)):
            # Refit expandiendo la ventana
            window = self.returns.iloc[:train_size + i]
            model = arch_model(window * 100, vol="Garch", p=1, q=1, dist="t")
            res = model.fit(disp="off", show_warning=False, last_obs=len(window))
            forecast = res.forecast(horizon=1)
            sigma = np.sqrt(forecast.variance.values[-1, 0]) / 100

            # VaR bajo distribución t
            from scipy.stats import t as t_dist
            df = res.params.get("nu", 5)  # Grados de libertad estimados
            var_99 = -t_dist.ppf(0.01, df=df) * sigma
            es_975 = (t_dist.pdf(t_dist.ppf(0.025, df=df), df=df) / 0.025) * sigma * (df / (df - 1))

            var_99_forecasts.append(-var_99)  # Negativo porque VaR es pérdida
            es_975_forecasts.append(-es_975)

        mae = float(np.mean(np.abs(np.array(var_99_forecasts) - val_returns.values)))

        return {
            "val_loss": float(self.fitted.aic),
            "mae": mae,
            "predictions": np.array(var_99_forecasts),
            "es_975": np.array(es_975_forecasts),
            "params": {
                "omega": float(self.fitted.params["omega"]),
                "alpha[1]": float(self.fitted.params.get("alpha[1]", 0)),
                "beta[1]": float(self.fitted.params.get("beta[1]", 0)),
                "nu": float(self.fitted.params.get("nu", 5)),  # Grados de libertad t
            },
            "persistence": float(
                self.fitted.params.get("alpha[1]", 0) +
                self.fitted.params.get("beta[1]", 0)
            ),  # Suma cercana a 1 = alta persistencia de volatilidad
        }

    def _fit_manual(self) -> Dict:
        """Implementación manual de GARCH(1,1) sin arch library."""
        returns = self.returns.values
        n = len(returns)
        omega, alpha, beta = 0.00001, 0.09, 0.90

        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)

        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]

        train_size = int(n * 0.85)
        val_sigma = np.sqrt(sigma2[train_size:])

        from scipy.stats import norm
        var_99 = -norm.ppf(0.01) * val_sigma

        mae = float(np.mean(np.abs(-var_99 - returns[train_size:])))

        return {
            "val_loss": float(np.mean(sigma2[train_size:])),
            "mae": mae,
            "predictions": -var_99,
            "params": {"omega": omega, "alpha[1]": alpha, "beta[1]": beta},
            "persistence": alpha + beta,
        }
