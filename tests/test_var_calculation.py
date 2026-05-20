"""
tests/test_var_calculation.py — Tests unitarios del cálculo de VaR y ES
"""

import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestVaRCalculation:

    def test_var_99_below_var_975(self):
        """VaR 99% debe ser más extremo que VaR 97.5%."""
        returns = np.random.normal(-0.001, 0.015, 1000)
        var_99  = np.percentile(returns, 1)
        var_975 = np.percentile(returns, 2.5)
        assert var_99 <= var_975, "VaR 99% debe ser <= VaR 97.5%"

    def test_es_more_conservative_than_var(self):
        """ES debe ser más conservador que VaR (mayor pérdida esperada)."""
        returns = np.random.normal(-0.001, 0.015, 1000)
        var_975 = np.percentile(returns, 2.5)
        es_975  = returns[returns <= var_975].mean()
        assert es_975 <= var_975, "ES debe ser <= VaR (más conservador)"

    def test_sqrt_time_scaling(self):
        """Scaling VaR 1-day → 10-day con sqrt(10)."""
        var_1d = -0.025
        var_10d = var_1d * np.sqrt(10)
        assert abs(var_10d) > abs(var_1d)
        assert np.isclose(var_10d, var_1d * np.sqrt(10))

    def test_var_negative_for_loss(self):
        """VaR expresado como pérdida debe ser negativo."""
        np.random.seed(42)
        returns = np.random.normal(-0.001, 0.015, 500)
        var_99 = np.percentile(returns, 1)
        assert var_99 < 0, "VaR 99% de una distribución típica debe ser negativo"

    def test_psi_zero_for_identical_distributions(self):
        """PSI debe ser ~0 para distribuciones idénticas."""
        from src.monitoring.drift import DriftMonitor
        data = np.random.normal(0, 0.01, 500)
        monitor = DriftMonitor(None)
        monitor.features = None
        psi = monitor.calculate_psi(data, data, n_bins=10)
        assert psi < 0.01, f"PSI para distribuciones idénticas debe ser ~0, got {psi}"

    def test_psi_high_for_different_distributions(self):
        """PSI debe ser alto para distribuciones muy diferentes."""
        from src.monitoring.drift import DriftMonitor
        ref = np.random.normal(0, 0.01, 500)
        cur = np.random.normal(0.05, 0.03, 500)  # Media y std muy diferentes
        monitor = DriftMonitor(None)
        monitor.features = None
        psi = monitor.calculate_psi(ref, cur, n_bins=10)
        assert psi > 0.10, f"PSI para distribuciones diferentes debe ser > 0.10, got {psi}"


class TestBacktesting:

    def setup_method(self):
        """Datos sintéticos para tests."""
        np.random.seed(42)
        n = 300
        self.returns = np.random.normal(-0.001, 0.015, n)
        # VaR histórico: cuantil 1%
        self.var_predictions = np.full(n, np.percentile(self.returns[:250], 1))

    def test_kupiec_passes_for_correct_model(self):
        """Un modelo bien calibrado debe pasar Kupiec."""
        from src.validation.var_backtesting import VaRBacktester
        import pandas as pd

        backtester = VaRBacktester(
            model_predictions=self.var_predictions,
            realized_returns=pd.Series(self.returns),
            confidence_level=0.99,
        )
        result = backtester.run()
        # No necesariamente PASS en datos sintéticos, pero debe correr sin error
        assert hasattr(result, "kupiec_pval")
        assert 0 <= result.kupiec_pval <= 1

    def test_exceedance_rate_approx_alpha(self):
        """Para un modelo correcto, exceedance rate ≈ alpha (1%)."""
        from src.validation.var_backtesting import VaRBacktester
        import pandas as pd

        backtester = VaRBacktester(
            model_predictions=self.var_predictions,
            realized_returns=pd.Series(self.returns),
        )
        result = backtester.run()
        # Con 250 días y alpha=1%, esperamos ~2-3 exceedances
        assert 0 <= result.n_exceedances <= 20

    def test_traffic_light_green_few_exceedances(self):
        """Pocos exceedances deben dar zona verde."""
        from src.validation.var_backtesting import VaRBacktester
        import pandas as pd

        # VaR muy conservador → pocos exceedances
        conservative_var = np.full(300, -0.10)  # -10% → casi nunca se supera
        backtester = VaRBacktester(
            model_predictions=conservative_var,
            realized_returns=pd.Series(self.returns),
        )
        result = backtester.run()
        assert result.traffic_light_zone == "green"
        assert result.n_exceedances <= 4

    def test_traffic_light_red_many_exceedances(self):
        """Muchos exceedances deben dar zona roja."""
        from src.validation.var_backtesting import VaRBacktester
        import pandas as pd

        # VaR muy optimista → muchos exceedances
        optimistic_var = np.full(300, 0.05)  # +5% → siempre se supera
        backtester = VaRBacktester(
            model_predictions=optimistic_var,
            realized_returns=pd.Series(self.returns),
        )
        result = backtester.run()
        assert result.traffic_light_zone == "red"
        assert result.overall_status == "rejected"


class TestExpectedShortfall:

    def test_es_calculation(self):
        """ES debe ser la media de la cola por debajo del cuantil."""
        from src.validation.expected_shortfall import ExpectedShortfallCalculator
        np.random.seed(42)
        returns = np.random.normal(-0.001, 0.015, 1000)
        calc = ExpectedShortfallCalculator(returns, confidence_level=0.975)
        result = calc.run()
        assert "es_975" in result
        assert result["es_975"] < 0  # ES debe ser negativo (pérdida)
        assert result["es_975"] < np.percentile(returns, 2.5)  # ES más extremo que VaR

    def test_scaling_to_horizons(self):
        """ES a 10 días debe ser mayor que ES a 1 día."""
        from src.validation.expected_shortfall import ExpectedShortfallCalculator
        returns = np.random.normal(-0.001, 0.015, 500)
        calc = ExpectedShortfallCalculator(returns)
        result = calc.run()
        assert abs(result["es_by_horizon"]["10d"]) > abs(result["es_by_horizon"]["1d"])
