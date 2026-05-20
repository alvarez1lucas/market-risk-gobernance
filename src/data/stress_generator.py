"""
src/data/stress_generator.py — Generador de escenarios de stress
Escenarios calibrados a DFAST, Basel III y eventos históricos reales.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class StressScenario:
    name: str
    description: str
    equity_shock: float      # % cambio en equities
    credit_spread_shock: float  # bps de widening en spreads
    rate_shock: float        # bps de cambio en tasas
    fx_shock: float          # % depreciación USD
    vol_multiplier: float    # multiplicador de volatilidad
    horizon_days: int        # horizonte del escenario


# Escenarios históricos calibrados a datos reales
STRESS_SCENARIOS = {
    "gfc_2008": StressScenario(
        name="Global Financial Crisis 2008",
        description="Colapso Lehman Brothers — peor drawdown desde 1929",
        equity_shock=-0.57,
        credit_spread_shock=1800,   # HY spreads de 300bps a >2100bps
        rate_shock=-300,            # Fed corta 300bps
        fx_shock=0.15,              # USD aprecia (flight to safety)
        vol_multiplier=4.2,         # VIX llegó a 89.5
        horizon_days=252,
    ),
    "covid_2020": StressScenario(
        name="COVID-19 Market Crash Q1 2020",
        description="Crash más rápido de la historia — -34% en 33 días",
        equity_shock=-0.34,
        credit_spread_shock=900,
        rate_shock=-150,
        fx_shock=0.08,
        vol_multiplier=3.1,         # VIX llegó a 82.7
        horizon_days=33,
    ),
    "rate_shock_2022": StressScenario(
        name="Fed Rate Hike Cycle 2022",
        description="Subida de tasas más agresiva desde 1980 — bonos -20%",
        equity_shock=-0.25,
        credit_spread_shock=400,
        rate_shock=525,             # +525bps en 18 meses
        fx_shock=-0.05,
        vol_multiplier=1.8,
        horizon_days=365,
    ),
    "svb_2023": StressScenario(
        name="SVB Bank Run 2023",
        description="Colapso SVB — crisis de confianza bancaria regional",
        equity_shock=-0.15,
        credit_spread_shock=250,
        rate_shock=-50,
        fx_shock=0.03,
        vol_multiplier=1.6,
        horizon_days=21,
    ),
    "dfast_adverse": StressScenario(
        name="DFAST Severely Adverse Scenario",
        description="Escenario regulatorio Fed DFAST — recesión severa",
        equity_shock=-0.55,
        credit_spread_shock=600,
        rate_shock=-400,
        fx_shock=0.10,
        vol_multiplier=3.5,
        horizon_days=336,  # 9 trimestres × ~37 días
    ),
    "latam_crisis": StressScenario(
        name="LATAM Tail Risk",
        description="Crisis cambiaria emergente con contagio — relevante BCRA",
        equity_shock=-0.40,
        credit_spread_shock=800,
        rate_shock=200,
        fx_shock=0.35,              # Depreciación severa monedas EM
        vol_multiplier=2.5,
        horizon_days=90,
    ),
}


class StressScenarioGenerator:
    """
    Genera retornos y pérdidas bajo escenarios de stress.
    Combina shocks históricos con simulación Monte Carlo.
    """

    def __init__(self, features: pd.DataFrame, n_monte_carlo: int = 10_000):
        self.features = features
        self.n_mc = n_monte_carlo
        self.output_dir = Path("reports/stress_scenarios")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_all_scenarios(self) -> Dict:
        """Ejecuta todos los escenarios y retorna reporte consolidado."""
        results = {}

        for scenario_id, scenario in STRESS_SCENARIOS.items():
            logger.info(f"  Escenario: {scenario.name}")
            result = self._run_single_scenario(scenario)
            results[scenario_id] = result

        # Agregar escenarios Monte Carlo
        logger.info("  Monte Carlo — tail scenarios (10,000 simulaciones)")
        results["monte_carlo"] = self._run_monte_carlo()

        # Guardar reporte consolidado
        self._save_report(results)
        return results

    def _run_single_scenario(self, scenario: StressScenario) -> dict:
        """
        Aplica shocks del escenario sobre el portfolio y calcula P&L.
        Asume portfolio igual ponderado entre activos.
        """
        return_cols = [c for c in self.features.columns if c.startswith("log_return_")]

        if not return_cols:
            logger.warning("No se encontraron columnas de retorno")
            return {}

        # Distribución histórica de retornos
        hist_returns = self.features[return_cols].dropna()

        # Aplicar shock de equity
        stressed_returns = hist_returns.copy()
        for col in return_cols:
            if "SPX" in col or "EEM" in col or "MERVAL" in col:
                stressed_returns[col] = hist_returns[col] + (
                    scenario.equity_shock / scenario.horizon_days
                )

        # Calcular P&L del portfolio (igual ponderado)
        portfolio_pnl = stressed_returns.mean(axis=1)  # igual ponderado

        # Métricas del escenario
        total_loss = portfolio_pnl.sum()
        max_daily_loss = portfolio_pnl.min()
        stressed_vol = portfolio_pnl.std() * np.sqrt(252) * scenario.vol_multiplier

        # VaR y ES bajo el escenario
        var_99 = np.percentile(portfolio_pnl, 1)
        es_975 = portfolio_pnl[portfolio_pnl <= np.percentile(portfolio_pnl, 2.5)].mean()

        return {
            "scenario": asdict(scenario),
            "total_loss_pct": float(total_loss),
            "max_daily_loss_pct": float(max_daily_loss),
            "stressed_annual_vol": float(stressed_vol),
            "var_99_1day": float(var_99),
            "es_975_1day": float(es_975) if not np.isnan(es_975) else None,
            "capital_requirement_multiplier": scenario.vol_multiplier,
        }

    def _run_monte_carlo(self) -> dict:
        """
        Simulación Monte Carlo para distribución de pérdidas en la cola.
        Usa distribución t-Student (colas más pesadas que normal — más realista).
        """
        return_cols = [c for c in self.features.columns if c.startswith("log_return_")]
        if not return_cols:
            return {}

        hist_returns = self.features[return_cols].dropna()
        mu = hist_returns.mean().values
        cov = hist_returns.cov().values

        # Simulación con distribución t (grados de libertad = 5 — colas pesadas)
        np.random.seed(42)
        df_t = 5  # degrees of freedom
        z = np.random.standard_t(df_t, size=(self.n_mc, len(return_cols)))

        # Escalar por la covarianza histórica (Cholesky decomposition)
        try:
            L = np.linalg.cholesky(cov)
            simulated = z @ L.T + mu
        except np.linalg.LinAlgError:
            # Fallback si la matriz no es definida positiva
            simulated = z * np.sqrt(np.diag(cov)) + mu

        portfolio_pnl = simulated.mean(axis=1)  # igual ponderado

        var_levels = [0.01, 0.025, 0.05]
        var_results = {
            f"var_{int(v*100)}": float(np.percentile(portfolio_pnl, v * 100))
            for v in var_levels
        }

        es_975 = float(
            portfolio_pnl[portfolio_pnl <= np.percentile(portfolio_pnl, 2.5)].mean()
        )

        return {
            "n_simulations": self.n_mc,
            "distribution": "t-Student (df=5)",
            "var_results": var_results,
            "es_975": es_975,
            "worst_1pct_avg": float(
                np.mean(np.sort(portfolio_pnl)[: int(self.n_mc * 0.01)])
            ),
        }

    def _save_report(self, results: dict):
        """Guarda reporte de stress en JSON y CSV para el pipeline."""
        # JSON completo
        output_path = self.output_dir / "stress_report.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Tabla resumen CSV
        summary_rows = []
        for scenario_id, result in results.items():
            if scenario_id == "monte_carlo":
                continue
            summary_rows.append({
                "scenario": scenario_id,
                "name": result.get("scenario", {}).get("name", ""),
                "total_loss_pct": result.get("total_loss_pct"),
                "var_99_1day": result.get("var_99_1day"),
                "es_975_1day": result.get("es_975_1day"),
                "stressed_vol": result.get("stressed_annual_vol"),
            })

        pd.DataFrame(summary_rows).to_csv(
            self.output_dir / "stress_summary.csv", index=False
        )
        logger.info(f"  Reporte de stress guardado en {self.output_dir}")
