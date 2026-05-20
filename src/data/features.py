"""
src/data/features.py — Stage 2: Feature engineering financiero
Genera features estándar de mercado para modelos de riesgo.
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict


class FeatureEngine:
    """
    Construye features para modelos de VaR y ES:
    - Log-returns
    - Volatilidad realizada (múltiples ventanas)
    - Correlaciones rolling
    - Features de régimen de mercado
    - Indicadores técnicos relevantes para riesgo
    """

    def __init__(self, raw_data: Dict[str, pd.DataFrame]):
        self.raw = raw_data
        self.equities = raw_data.get("equities", pd.DataFrame())
        self.fx = raw_data.get("fx", pd.DataFrame())
        self.bonds = raw_data.get("bonds", pd.DataFrame())
        self.macro = raw_data.get("macro", pd.DataFrame())

        self.equities = self._normalize_columns(self.equities, "equities_")
        self.fx = self._normalize_columns(self.fx, "fx_")
        self.bonds = self._normalize_columns(self.bonds, "bonds_")
        self.macro = self._normalize_columns(self.macro, "macro_")

    def _normalize_columns(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if df.empty:
            return df
        if all(isinstance(col, str) and col.startswith(prefix) for col in df.columns):
            rename_map = {col: col[len(prefix):] for col in df.columns}
            return df.rename(columns=rename_map)
        return df

    def run(self) -> pd.DataFrame:
        logger.info("Calculando features financieras...")
        features = pd.DataFrame(index=self.equities.index)

        features = self._add_log_returns(features)
        features = self._add_realized_volatility(features)
        features = self._add_rolling_correlations(features)
        features = self._add_regime_features(features)
        features = self._add_macro_features(features)
        features = self._add_temporal_features(features)

        # Eliminar NaNs del inicio (por ventanas rolling)
        features = features.dropna()

        logger.info(f"  Features generadas: {features.shape[1]} columnas")
        features.to_csv("data/raw/features.csv")
        return features

    def _add_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log-returns para todos los activos (más estables que returns aritméticos)."""
        for col in self.equities.columns:
            df[f"log_return_{col}"] = np.log(self.equities[col] / self.equities[col].shift(1))

        for col in self.fx.columns:
            df[f"log_return_{col}"] = np.log(self.fx[col] / self.fx[col].shift(1))

        # Cambios en yields (no log-return sino diferencia en bps)
        for col in self.bonds.columns:
            df[f"yield_change_{col}"] = self.bonds[col].diff()

        return df

    def _add_realized_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volatilidad realizada en múltiples ventanas.
        Basel III usa 250 días para VaR; también calculamos 21 (mensual) y 63 (trimestral).
        """
        key_assets = [c for c in self.equities.columns if c in ["SPX", "EEM", "HYG"]]
        windows = {"21d": 21, "63d": 63, "250d": 250}

        for asset in key_assets:
            ret_col = f"log_return_{asset}"
            if ret_col not in df.columns:
                continue
            for name, window in windows.items():
                df[f"realized_vol_{asset}_{name}"] = (
                    df[ret_col].rolling(window).std() * np.sqrt(252)
                )

        # VIX como proxy de vol implícita (si está disponible)
        if "VIX" in self.equities.columns:
            df["vix_level"] = self.equities["VIX"]
            df["vix_change"] = self.equities["VIX"].diff()
            # Term structure (vol implícita vs realizada) — señal de estrés
            if "log_return_SPX" in df.columns:
                realized_21 = df["log_return_SPX"].rolling(21).std() * np.sqrt(252) * 100
                df["vol_risk_premium"] = self.equities["VIX"] - realized_21

        return df

    def _add_rolling_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Correlaciones rolling entre activos clave.
        Las correlaciones aumentan en crisis (efecto de contagio) — feature crítica para VaR.
        """
        if "log_return_SPX" in df.columns and "log_return_HYG" in df.columns:
            df["corr_spx_hyg_63d"] = (
                df["log_return_SPX"]
                .rolling(63)
                .corr(df["log_return_HYG"])
            )

        if "log_return_SPX" in df.columns and "log_return_EURUSD" in df.columns:
            df["corr_spx_eurusd_63d"] = (
                df["log_return_SPX"]
                .rolling(63)
                .corr(df["log_return_EURUSD"])
            )

        return df

    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Indicadores de régimen de mercado (bull/bear/crisis).
        Crítico para modelos de riesgo que deben comportarse diferente en crisis.
        """
        if "log_return_SPX" in df.columns:
            # Drawdown desde máximo
            spx_prices = self.equities.get("SPX", pd.Series())
            if not spx_prices.empty:
                rolling_max = spx_prices.rolling(252, min_periods=1).max()
                df["spx_drawdown"] = (spx_prices / rolling_max) - 1

            # Tendencia de medio plazo
            df["spx_momentum_63d"] = self.equities["SPX"].pct_change(63)
            df["spx_momentum_21d"] = self.equities["SPX"].pct_change(21)

        # Crisis flag (drawdown > 20% o VIX > 30)
        if "spx_drawdown" in df.columns and "vix_level" in df.columns:
            df["crisis_flag"] = (
                (df["spx_drawdown"] < -0.20) | (df["vix_level"] > 30)
            ).astype(int)

        return df

    def _add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features macro del ciclo económico."""
        if self.macro.empty:
            return df

        for col in self.macro.columns:
            # Alinear al índice de equities (macro puede ser mensual)
            aligned = self.macro[col].reindex(df.index, method="ffill")
            df[f"macro_{col}"] = aligned

            # Cambio del nivel macro
            df[f"macro_{col}_change"] = aligned.diff(21)  # Cambio mensual

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features temporales — estacionalidad y efectos de calendario."""
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["is_month_end"] = df.index.is_month_end.astype(int)
        df["is_quarter_end"] = df.index.is_quarter_end.astype(int)

        return df
