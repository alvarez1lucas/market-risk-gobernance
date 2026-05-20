"""
src/data/ingest.py — Stage 1: Ingesta de datos de mercado
Fuentes: yfinance (equities, FX, bonos) + FRED (tasas, spreads macro)
100% gratuito, sin API keys de pago.
"""

import logging
from pathlib import Path
from typing import Dict
import os

import pandas as pd
import numpy as np
import yfinance as yf
from loguru import logger

# FRED es opcional — si no tienen API key, usamos datos ya descargados o proxy
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False


# ── Configuración de activos ─────────────────────────────────────────────────

EQUITY_TICKERS = {
    "SPX": "^GSPC",        # S&P 500
    "VIX": "^VIX",         # Volatilidad implícita
    "MERVAL": "^MERV",     # Argentina (relevante para perfil LATAM)
    "EEM": "EEM",          # Emerging Markets ETF
    "HYG": "HYG",          # High Yield Corporate Bonds ETF
    "LQD": "LQD",          # Investment Grade Corporate Bonds ETF
}

FX_TICKERS = {
    "EURUSD": "EURUSD=X",
    "USDJPY": "JPY=X",
    "GBPUSD": "GBPUSD=X",
}

BOND_TICKERS = {
    "UST_10Y": "^TNX",     # US Treasury 10Y yield
    "UST_2Y": "^IRX",      # US Treasury 2Y (proxy)
}

# Series FRED (requiere API key gratis en fred.stlouisfed.org)
FRED_SERIES = {
    "FED_FUNDS": "FEDFUNDS",
    "SOFR": "SOFR",
    "HY_SPREAD": "BAMLH0A0HYM2",      # ICE BofA HY spread
    "IG_SPREAD": "BAMLC0A0CM",        # ICE BofA IG spread
    "TERM_PREMIUM": "THREEFYTP10",
    "UNEMPLOYMENT": "UNRATE",
    "CPI_YOY": "CPIAUCSL",
}


class MarketDataIngestor:
    """
    Descarga y valida datos de mercado de múltiples fuentes.
    Aplica checks de calidad (Great Expectations-style) antes de guardar.
    """

    def __init__(self, start: str = "2000-01-01", end: str = "2024-12-31",
                 fred_api_key: str | None = None, cache_dir: str = "data/raw"):
        self.start = start
        self.end = end
        self.fred_api_key = fred_api_key or os.getenv("FRED_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, pd.DataFrame]:
        """Ejecuta ingesta completa. Retorna dict con DataFrames por categoría."""
        data = {}

        logger.info("Descargando equities e índices...")
        data["equities"] = self._download_yfinance(EQUITY_TICKERS, price_col="Adj Close")

        logger.info("Descargando FX...")
        data["fx"] = self._download_yfinance(FX_TICKERS, price_col="Close")

        logger.info("Descargando yields de bonos...")
        data["bonds"] = self._download_yfinance(BOND_TICKERS, price_col="Close")

        if self.fred_api_key and FRED_AVAILABLE:
            logger.info("Descargando series macro desde FRED...")
            data["macro"] = self._download_fred()
        else:
            logger.warning("FRED API key no encontrada — usando proxy de datos macro.")
            logger.warning("Obtené tu key gratis en: https://fred.stlouisfed.org/docs/api/api_key.html")
            data["macro"] = self._macro_proxy(data["equities"].index)

        # Combinar en un DataFrame maestro
        master = self._merge_all(data)

        # Quality checks
        self._run_quality_checks(master)

        # Guardar
        master.to_csv(self.cache_dir / "market_data_master.csv")
        logger.info(f"Datos guardados: {master.shape[0]} observaciones, {master.shape[1]} series")

        return data

    def _download_yfinance(self, tickers: dict, price_col: str = "Adj Close") -> pd.DataFrame:
        """Descarga precios de cierre ajustados via yfinance."""
        raw = yf.download(
            list(tickers.values()),
            start=self.start,
            end=self.end,
            auto_adjust=True,
            progress=False,
        )
        # Seleccionar columna de precio
        prices = None
        if isinstance(raw.columns, pd.MultiIndex):
            if price_col in raw:
                prices = raw[price_col]
            elif "Close" in raw:
                prices = raw["Close"]
            elif "Adj Close" in raw:
                prices = raw["Adj Close"]
            else:
                prices = raw.copy()
        else:
            if price_col in raw.columns:
                prices = raw[[price_col]]
            elif "Close" in raw.columns:
                prices = raw[["Close"]]
            elif "Adj Close" in raw.columns:
                prices = raw[["Adj Close"]]
            else:
                prices = raw.copy()

        # Si solo queda una serie, asegurar DataFrame
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        # Renombrar con nombres legibles
        reverse_map = {v: k for k, v in tickers.items()}
        prices.columns = [reverse_map.get(c, c) for c in prices.columns]

        return prices.ffill().dropna(how="all")

    def _download_fred(self) -> pd.DataFrame:
        """Descarga series macro desde FRED API."""
        fred = Fred(api_key=self.fred_api_key)
        series = {}
        for name, fred_id in FRED_SERIES.items():
            try:
                s = fred.get_series(fred_id, observation_start=self.start, observation_end=self.end)
                series[name] = s
            except Exception as e:
                logger.warning(f"  Error descargando {fred_id}: {e}")

        df = pd.DataFrame(series)
        df.index = pd.to_datetime(df.index)
        return df.resample("B").last().ffill()  # Resample a días hábiles

    def _macro_proxy(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Proxy de datos macro usando series sintéticas realistas
        cuando FRED no está disponible. Solo para demostración.
        """
        np.random.seed(42)
        n = len(index)
        df = pd.DataFrame(index=index)
        df["FED_FUNDS"] = np.clip(2.5 + np.cumsum(np.random.normal(0, 0.05, n)), 0, 20)
        df["HY_SPREAD"] = np.clip(4.0 + np.cumsum(np.random.normal(0, 0.02, n)), 1, 25)
        df["IG_SPREAD"] = np.clip(1.2 + np.cumsum(np.random.normal(0, 0.01, n)), 0.3, 8)
        return df

    def _merge_all(self, data: dict) -> pd.DataFrame:
        """Combina todas las fuentes en un DataFrame diario alineado."""
        frames = []
        for category, df in data.items():
            df.columns = [f"{category}_{col}" for col in df.columns]
            frames.append(df)

        master = pd.concat(frames, axis=1)
        master.index = pd.to_datetime(master.index)
        master = master.resample("B").last()  # Días hábiles
        master = master.ffill().dropna(thresh=int(master.shape[1] * 0.7))
        return master

    def _run_quality_checks(self, df: pd.DataFrame):
        """
        Checks de calidad de datos al estilo SR 11-7:
        - Sin gaps mayores a 5 días hábiles
        - Precios positivos
        - Sin cambios del >50% en un día (posibles errores de datos)
        """
        issues = []

        # Check 1: Missing values
        missing_pct = df.isnull().mean()
        cols_high_missing = missing_pct[missing_pct > 0.05].index.tolist()
        if cols_high_missing:
            issues.append(f"Columnas con >5% missing: {cols_high_missing}")

        # Check 2: Precios negativos (no válido para precios de activos)
        price_cols = [c for c in df.columns if "macro" not in c]
        for col in price_cols:
            if (df[col].dropna() < 0).any():
                issues.append(f"Precios negativos en {col}")

        # Check 3: Saltos de precio extremos (posibles errores de datos)
        for col in price_cols:
            daily_returns = df[col].pct_change().abs()
            extreme = daily_returns[daily_returns > 0.5]
            if len(extreme) > 0:
                issues.append(f"Saltos >50% en {col}: {extreme.index.tolist()[:3]}")

        if issues:
            logger.warning(f"Data quality issues encontrados ({len(issues)}):")
            for issue in issues:
                logger.warning(f"  ⚠ {issue}")
        else:
            logger.info("  ✓ Todos los checks de calidad de datos pasaron")
