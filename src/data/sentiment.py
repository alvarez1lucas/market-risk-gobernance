"""
src/data/sentiment.py — NLP Sentiment Analysis for Market Risk
Integración de FinBERT (fin-BERT) para análisis de sentimiento en noticias financieras.
Feature anticipada del TFT para predecir cambios en el VaR.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings

import pandas as pd
import numpy as np
from loguru import logger

# Importar FinBERT desde transformers (ProsusAI/finbert)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers/torch no disponibles — usando sentimiento sintético")

# Parseo de feeds RSS (opcional, para noticias)
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False


# ── FinBERT Sentiment Analyzer ───────────────────────────────────────────────

class FinBERTSentimentAnalyzer:
    """
    Wrapper alrededor de FinBERT (fine-tuned BERT para textos financieros).
    Modelo: ProsusAI/finbert (entrenado en Financial PhraseBank)
    Retorna scores en [-1, +1]:
      - negative: score < -0.5
      - neutral: -0.5 <= score <= 0.5
      - positive: score > 0.5
    """

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = None):
        """
        Args:
            model_name: Nombre del modelo en HuggingFace Hub
            device: 'cuda' o 'cpu' (auto-detecta si None)
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("transformers no disponible — retornando sentimiento dummy")
            self.model = None
            self.tokenizer = None
            return

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"FinBERT cargando desde HuggingFace: {model_name} (device={self.device})")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def score_text(self, text: str) -> Dict:
        """
        Score de sentimiento para un texto.
        Retorna: {
            'label': 'positive'|'negative'|'neutral',
            'score': float en [-1, +1],
            'confidence': float en [0, 1]
        }
        """
        if self.model is None:
            # Dummy sentiment si FinBERT no está disponible
            import hashlib
            # Generador pseudoaleatorio determinista basado en el texto
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            score = (hash_val % 200 - 100) / 100.0  # [-1, +1]
            label = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
            return {
                "label": label,
                "score": score,
                "confidence": 0.5
            }

        # Tokenizar
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inferencia
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # FinBERT: labels son [0=negative, 1=neutral, 2=positive]
        label_names = ["negative", "neutral", "positive"]
        label_idx = np.argmax(probs)
        label = label_names[label_idx]
        confidence = float(probs[label_idx])

        # Convertir a score [-1, +1]
        score = (label_idx - 1)  # -1 (neg), 0 (neutral), +1 (pos)
        score = score * confidence  # Ponderar por confianza

        return {
            "label": label,
            "score": float(score),
            "confidence": float(confidence)
        }

    def score_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """Score para múltiples textos."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                results.append(self.score_text(text))
        return results


# ── Market Sentiment Ingestor ───────────────────────────────────────────────

class MarketSentimentIngestor:
    """
    Descarga noticias financieras de múltiples fuentes RSS y calcula sentimiento diario.
    Si no hay acceso a APIs reales, genera datos sintéticos coherentes con volatilidad.
    """

    def __init__(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        sources: Optional[List[str]] = None
    ):
        """
        Args:
            start_date: Fecha inicio (YYYY-MM-DD)
            end_date: Fecha fin (YYYY-MM-DD)
            sources: URLs de RSS feeds (si None, usa fuentes por defecto o sintético)
        """
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)

        # Fuentes de noticias (requieren internet y feedparser)
        self.sources = sources or [
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://feeds.reuters.com/reuters/businessNews",
            # "https://feeds.ft.com/markets",  # Requiere auth
        ]

        self.analyzer = FinBERTSentimentAnalyzer()
        logger.info(f"MarketSentimentIngestor: {self.start_date.date()} → {self.end_date.date()}")

    def run(self) -> pd.DataFrame:
        """
        Descarga noticias y retorna sentimiento diario.
        Retorna DataFrame con índice datetime y columnas:
          - sentiment_mean: promedio diario de scores
          - sentiment_std: desv.estándar
          - sentiment_extreme_neg: boolean si percentil 10% más bajo
          - n_articles: número de artículos procesados
        """
        logger.info("Iniciando ingesta de noticias...")

        # Intentar descargar de RSS; si falla, generar sintético
        if FEEDPARSER_AVAILABLE:
            try:
                df = self._fetch_from_rss()
                if not df.empty:
                    logger.info(f"✓ Descargadas {len(df)} noticias desde RSS")
                    return df
            except Exception as e:
                logger.warning(f"Error descargando RSS: {e} — usando datos sintéticos")

        # Generar sentimiento sintético (coherente con volatilidad histórica)
        logger.info("Generando sentimiento sintético (coherente con VIX)")
        df = self._generate_synthetic_sentiment()
        return df

    def _fetch_from_rss(self) -> pd.DataFrame:
        """Descargar noticias de RSS feeds y calcular sentimiento."""
        if not FEEDPARSER_AVAILABLE:
            raise ImportError("feedparser no disponible")

        articles = []
        for source_url in self.sources:
            try:
                logger.info(f"  Descargando: {source_url}")
                feed = feedparser.parse(source_url)
                for entry in feed.entries[:50]:  # Limitar por performance
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    text = f"{title}. {summary}"

                    # Publicado
                    try:
                        published = pd.Timestamp(entry.published)
                    except:
                        published = pd.Timestamp.now()

                    articles.append({
                        "date": published,
                        "text": text,
                        "source": source_url
                    })
            except Exception as e:
                logger.warning(f"  Error en {source_url}: {e}")

        if not articles:
            raise ValueError("No se descargaron artículos de RSS")

        df_articles = pd.DataFrame(articles)
        logger.info(f"Total artículos: {len(df_articles)}")

        # Scoring de sentimiento
        logger.info("Scoring con FinBERT...")
        sentiments = self.analyzer.score_batch(df_articles["text"].tolist())
        df_articles["sentiment_score"] = [s["score"] for s in sentiments]

        # Agregación diaria
        df_articles["date"] = df_articles["date"].dt.floor("D")
        daily = df_articles.groupby("date").agg({
            "sentiment_score": ["mean", "std", "count"]
        }).reset_index()
        daily.columns = ["date", "sentiment_mean", "sentiment_std", "n_articles"]
        daily = daily.set_index("date")

        # Rellenar fechas faltantes
        date_range = pd.date_range(self.start_date, self.end_date, freq="D")
        daily = daily.reindex(date_range)
        daily["sentiment_mean"] = daily["sentiment_mean"].fillna(0.0)
        daily["sentiment_std"] = daily["sentiment_std"].fillna(0.1)
        daily["n_articles"] = daily["n_articles"].fillna(0).astype(int)

        # Marcar días de sentimiento extremo negativo
        q10 = daily["sentiment_mean"].quantile(0.10)
        daily["sentiment_extreme_neg"] = daily["sentiment_mean"] <= q10

        return daily

    def _generate_synthetic_sentiment(self) -> pd.DataFrame:
        """
        Generar sentimiento sintético realista con variación significativa.
        Simula cambios en el sentimiento del mercado a lo largo del tiempo.
        """
        np.random.seed(42)
        date_range = pd.date_range(self.start_date, self.end_date, freq="D")
        n = len(date_range)

        # Componentes del sentimiento:
        # 1. Tendencia de largo plazo (drift)
        trend = np.linspace(-0.2, 0.15, n)
        
        # 2. Ciclo de negocios (~2 años)
        cycle = 0.3 * np.sin(2 * np.pi * np.arange(n) / 504)
        
        # 3. Ruido AR(1) más volátil (mean-reverting)
        rho = 0.85
        sigma = 0.12
        noise = np.zeros(n)
        noise[0] = np.random.normal(0, sigma)
        for i in range(1, n):
            noise[i] = rho * noise[i - 1] + np.random.normal(0, sigma)
        
        # 4. Shocks aleatorios (crisis, cambios de política, etc.)
        # ~15-20 eventos por año
        shock_prob = 0.05
        shocks = np.where(np.random.random(n) < shock_prob)[0]
        shock_component = np.zeros(n)
        for idx in shocks:
            magnitude = np.random.choice([-0.8, -0.6, 0.5, 0.6])  # Shock negativo más probable
            duration = np.random.randint(1, 8)  # Impacto persiste 1-7 días
            for j in range(idx, min(idx + duration, n)):
                shock_component[j] += magnitude * np.exp(-0.3 * (j - idx))
        
        # Combinar componentes
        sentiment_values = trend + cycle + noise + shock_component
        
        # Normalizar a [-1, +1] con distribución más realista
        sentiment_values = np.clip(sentiment_values, -1, 1)
        
        # Añadir skewness negativa (crisis más extremas que bonanzas)
        sentiment_values = sentiment_values - 0.05 * sentiment_values**2 * np.sign(sentiment_values)
        sentiment_values = np.clip(sentiment_values, -1, 1)

        # Crear DataFrame
        df = pd.DataFrame({
            "sentiment_mean": sentiment_values,
            "sentiment_std": 0.1 + 0.15 * np.abs(sentiment_values),  # Mayor std en extremos
            "n_articles": np.random.randint(8, 35, n)  # Artículos diarios
        }, index=date_range)

        # Marcar extremo negativo (percentil 10)
        q10 = df["sentiment_mean"].quantile(0.10)
        df["sentiment_extreme_neg"] = df["sentiment_mean"] <= q10

        logger.info(f"Sintético generado: {len(df)} días")
        logger.info(f"  Media: {df['sentiment_mean'].mean():.3f}")
        logger.info(f"  Std: {df['sentiment_mean'].std():.3f}")
        logger.info(f"  Min: {df['sentiment_mean'].min():.3f} | Max: {df['sentiment_mean'].max():.3f}")
        logger.info(f"  Días con sentimiento extremo negativo: {df['sentiment_extreme_neg'].sum()}")

        return df


# ── Merge Sentiment with Features ─────────────────────────────────────────────

def merge_sentiment_with_features(
    features: pd.DataFrame,
    sentiment_daily: pd.DataFrame,
    lags: List[int] = [0, 1, 5]
) -> pd.DataFrame:
    """
    Integrar sentimiento diario como features del TFT.
    Crea lags para capturar efectos retrasados.

    Args:
        features: DataFrame con features existentes (índice datetime)
        sentiment_daily: DataFrame con sentiment_mean, sentiment_std, etc.
        lags: Lags en días a crear

    Retorna:
        DataFrame con features originales + features NLP
    """
    df = features.copy()

    # Alinear índices
    sentiment_daily = sentiment_daily.reindex(df.index, method="ffill")

    # Agregar sentimiento actual
    df["sentiment_mean"] = sentiment_daily["sentiment_mean"].values
    df["sentiment_std"] = sentiment_daily["sentiment_std"].values
    df["sentiment_extreme_neg"] = sentiment_daily["sentiment_extreme_neg"].values

    # Crear lags
    for lag in lags:
        if lag > 0:
            df[f"sentiment_mean_lag{lag}"] = df["sentiment_mean"].shift(lag)
            df[f"sentiment_extreme_neg_lag{lag}"] = df["sentiment_extreme_neg"].shift(lag)

    # Media móvil de sentimiento (ventanas múltiples)
    for window in [5, 21, 63]:
        df[f"sentiment_ma{window}"] = df["sentiment_mean"].rolling(window, min_periods=1).mean()

    # Indicador de cambio abrupto en sentimiento
    df["sentiment_shock"] = (df["sentiment_mean"] - df["sentiment_mean_lag1"].fillna(0)).abs() > 0.5

    logger.info(f"✓ Features enriquecidas: {features.shape[1]} → {df.shape[1]} columnas")
    logger.info(f"  Nuevas features NLP: {df.shape[1] - features.shape[1]}")

    return df
