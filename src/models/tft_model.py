"""
src/models/tft_model.py — Temporal Fusion Transformer para Market Risk
Arquitectura de estado del arte para series temporales financieras.
Referencia: Lim et al. (2021) "Temporal Fusion Transformers for Interpretable
            Multi-horizon Time Series Forecasting"
"""

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import mlflow
from pathlib import Path
from loguru import logger
from typing import Dict, Optional

try:
    from pytorch_forecasting import (
        TemporalFusionTransformer,
        TimeSeriesDataSet,
        QuantileLoss,
    )
    from pytorch_forecasting.data import GroupNormalizer
    TFT_AVAILABLE = True
except ImportError:
    logger.warning("pytorch-forecasting no instalado. Usando TFT simplificado.")
    TFT_AVAILABLE = False


# ── Parámetros del modelo ────────────────────────────────────────────────────

TFT_CONFIG = {
    "max_encoder_length": 60,          # 60 días de contexto (~3 meses)
    "max_prediction_length": 10,       # Predice 10 días hacia adelante
    "hidden_size": 64,
    "attention_head_size": 4,
    "dropout": 0.1,
    "hidden_continuous_size": 32,
    "quantiles": [0.01, 0.025, 0.05, 0.5, 0.95, 0.975, 0.99],  # VaR levels
    "learning_rate": 1e-3,
    "batch_size": 64,
    "max_epochs": 50,
    "gradient_clip_val": 0.1,          # Importante para estabilidad en series financieras
}


class TFTRiskModel:
    """
    Wrapper del Temporal Fusion Transformer para estimación de VaR y ES.
    
    El TFT es especialmente adecuado para riesgo de mercado porque:
    - Maneja múltiples horizontes (1-day VaR, 10-day VaR)
    - Produce distribuciones completas (no solo punto estimado)
    - Cuantiles directos → VaR sin suposiciones de normalidad
    - Mecanismo de atención interpretable → cumple SR 11-7 explainability
    - Distingue entre variables conocidas en el futuro (calendario) y variables
      solo conocidas hasta el presente (retornos, spreads)
    """

    def __init__(self, features: pd.DataFrame, config: dict = TFT_CONFIG,
                 target_col: str = "log_return_SPX"):
        self.features = features
        self.config = config
        self.target_col = target_col
        self.model = None
        self.trainer = None
        self.dataset_train = None
        self.checkpoint_path = Path("models/champion/tft_model.ckpt")
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def train(self) -> Dict:
        """Entrena el TFT y retorna métricas de validación."""
        logger.info("Preparando dataset para TFT...")
        data = self._prepare_dataset()

        if TFT_AVAILABLE:
            return self._train_pytorch_forecasting(data)
        else:
            return self._train_simplified(data)

    def _prepare_dataset(self) -> pd.DataFrame:
        """
        Prepara el DataFrame en el formato requerido por pytorch-forecasting.
        Requiere columna 'time_idx' (int), 'group_id' (str para multi-serie),
        y todas las features.
        """
        df = self.features.copy()
        df = df.dropna()

        # Agregar índice temporal requerido por TimeSeriesDataSet
        df = df.reset_index()
        df.rename(columns={"index": "date"}, inplace=True)
        df["time_idx"] = range(len(df))
        df["group"] = "SPX_portfolio"  # Single portfolio — extendible a multi-asset

        return df

    def _train_pytorch_forecasting(self, data: pd.DataFrame) -> Dict:
        """Entrenamiento completo con pytorch-forecasting."""
        
        # Variables conocidas solo en el pasado (series de mercado)
        time_varying_unknown = [c for c in data.columns if c.startswith((
            "log_return_", "yield_change_", "realized_vol_", "corr_", "macro_"
        ))]
        
        # Variables conocidas en el futuro (calendario)
        time_varying_known = ["day_of_week", "month", "quarter",
                              "is_month_end", "is_quarter_end"]

        # Filtrar columnas que realmente existen
        time_varying_unknown = [c for c in time_varying_unknown if c in data.columns]
        time_varying_known = [c for c in time_varying_known if c in data.columns]

        # Split train / validation
        max_idx = data["time_idx"].max()
        val_cutoff = int(max_idx * 0.85)

        training = TimeSeriesDataSet(
            data[data["time_idx"] <= val_cutoff],
            time_idx="time_idx",
            target=self.target_col,
            group_ids=["group"],
            min_encoder_length=self.config["max_encoder_length"] // 2,
            max_encoder_length=self.config["max_encoder_length"],
            min_prediction_length=1,
            max_prediction_length=self.config["max_prediction_length"],
            time_varying_known_reals=time_varying_known,
            time_varying_unknown_reals=time_varying_unknown,
            target_normalizer=GroupNormalizer(groups=["group"]),
            add_relative_time_idx=True,
            add_target_scales=True,
        )

        validation = TimeSeriesDataSet.from_dataset(
            training,
            data[data["time_idx"] > val_cutoff],
            predict=True,
        )

        train_dl = training.to_dataloader(
            train=True, batch_size=self.config["batch_size"], num_workers=0
        )
        val_dl = validation.to_dataloader(
            train=False, batch_size=self.config["batch_size"] * 2, num_workers=0
        )

        # Construir modelo
        self.model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=self.config["learning_rate"],
            hidden_size=self.config["hidden_size"],
            attention_head_size=self.config["attention_head_size"],
            dropout=self.config["dropout"],
            hidden_continuous_size=self.config["hidden_continuous_size"],
            loss=QuantileLoss(quantiles=self.config["quantiles"]),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

        if not isinstance(self.model, pl.LightningModule):
            logger.warning(
                "TemporalFusionTransformer is not compatible with the installed Lightning version. "
                "Falling back to simplified LSTM training."
            )
            return self._train_simplified(data)

        logger.info(f"  Parámetros del TFT: {sum(p.numel() for p in self.model.parameters()):,}")

        # Trainer con Early Stopping
        from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            ModelCheckpoint(
                dirpath="models/champion",
                filename="tft_model",
                monitor="val_loss",
                save_top_k=1,
            ),
        ]

        self.trainer = pl.Trainer(
            max_epochs=self.config["max_epochs"],
            gradient_clip_val=self.config["gradient_clip_val"],
            callbacks=callbacks,
            enable_progress_bar=True,
            logger=False,  # Usamos MLflow directamente
        )

        with mlflow.start_run(run_name="tft_training", nested=True):
            mlflow.log_params(self.config)
            try:
                self.trainer.fit(self.model, train_dl, val_dl)
            except TypeError as e:
                logger.error("TFT training failed due to compatibility issue: %s", e)
                return self._train_simplified(data)
            val_loss = self.trainer.callback_metrics.get("val_loss", torch.tensor(float("nan")))
            mlflow.log_metric("val_loss", float(val_loss))

        # Predicciones en validación para backtesting
        predictions = self.model.predict(val_dl, return_y=True)

        # El cuantil 0.01 es el VaR 99%
        var_99_predictions = predictions.output[:, 0].numpy()  # Primer cuantil = 1%

        return {
            "val_loss": float(val_loss),
            "mae": float(torch.mean(torch.abs(predictions.output[:, 3] - predictions.y[0])).item()),
            "predictions": var_99_predictions,
            "all_quantiles": predictions.output.numpy(),
            "model_params": sum(p.numel() for p in self.model.parameters()),
        }

    def _train_simplified(self, data: pd.DataFrame) -> Dict:
        """
        Implementación simplificada si pytorch-forecasting no está disponible.
        Usa un LSTM simple como proxy para mostrar la estructura del código.
        """
        logger.warning("Usando TFT simplificado (pytorch-forecasting no disponible)")

        from src.models.lstm_attention import LSTMAttentionModel
        lstm = LSTMAttentionModel(self.features)
        results = lstm.train()
        results["note"] = "TFT simplificado — instalar pytorch-forecasting para TFT completo"
        return results

    def get_var_forecast(self, confidence_level: float = 0.99) -> np.ndarray:
        """
        Retorna forecast de VaR al nivel de confianza dado.
        confidence_level=0.99 → VaR 99% (Basel Internal Models)
        confidence_level=0.975 → VaR 97.5% (ES Basel III)
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado — correr train() primero")

        quantile_map = {0.99: 0, 0.975: 1, 0.95: 2}
        quantile_idx = quantile_map.get(confidence_level, 0)

        # Retornar el cuantil correspondiente
        logger.info(f"VaR forecast al {confidence_level:.1%}")
        return self.model  # Placeholder — implementar con datos reales

    def get_attention_weights(self) -> Optional[Dict]:
        """
        Extrae pesos de atención del TFT — requerido para explainability SR 11-7.
        Muestra qué features y períodos el modelo considera más importantes.
        """
        if self.model is None:
            return None

        try:
            # pytorch-forecasting expone interpretation directamente
            interpretation = self.model.interpret_output(
                self.model.predict(self.dataset_train, return_attention=True),
                reduction="sum",
            )
            return {
                "attention_weights": interpretation["attention"].numpy(),
                "variable_importance": interpretation["encoder_variables"].numpy(),
            }
        except Exception as e:
            logger.warning(f"No se pudieron extraer attention weights: {e}")
            return None
