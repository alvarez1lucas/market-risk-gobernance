"""
src/models/lstm_attention.py — LSTM con mecanismo de atención
Modelo challenger del TFT. Más liviano, más interpretable clásicamente.
Sirve como ablation study: TFT vs LSTM para justificar elección del champion.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
from pathlib import Path
from loguru import logger
from typing import Dict, Tuple


class BahdanauAttention(nn.Module):
    """
    Mecanismo de atención de Bahdanau (2015).
    Permite al modelo enfocarse en diferentes timesteps del pasado
    para predecir el riesgo futuro — crítico en crisis donde períodos
    específicos del pasado son más relevantes.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query: [batch, hidden] — estado oculto final del LSTM
        keys:  [batch, seq_len, hidden] — todos los estados ocultos
        Retorna: contexto [batch, hidden] y pesos de atención [batch, seq_len]
        """
        query_exp = query.unsqueeze(1).expand_as(keys)
        energy = self.v(torch.tanh(self.W_query(query_exp) + self.W_key(keys)))
        attention_weights = torch.softmax(energy.squeeze(-1), dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)
        return context, attention_weights


class LSTMWithAttention(nn.Module):
    """
    LSTM bidireccional + atención de Bahdanau para forecasting de riesgo.
    
    Arquitectura:
        Input → LSTM Encoder (bidireccional) → Atención → FC → Output (cuantiles)
    
    Output son cuantiles directos [1%, 2.5%, 5%, 50%, 95%, 97.5%, 99%]
    para estimar VaR y ES sin suponer distribución normal.
    """

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2,
                 output_quantiles: int = 7):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention opera sobre la dimensión hidden del LSTM bidireccional
        self.attention = BahdanauAttention(hidden_size * 2)

        # Capa de output — proyecta a cuantiles directamente
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_quantiles),
        )

        # Layer norm para estabilidad en series financieras
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [batch, seq_len, input_size]
        Retorna: predicciones [batch, output_quantiles], attention_weights [batch, seq_len]
        """
        lstm_out, (h_n, _) = self.lstm(x)

        # Combinar estados del último layer (forward + backward)
        h_forward = h_n[-2]   # Último layer, dirección forward
        h_backward = h_n[-1]  # Último layer, dirección backward
        query = torch.cat([h_forward, h_backward], dim=-1)

        # Atención sobre todos los timesteps
        context, attention_weights = self.attention(query, lstm_out)

        # Layer norm + output
        context_normed = self.layer_norm(context)
        output = self.fc_out(context_normed)

        return output, attention_weights


class QuantileLoss(nn.Module):
    """
    Pinball loss para regresión de cuantiles.
    Permite estimar directamente los cuantiles de la distribución de retornos
    sin asumir normalidad — fundamental para VaR en distribuciones con colas pesadas.
    """

    def __init__(self, quantiles: list):
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_exp = targets.unsqueeze(-1).expand_as(predictions)
        quantiles_exp = self.quantiles.to(predictions.device)
        errors = targets_exp - predictions
        loss = torch.max(
            quantiles_exp * errors,
            (quantiles_exp - 1) * errors
        )
        return loss.mean()


class LSTMAttentionModel:
    """Wrapper de entrenamiento para el LSTM con atención."""

    QUANTILES = [0.01, 0.025, 0.05, 0.5, 0.95, 0.975, 0.99]
    SEQ_LEN = 60         # 60 días de historia
    PREDICTION_LEN = 1   # Predice 1 día adelante (VaR 1-day)
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    BATCH_SIZE = 128
    MAX_EPOCHS = 100
    LR = 1e-3

    def __init__(self, features: pd.DataFrame, target_col: str = "log_return_SPX"):
        self.features = features
        self.target_col = target_col
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def train(self) -> Dict:
        """Entrena el modelo y retorna métricas."""
        logger.info(f"  Entrenando en: {self.device}")

        X, y = self._prepare_sequences()

        # Split 85/15 manteniendo orden temporal
        split = int(len(X) * 0.85)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.BATCH_SIZE,
            shuffle=False,  # No shuffle en series temporales
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=self.BATCH_SIZE,
            shuffle=False,
        )

        # Construir modelo
        input_size = X.shape[-1]
        self.model = LSTMWithAttention(
            input_size=input_size,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS,
            output_quantiles=len(self.QUANTILES),
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"  Parámetros LSTM: {n_params:,}")

        criterion = QuantileLoss(self.QUANTILES)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.MAX_EPOCHS)

        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        with mlflow.start_run(run_name="lstm_attention", nested=True):
            mlflow.log_params({
                "hidden_size": self.HIDDEN_SIZE,
                "num_layers": self.NUM_LAYERS,
                "seq_len": self.SEQ_LEN,
                "quantiles": self.QUANTILES,
            })

            for epoch in range(self.MAX_EPOCHS):
                # Train
                self.model.train()
                train_losses = []
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    optimizer.zero_grad()
                    preds, _ = self.model(X_batch)
                    loss = criterion(preds, y_batch)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    train_losses.append(loss.item())

                # Validation
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        preds, _ = self.model(X_batch)
                        val_losses.append(criterion(preds, y_batch).item())

                train_loss = np.mean(train_losses)
                val_loss = np.mean(val_losses)
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)

                scheduler.step()
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), "models/challenger/lstm_attention.pt")
                else:
                    patience_counter += 1

                if patience_counter >= 10:
                    logger.info(f"  Early stopping en epoch {epoch}")
                    break

                if epoch % 10 == 0:
                    logger.info(f"  Epoch {epoch:3d} | Train: {train_loss:.5f} | Val: {val_loss:.5f}")

        # Predicciones finales en validación
        self.model.load_state_dict(torch.load("models/challenger/lstm_attention.pt"))
        self.model.eval()
        all_preds, all_attention = [], []

        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(self.device)
                preds, attn = self.model(X_batch)
                all_preds.append(preds.cpu().numpy())
                all_attention.append(attn.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        var_99_preds = all_preds[:, 0]  # Cuantil 1% = VaR 99%

        mae = float(np.mean(np.abs(all_preds[:, 3] - y_val.numpy())))

        return {
            "val_loss": best_val_loss,
            "mae": mae,
            "predictions": var_99_preds,
            "all_quantiles": all_preds,
            "attention_weights": np.concatenate(all_attention, axis=0),
            "history": history,
            "model_params": n_params,
        }

    def _prepare_sequences(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convierte el DataFrame en secuencias para el LSTM."""
        feature_cols = [c for c in self.features.columns
                        if c != self.target_col and self.features[c].dtype in [np.float64, np.float32]]
        data = self.features[feature_cols + [self.target_col]].dropna()

        X_cols = feature_cols
        y_col = self.target_col

        X_data = data[X_cols].values.astype(np.float32)
        y_data = data[y_col].values.astype(np.float32)

        # Normalización por z-score (usando solo train para evitar data leakage)
        split = int(len(X_data) * 0.85)
        self.feature_mean = X_data[:split].mean(axis=0)
        self.feature_std = X_data[:split].std(axis=0) + 1e-8
        X_data = (X_data - self.feature_mean) / self.feature_std

        # Crear secuencias
        X_seqs, y_seqs = [], []
        for i in range(self.SEQ_LEN, len(X_data) - self.PREDICTION_LEN + 1):
            X_seqs.append(X_data[i - self.SEQ_LEN:i])
            y_seqs.append(y_data[i])

        return (
            torch.tensor(np.array(X_seqs), dtype=torch.float32),
            torch.tensor(np.array(y_seqs), dtype=torch.float32),
        )
