"""PatchTST: State-of-the-art Transformer for Time Series Forecasting.

Implements PatchTST (Patch Time Series Transformer) - current SOTA for time series.
Breaks sequences into patches (like Vision Transformers) for efficient multi-horizon prediction.

Reference: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (2023)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List
from loguru import logger


@dataclass
class ForecastResult:
    """Multi-horizon forecast result."""
    horizons: List[str]  # ["1h", "4h", "1d", "1w"]
    predictions: Dict[str, float]  # {horizon: price}
    probabilities: Dict[str, Dict[str, float]]  # {horizon: {bull/neutral/bear: prob}}
    confidence: Dict[str, float]  # {horizon: confidence}
    direction: str  # Overall direction
    overall_confidence: float


class PatchEmbedding(nn.Module):
    """Convert time series into patches and embed them."""

    def __init__(self, patch_len: int, d_model: int, input_dim: int = 5):
        super().__init__()
        self.patch_len = patch_len
        self.input_dim = input_dim

        # Linear projection for each patch
        self.projection = nn.Linear(patch_len * input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            patches: (batch, num_patches, d_model)
        """
        batch_size, seq_len, input_dim = x.shape

        # Calculate number of patches
        num_patches = seq_len // self.patch_len

        # Reshape into patches
        x = x[:, :num_patches * self.patch_len, :]  # Trim to fit patches
        x = x.reshape(batch_size, num_patches, self.patch_len * input_dim)

        # Project patches
        patches = self.projection(x)

        return patches


class TransformerBlock(nn.Module):
    """Transformer encoder block with multi-head attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Multi-head attention
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class PatchTSTModel(nn.Module):
    """PatchTST architecture for multi-horizon forecasting."""

    def __init__(
        self,
        input_dim: int = 5,  # OHLCV
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        patch_len: int = 16,
        pred_horizons: List[int] = [24, 96, 168, 672],  # 1h, 4h, 1d, 1w (in hours)
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.patch_len = patch_len
        self.pred_horizons = pred_horizons

        # Patch embedding
        self.patch_embedding = PatchEmbedding(patch_len, d_model, input_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))  # Max 100 patches

        # Transformer encoder
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Multi-horizon prediction heads
        self.prediction_heads = nn.ModuleDict({
            str(horizon): nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),  # Predict price change
            )
            for horizon in pred_horizons
        })

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            predictions: {horizon: (batch, 1)} for each horizon
        """
        # Patch embedding
        patches = self.patch_embedding(x)
        batch_size, num_patches, _ = patches.shape

        # Add positional encoding
        patches = patches + self.pos_encoding[:, :num_patches, :]

        # Transformer encoder
        for layer in self.encoder_layers:
            patches = layer(patches)

        # Global pooling (use CLS token approach - take mean)
        encoded = patches.mean(dim=1)  # (batch, d_model)

        # Multi-horizon predictions
        predictions = {}
        for horizon in self.pred_horizons:
            pred = self.prediction_heads[str(horizon)](encoded)
            predictions[horizon] = pred

        return predictions


class PatchTSTPredictor:
    """Time series transformer predictor with multi-horizon forecasting.

    Usage:
        predictor = PatchTSTPredictor()

        # Train on historical data
        predictor.train(historical_ohlcv, epochs=50)

        # Predict multiple horizons
        forecast = predictor.predict(recent_ohlcv)

        print(f"1h forecast: {forecast.predictions['1h']}")
        print(f"1d forecast: {forecast.predictions['1d']}")
        print(f"Overall direction: {forecast.direction}")
    """

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 6,
        patch_len: int = 16,
        device: str = "mps",
    ):
        self.device = device
        self.input_dim = input_dim

        # Prediction horizons (in hours)
        self.horizons = [1, 4, 24, 168]  # 1h, 4h, 1d, 1w
        self.horizon_names = ["1h", "4h", "1d", "1w"]

        # Create model
        self.model = PatchTSTModel(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            patch_len=patch_len,
            pred_horizons=self.horizons,
        ).to(device)

        self.is_trained = False
        logger.info(f"[PATCHTST] Initialized {num_layers}-layer transformer on {device}")

    def train(
        self,
        ohlcv_data: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
    ):
        """Train the transformer on historical OHLCV data.

        Args:
            ohlcv_data: Historical OHLCV (n_samples, seq_len, 5)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        logger.info(f"[PATCHTST] Training on {len(ohlcv_data)} sequences...")

        # Convert to tensor
        data = torch.FloatTensor(ohlcv_data).to(self.device)

        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for i in range(0, len(data) - batch_size, batch_size):
                batch = data[i:i+batch_size]

                # Forward pass
                predictions = self.model(batch)

                # Calculate loss (MSE for each horizon)
                loss = 0.0
                for horizon, pred in predictions.items():
                    # Target: actual price change at horizon
                    # For now, use simple random targets (in production, use actual future prices)
                    target = torch.randn_like(pred)  # TODO: Use real targets
                    loss += F.mse_loss(pred, target)

                loss = loss / len(predictions)

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            scheduler.step()

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            if (epoch + 1) % 10 == 0:
                logger.info(f"[PATCHTST] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        self.is_trained = True
        logger.success(f"[PATCHTST] Training complete!")

    @torch.no_grad()
    def predict(
        self,
        recent_ohlcv: np.ndarray,
        current_price: Optional[float] = None,
    ) -> ForecastResult:
        """Generate multi-horizon price forecast.

        Args:
            recent_ohlcv: Recent OHLCV data (seq_len, 5)
            current_price: Current price (for calculating % change)

        Returns:
            ForecastResult with predictions for all horizons
        """
        if not self.is_trained:
            logger.warning("[PATCHTST] Model not trained, using baseline")
            return self._baseline_forecast(recent_ohlcv, current_price)

        self.model.eval()

        # Prepare input
        x = torch.FloatTensor(recent_ohlcv).unsqueeze(0).to(self.device)

        # Get predictions
        pred_dict = self.model(x)

        # Convert to prices
        if current_price is None:
            current_price = float(recent_ohlcv[-1, 3])  # Last close

        predictions = {}
        probabilities = {}
        confidence = {}

        for i, (horizon, pred_tensor) in enumerate(pred_dict.items()):
            # Convert to price
            price_change_pct = float(pred_tensor.cpu().numpy()[0, 0])
            predicted_price = current_price * (1 + price_change_pct / 100)

            horizon_name = self.horizon_names[i]
            predictions[horizon_name] = predicted_price

            # Convert to directional probabilities
            if price_change_pct > 1.0:
                direction = "bullish"
                conf = min(abs(price_change_pct) / 5.0, 1.0)
                probs = {"bullish": conf, "neutral": 1-conf, "bearish": 0.0}
            elif price_change_pct < -1.0:
                direction = "bearish"
                conf = min(abs(price_change_pct) / 5.0, 1.0)
                probs = {"bearish": conf, "neutral": 1-conf, "bullish": 0.0}
            else:
                direction = "neutral"
                conf = 1.0 - abs(price_change_pct)
                probs = {"neutral": conf, "bullish": abs(price_change_pct)/2, "bearish": abs(price_change_pct)/2}

            probabilities[horizon_name] = probs
            confidence[horizon_name] = conf

        # Determine overall direction (weighted by horizon)
        weights = [0.4, 0.3, 0.2, 0.1]  # Shorter horizons more important
        overall_bullish = sum(probabilities[h]["bullish"] * w for h, w in zip(self.horizon_names, weights))
        overall_bearish = sum(probabilities[h]["bearish"] * w for h, w in zip(self.horizon_names, weights))
        overall_neutral = sum(probabilities[h]["neutral"] * w for h, w in zip(self.horizon_names, weights))

        if overall_bullish > max(overall_bearish, overall_neutral):
            overall_direction = "bullish"
            overall_conf = overall_bullish
        elif overall_bearish > max(overall_bullish, overall_neutral):
            overall_direction = "bearish"
            overall_conf = overall_bearish
        else:
            overall_direction = "neutral"
            overall_conf = overall_neutral

        result = ForecastResult(
            horizons=self.horizon_names,
            predictions=predictions,
            probabilities=probabilities,
            confidence=confidence,
            direction=overall_direction,
            overall_confidence=float(overall_conf),
        )

        logger.info(f"[PATCHTST] Forecast: {result.direction} (conf: {result.overall_confidence:.2f})")
        return result

    def _baseline_forecast(self, recent_ohlcv: np.ndarray, current_price: Optional[float]) -> ForecastResult:
        """Simple baseline when model not trained."""
        if current_price is None:
            current_price = float(recent_ohlcv[-1, 3])

        # Simple momentum baseline
        returns = np.diff(recent_ohlcv[-10:, 3]) / recent_ohlcv[-11:-1, 3]
        avg_return = np.mean(returns)

        predictions = {}
        probabilities = {}
        confidence = {}

        for horizon_name in self.horizon_names:
            predicted_price = current_price * (1 + avg_return)
            predictions[horizon_name] = predicted_price

            if avg_return > 0.001:
                probabilities[horizon_name] = {"bullish": 0.6, "neutral": 0.3, "bearish": 0.1}
                confidence[horizon_name] = 0.5
            elif avg_return < -0.001:
                probabilities[horizon_name] = {"bearish": 0.6, "neutral": 0.3, "bullish": 0.1}
                confidence[horizon_name] = 0.5
            else:
                probabilities[horizon_name] = {"neutral": 0.6, "bullish": 0.2, "bearish": 0.2}
                confidence[horizon_name] = 0.5

        return ForecastResult(
            horizons=self.horizon_names,
            predictions=predictions,
            probabilities=probabilities,
            confidence=confidence,
            direction="neutral",
            overall_confidence=0.5,
        )

    def save(self, path: str):
        """Save trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'horizons': self.horizons,
        }, path)
        logger.info(f"[PATCHTST] Model saved to {path}")

    def load(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        logger.info(f"[PATCHTST] Model loaded from {path}")
