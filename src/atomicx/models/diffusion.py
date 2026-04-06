"""Diffusion Models for Price Trajectory Prediction.

Uses denoising diffusion probabilistic models (DDPM) to generate
multiple possible future price trajectories with uncertainty quantification.

This is cutting-edge: similar to how DALL-E generates images, but for time series.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class TrajectoryDistribution:
    """Output from diffusion model containing trajectory samples."""

    mean_trajectory: np.ndarray  # Shape: (horizon,)
    percentile_10: np.ndarray    # 10th percentile trajectory
    percentile_50: np.ndarray    # Median trajectory
    percentile_90: np.ndarray    # 90th percentile trajectory
    std_bands: np.ndarray        # Standard deviation at each timestep
    all_samples: np.ndarray      # All generated samples (num_samples, horizon)
    uncertainty: float            # Overall uncertainty score (0-1)

    def summary(self) -> dict:
        """Get summary statistics."""
        return {
            "mean_price_change_pct": float((self.mean_trajectory[-1] / self.mean_trajectory[0] - 1) * 100),
            "downside_10pct": float((self.percentile_10[-1] / self.percentile_10[0] - 1) * 100),
            "upside_90pct": float((self.percentile_90[-1] / self.percentile_90[0] - 1) * 100),
            "uncertainty": float(self.uncertainty),
            "confidence": 1.0 - self.uncertainty,
        }


class SinusoidalPositionEmbeddings(nn.Module):
    """Positional embeddings for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time embedding."""

    def __init__(self, dim: int, time_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, dim),
            nn.SiLU(),
        )
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        time_proj = self.mlp(time_emb)
        h = self.block(x + time_proj)
        return x + h  # Residual connection


class DiffusionUNet(nn.Module):
    """U-Net architecture for denoising trajectories."""

    def __init__(
        self,
        data_dim: int = 5,  # OHLCV
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_layers: int = 4,
    ):
        super().__init__()
        self.data_dim = data_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Input projection
        self.input_proj = nn.Linear(data_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, time_dim)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, data_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy trajectory (batch, seq_len, data_dim)
            t: Diffusion timestep (batch,)

        Returns:
            Predicted noise (batch, seq_len, data_dim)
        """
        # Time embedding
        time_emb = self.time_mlp(t)

        # Process sequence
        h = self.input_proj(x)

        for block in self.blocks:
            h = block(h, time_emb)

        return self.output_proj(h)


class DiffusionPredictor:
    """Diffusion model for generating price trajectory distributions.

    This model generates multiple possible future price paths by iteratively
    denoising from random noise. Similar to how image diffusion models work,
    but for time series prediction.

    Usage:
        predictor = DiffusionPredictor()
        predictor.train(historical_ohlcv_data)

        distribution = predictor.predict(
            current_state=recent_ohlcv,
            horizon=24,  # 24 hours ahead
            num_samples=1000
        )

        print(f"Mean prediction: ${distribution.mean_trajectory[-1]:.2f}")
        print(f"10th percentile: ${distribution.percentile_10[-1]:.2f}")
        print(f"90th percentile: ${distribution.percentile_90[-1]:.2f}")
        print(f"Uncertainty: {distribution.uncertainty:.2%}")
    """

    def __init__(
        self,
        data_dim: int = 5,
        hidden_dim: int = 128,
        diffusion_steps: int = 1000,
        device: str = "mps",  # Apple Silicon
    ):
        self.data_dim = data_dim
        self.diffusion_steps = diffusion_steps
        self.device = device

        # Create model
        self.model = DiffusionUNet(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
        ).to(device)

        # Diffusion schedule (cosine schedule is better than linear)
        self.betas = self._cosine_beta_schedule(diffusion_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.is_trained = False
        logger.info(f"[DIFFUSION] Initialized with {diffusion_steps} diffusion steps on {device}")

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule for beta values (better than linear)."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def train(self, ohlcv_data: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """Train the diffusion model on historical OHLCV data.

        Args:
            ohlcv_data: Historical OHLCV data (n_samples, seq_len, 5)
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        logger.info(f"[DIFFUSION] Training on {len(ohlcv_data)} sequences...")

        # Convert to tensor
        data = torch.FloatTensor(ohlcv_data).to(self.device)

        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = len(data) // batch_size

            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                if len(batch) < 2:
                    continue

                # Sample random timesteps
                t = torch.randint(0, self.diffusion_steps, (len(batch),), device=self.device)

                # Add noise according to diffusion schedule
                noise = torch.randn_like(batch)
                alpha_t = self.alphas_cumprod[t][:, None, None]
                noisy_batch = torch.sqrt(alpha_t) * batch + torch.sqrt(1 - alpha_t) * noise

                # Predict noise
                predicted_noise = self.model(noisy_batch, t)

                # Loss: MSE between predicted and actual noise
                loss = F.mse_loss(predicted_noise, noise)

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / num_batches
            if (epoch + 1) % 10 == 0:
                logger.info(f"[DIFFUSION] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        self.is_trained = True
        logger.success(f"[DIFFUSION] Training complete!")

    @torch.no_grad()
    def predict(
        self,
        current_state: np.ndarray,
        horizon: int = 24,
        num_samples: int = 1000,
    ) -> TrajectoryDistribution:
        """Generate multiple possible future trajectories.

        Args:
            current_state: Recent OHLCV data (context_len, 5)
            horizon: Number of timesteps to predict ahead
            num_samples: Number of trajectory samples to generate

        Returns:
            TrajectoryDistribution with multiple trajectory samples
        """
        if not self.is_trained:
            logger.warning("[DIFFUSION] Model not trained, using random walk baseline")
            return self._random_walk_baseline(current_state, horizon, num_samples)

        self.model.eval()

        # Start from pure noise
        shape = (num_samples, horizon, self.data_dim)
        x_t = torch.randn(shape, device=self.device)

        # Iteratively denoise (reverse diffusion process)
        for t in reversed(range(self.diffusion_steps)):
            t_tensor = torch.full((num_samples,), t, device=self.device, dtype=torch.long)

            # Predict noise
            predicted_noise = self.model(x_t, t_tensor)

            # Denoise step
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]

            # Compute mean
            x_t = (x_t - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise) / torch.sqrt(alpha_t)

            # Add noise (except for last step)
            if t > 0:
                noise = torch.randn_like(x_t)
                x_t += torch.sqrt(beta_t) * noise

        # Convert to numpy
        samples = x_t.cpu().numpy()  # Shape: (num_samples, horizon, 5)

        # Extract close prices
        close_prices = samples[:, :, 3]  # Close is 4th column (index 3)

        # Calculate statistics
        mean_trajectory = close_prices.mean(axis=0)
        percentile_10 = np.percentile(close_prices, 10, axis=0)
        percentile_50 = np.percentile(close_prices, 50, axis=0)
        percentile_90 = np.percentile(close_prices, 90, axis=0)
        std_bands = close_prices.std(axis=0)

        # Uncertainty: normalized std at final timestep
        final_std = std_bands[-1]
        final_price = mean_trajectory[-1]
        uncertainty = min(final_std / final_price, 1.0)

        distribution = TrajectoryDistribution(
            mean_trajectory=mean_trajectory,
            percentile_10=percentile_10,
            percentile_50=percentile_50,
            percentile_90=percentile_90,
            std_bands=std_bands,
            all_samples=close_prices,
            uncertainty=float(uncertainty),
        )

        logger.info(f"[DIFFUSION] Generated {num_samples} trajectories, uncertainty: {uncertainty:.2%}")
        return distribution

    def _random_walk_baseline(
        self, current_state: np.ndarray, horizon: int, num_samples: int
    ) -> TrajectoryDistribution:
        """Fallback: random walk if model not trained."""
        current_price = current_state[-1, 3]  # Last close price

        # Generate random walk samples
        returns = np.random.normal(0, 0.01, (num_samples, horizon))
        cum_returns = np.cumsum(returns, axis=1)
        trajectories = current_price * (1 + cum_returns)

        # Statistics
        mean_trajectory = trajectories.mean(axis=0)
        percentile_10 = np.percentile(trajectories, 10, axis=0)
        percentile_50 = np.percentile(trajectories, 50, axis=0)
        percentile_90 = np.percentile(trajectories, 90, axis=0)
        std_bands = trajectories.std(axis=0)

        uncertainty = 0.5  # Medium uncertainty for random walk

        return TrajectoryDistribution(
            mean_trajectory=mean_trajectory,
            percentile_10=percentile_10,
            percentile_50=percentile_50,
            percentile_90=percentile_90,
            std_bands=std_bands,
            all_samples=trajectories,
            uncertainty=uncertainty,
        )

    def save(self, path: str):
        """Save trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'diffusion_steps': self.diffusion_steps,
            'data_dim': self.data_dim,
        }, path)
        logger.info(f"[DIFFUSION] Model saved to {path}")

    def load(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        logger.info(f"[DIFFUSION] Model loaded from {path}")
