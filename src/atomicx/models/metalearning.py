"""Meta-Learning for Fast Adaptation to Regime Shifts.

Implements MAML (Model-Agnostic Meta-Learning) to enable rapid adaptation
to new market regimes in hours instead of weeks.

Pre-trains on historical regime shifts (COVID crash, FTX collapse, etc.)
then adapts quickly when new regimes are detected.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from loguru import logger
from copy import deepcopy


@dataclass
class RegimeEpisode:
    """Single regime episode for meta-training."""
    name: str                    # e.g., "covid_crash_2020"
    support_data: np.ndarray     # Training data for this regime
    query_data: np.ndarray       # Test data for this regime
    labels: np.ndarray           # Ground truth labels (buy/sell/hold)


class BasePredictor(nn.Module):
    """Base neural network for prediction."""

    def __init__(self, input_dim: int = 46, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 3 classes: bullish, neutral, bearish
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MAMLPredictor:
    """Model-Agnostic Meta-Learning for fast regime adaptation.

    Usage:
        # Phase 1: Meta-train on historical regimes
        maml = MAMLPredictor()
        historical_regimes = [
            RegimeEpisode("covid_crash_2020", support, query, labels),
            RegimeEpisode("ftx_collapse_2022", support, query, labels),
            RegimeEpisode("luna_crash_2022", support, query, labels),
            RegimeEpisode("china_ban_2021", support, query, labels),
        ]
        maml.meta_train(historical_regimes, iterations=1000)

        # Phase 2: When new regime detected, adapt quickly
        maml.fast_adapt(
            new_regime_data=last_24h_data,
            new_regime_labels=last_24h_labels,
            num_steps=10  # Adapt in just 10 gradient steps
        )

        # Phase 3: Make predictions in new regime
        prediction = maml.predict(current_variables)
    """

    def __init__(
        self,
        input_dim: int = 46,
        hidden_dim: int = 128,
        meta_lr: float = 1e-3,
        inner_lr: float = 1e-2,
        device: str = "mps",
    ):
        self.input_dim = input_dim
        self.device = device
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr

        # Create base model
        self.model = BasePredictor(input_dim, hidden_dim).to(device)

        # Meta-optimizer (updates model across tasks)
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)

        self.is_meta_trained = False
        logger.info(f"[MAML] Initialized with meta_lr={meta_lr}, inner_lr={inner_lr}")

    def meta_train(
        self,
        regime_episodes: List[RegimeEpisode],
        iterations: int = 1000,
        inner_steps: int = 5,
    ):
        """Meta-train on multiple historical regime shifts.

        Args:
            regime_episodes: List of historical regime episodes
            iterations: Number of meta-training iterations
            inner_steps: Number of gradient steps per regime (inner loop)
        """
        logger.info(f"[MAML] Meta-training on {len(regime_episodes)} historical regimes...")

        self.model.train()

        for iteration in range(iterations):
            # Sample batch of regimes
            sampled_regimes = np.random.choice(regime_episodes, size=min(4, len(regime_episodes)), replace=False)

            meta_loss = 0.0

            for regime in sampled_regimes:
                # Clone model for this regime
                task_model = deepcopy(self.model)

                # Convert to tensors
                support_x = torch.FloatTensor(regime.support_data).to(self.device)
                support_y = torch.LongTensor(regime.labels).to(self.device)
                query_x = torch.FloatTensor(regime.query_data).to(self.device)
                query_y = torch.LongTensor(regime.labels).to(self.device)

                # Inner loop: adapt to this regime
                for _ in range(inner_steps):
                    # Forward pass on support set
                    support_pred = task_model(support_x)
                    support_loss = F.cross_entropy(support_pred, support_y)

                    # Inner gradient step
                    grads = torch.autograd.grad(support_loss, task_model.parameters(), create_graph=True)
                    task_model = self._apply_gradients(task_model, grads, self.inner_lr)

                # Outer loop: evaluate on query set
                query_pred = task_model(query_x)
                query_loss = F.cross_entropy(query_pred, query_y)

                meta_loss += query_loss

            # Meta-update (outer loop)
            meta_loss = meta_loss / len(sampled_regimes)
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()

            # Logging
            if (iteration + 1) % 100 == 0:
                logger.info(f"[MAML] Iteration {iteration+1}/{iterations}, Meta Loss: {meta_loss.item():.4f}")

        self.is_meta_trained = True
        logger.success(f"[MAML] Meta-training complete! Ready for fast adaptation.")

    def _apply_gradients(self, model: nn.Module, grads: tuple, lr: float) -> nn.Module:
        """Apply gradients to create updated model (for inner loop)."""
        updated_model = deepcopy(model)

        for (name, param), grad in zip(model.named_parameters(), grads):
            if grad is not None:
                updated_param = param - lr * grad
                # Update parameter in cloned model
                self._set_parameter(updated_model, name, updated_param)

        return updated_model

    def _set_parameter(self, model: nn.Module, name: str, value: torch.Tensor):
        """Set parameter value in model."""
        parts = name.split('.')
        module = model
        for part in parts[:-1]:
            module = getattr(module, part)
        setattr(module, parts[-1], nn.Parameter(value))

    @torch.no_grad()
    def fast_adapt(
        self,
        new_regime_data: np.ndarray,
        new_regime_labels: np.ndarray,
        num_steps: int = 10,
    ):
        """Quickly adapt to a new regime in just a few gradient steps.

        Args:
            new_regime_data: Recent data from new regime (e.g., last 24h)
            new_regime_labels: Ground truth labels for adaptation
            num_steps: Number of adaptation steps (default: 10)
        """
        if not self.is_meta_trained:
            logger.warning("[MAML] Model not meta-trained, fast adaptation may be suboptimal")

        logger.info(f"[MAML] Fast adapting to new regime in {num_steps} steps...")

        # Convert to tensors
        x = torch.FloatTensor(new_regime_data).to(self.device)
        y = torch.LongTensor(new_regime_labels).to(self.device)

        # Fine-tune with higher learning rate (fast adaptation)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)

        self.model.train()
        for step in range(num_steps):
            pred = self.model(x)
            loss = F.cross_entropy(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 5 == 0:
                accuracy = (pred.argmax(dim=1) == y).float().mean().item()
                logger.info(f"[MAML] Adaptation step {step+1}/{num_steps}, Loss: {loss.item():.4f}, Acc: {accuracy:.2%}")

        logger.success(f"[MAML] Adaptation complete! Model ready for new regime predictions.")

    @torch.no_grad()
    def predict(self, variables: Dict[str, float]) -> Dict[str, float]:
        """Make prediction using adapted model.

        Args:
            variables: Dictionary of variable values (46 variables)

        Returns:
            Dictionary with prediction and confidence
        """
        self.model.eval()

        # Convert variables dict to array
        # Assuming variables are ordered consistently
        x = np.array([variables.get(f"VAR_{i}", 0.0) for i in range(self.input_dim)])
        x_tensor = torch.FloatTensor(x).unsqueeze(0).to(self.device)

        # Forward pass
        logits = self.model(x_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # Class labels
        classes = ["bearish", "neutral", "bullish"]
        predicted_class = classes[probs.argmax()]
        confidence = float(probs.max())

        return {
            "direction": predicted_class,
            "confidence": confidence,
            "probabilities": {
                "bearish": float(probs[0]),
                "neutral": float(probs[1]),
                "bullish": float(probs[2]),
            }
        }

    def save(self, path: str):
        """Save meta-learned model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'is_meta_trained': self.is_meta_trained,
        }, path)
        logger.info(f"[MAML] Model saved to {path}")

    def load(self, path: str):
        """Load meta-learned model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
        self.is_meta_trained = checkpoint['is_meta_trained']
        logger.info(f"[MAML] Model loaded from {path}")


class RegimeLibrary:
    """Library of historical regime shifts for meta-training."""

    KNOWN_REGIMES = [
        "covid_crash_2020_march",
        "covid_recovery_2020_may",
        "defi_summer_2020_aug",
        "bitcoin_ath_2021_april",
        "china_ban_2021_may",
        "el_salvador_2021_sept",
        "tapering_fears_2021_nov",
        "russia_ukraine_2022_feb",
        "luna_collapse_2022_may",
        "celsius_freeze_2022_june",
        "ftx_collapse_2022_nov",
        "silvergate_crisis_2023_march",
        "sec_lawsuits_2023_june",
        "etf_approval_2024_jan",
        "halving_2024_april",
        "iran_war_2026_march",
    ]

    @staticmethod
    def load_regime_data(regime_name: str) -> Optional[RegimeEpisode]:
        """Load historical data for a specific regime.

        In production, this would query your database for the relevant time period.
        For now, returns placeholder structure.

        Args:
            regime_name: Name of regime (e.g., "covid_crash_2020_march")

        Returns:
            RegimeEpisode with support/query data
        """
        # TODO: Implement actual database query
        logger.warning(f"[REGIME] Placeholder data for {regime_name} - implement DB query")

        # Placeholder: generate synthetic data
        support_data = np.random.randn(100, 46)  # 100 samples, 46 variables
        query_data = np.random.randn(50, 46)
        labels = np.random.randint(0, 3, 100)  # 0=bearish, 1=neutral, 2=bullish

        return RegimeEpisode(
            name=regime_name,
            support_data=support_data,
            query_data=query_data,
            labels=labels,
        )

    @classmethod
    def load_all_regimes(cls) -> List[RegimeEpisode]:
        """Load all known historical regimes."""
        logger.info(f"[REGIME] Loading {len(cls.KNOWN_REGIMES)} historical regimes...")

        episodes = []
        for regime_name in cls.KNOWN_REGIMES:
            episode = cls.load_regime_data(regime_name)
            if episode:
                episodes.append(episode)

        logger.success(f"[REGIME] Loaded {len(episodes)} regime episodes")
        return episodes
