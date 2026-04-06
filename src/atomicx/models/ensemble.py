"""Ensemble Prediction Architecture - Meta-Learner.

Combines predictions from multiple models with adaptive weighting:
- Causal Model (existing VAR)
- Transformer (PatchTST)
- Graph Neural Network
- Swarm Simulation (existing)
- Diffusion Model

Uses meta-learning to determine optimal weights based on recent performance.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque
from loguru import logger


@dataclass
class ModelPrediction:
    """Single model's prediction output."""
    model_name: str
    direction: str  # "bullish", "neutral", "bearish"
    confidence: float  # 0-1
    probabilities: Dict[str, float]  # {bullish: 0.6, neutral: 0.2, bearish: 0.2}
    uncertainty: Optional[float] = None  # For models that provide it
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsemblePrediction:
    """Final ensemble prediction with uncertainty."""
    direction: str
    confidence: float
    probabilities: Dict[str, float]
    uncertainty: float
    model_weights: Dict[str, float]  # Contribution of each model
    individual_predictions: List[ModelPrediction]
    regime_context: Optional[str] = None


class EnsemblePredictor:
    """Meta-learner that combines predictions from multiple models.

    Features:
    - Adaptive weighting based on recent performance
    - Regime-aware model selection
    - Uncertainty aggregation
    - Disagreement detection (when models conflict)

    Usage:
        ensemble = EnsemblePredictor()

        # Register models
        ensemble.register_model("causal", causal_model)
        ensemble.register_model("transformer", transformer_model)
        ensemble.register_model("gnn", gnn_model)
        ensemble.register_model("swarm", swarm_model)
        ensemble.register_model("diffusion", diffusion_model)

        # Get ensemble prediction
        prediction = ensemble.predict(current_variables, current_state)

        if prediction.uncertainty > 0.7:
            action = "STAY_OUT"  # Too uncertain
        elif prediction.confidence > 0.75:
            action = prediction.direction  # High confidence
    """

    def __init__(
        self,
        window_size: int = 100,  # Track last 100 predictions for performance
        learning_rate: float = 0.01,
    ):
        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}
        self.performance_history: Dict[str, deque] = {}
        self.window_size = window_size
        self.learning_rate = learning_rate

        # Regime-specific weights (learned over time)
        self.regime_weights: Dict[str, Dict[str, float]] = {}
        self.current_regime: Optional[str] = None

        logger.info("[ENSEMBLE] Initialized meta-learner")

    def register_model(self, name: str, model: Any, initial_weight: float = 1.0):
        """Register a model for ensemble."""
        self.models[name] = model
        self.weights[name] = initial_weight
        self.performance_history[name] = deque(maxlen=self.window_size)
        logger.info(f"[ENSEMBLE] Registered model: {name} (weight: {initial_weight:.2f})")

    def predict(
        self,
        variables: Dict[str, float],
        state: Dict[str, Any],
        regime: Optional[str] = None,
    ) -> EnsemblePrediction:
        """Generate ensemble prediction from all models.

        Args:
            variables: Current variable values
            state: Current market state (for context)
            regime: Optional regime label for regime-aware weighting

        Returns:
            EnsemblePrediction with combined forecast
        """
        if not self.models:
            raise ValueError("No models registered in ensemble")

        self.current_regime = regime

        # Collect predictions from all models
        predictions = []

        for name, model in self.models.items():
            try:
                pred = self._get_model_prediction(name, model, variables, state)
                predictions.append(pred)
                logger.debug(f"[ENSEMBLE] {name}: {pred.direction} ({pred.confidence:.2f})")
            except Exception as e:
                logger.warning(f"[ENSEMBLE] {name} prediction failed: {e}")
                continue

        if not predictions:
            raise RuntimeError("All models failed to generate predictions")

        # Get adaptive weights
        model_weights = self._compute_adaptive_weights(predictions, regime)

        # Combine predictions
        combined = self._combine_predictions(predictions, model_weights)

        # Calculate ensemble uncertainty
        uncertainty = self._calculate_ensemble_uncertainty(predictions, model_weights)

        # Build final prediction
        ensemble_pred = EnsemblePrediction(
            direction=combined["direction"],
            confidence=combined["confidence"],
            probabilities=combined["probabilities"],
            uncertainty=uncertainty,
            model_weights=model_weights,
            individual_predictions=predictions,
            regime_context=regime,
        )

        logger.info(
            f"[ENSEMBLE] Prediction: {ensemble_pred.direction} "
            f"(conf: {ensemble_pred.confidence:.2f}, unc: {ensemble_pred.uncertainty:.2f})"
        )

        return ensemble_pred

    def _get_model_prediction(
        self, name: str, model: Any, variables: Dict[str, float], state: Dict[str, Any]
    ) -> ModelPrediction:
        """Get prediction from a single model (with standardized interface)."""

        # Each model type has different interface - adapt here
        if name == "causal":
            # Existing causal model
            result = model.predict(variables)
            return ModelPrediction(
                model_name="causal",
                direction=result.get("direction", "neutral"),
                confidence=result.get("confidence", 0.5),
                probabilities=result.get("probabilities", {}),
            )

        elif name == "transformer":
            # PatchTST model
            result = model.predict(state.get("ohlcv_history"))
            return ModelPrediction(
                model_name="transformer",
                direction=result["direction"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
            )

        elif name == "gnn":
            # Graph neural network
            result = model.predict(state.get("market_graph"))
            return ModelPrediction(
                model_name="gnn",
                direction=result["direction"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
            )

        elif name == "swarm":
            # Existing swarm simulation
            result = model.simulate(state["current_price"], variables)
            direction = result.consensus_direction
            confidence = result.consensus_strength
            return ModelPrediction(
                model_name="swarm",
                direction=direction,
                confidence=confidence,
                probabilities={
                    "bullish": confidence if direction == "bullish" else 0.0,
                    "neutral": confidence if direction == "neutral" else 0.0,
                    "bearish": confidence if direction == "bearish" else 0.0,
                },
            )

        elif name == "diffusion":
            # Diffusion trajectory model
            distribution = model.predict(state.get("ohlcv_history"), horizon=24)
            summary = distribution.summary()

            # Interpret distribution into direction
            mean_change = summary["mean_price_change_pct"]
            if mean_change > 1.0:
                direction = "bullish"
            elif mean_change < -1.0:
                direction = "bearish"
            else:
                direction = "neutral"

            confidence = 1.0 - distribution.uncertainty

            return ModelPrediction(
                model_name="diffusion",
                direction=direction,
                confidence=confidence,
                probabilities={
                    "bullish": max(0, mean_change / 5.0) if mean_change > 0 else 0,
                    "neutral": 1.0 - abs(mean_change / 5.0),
                    "bearish": max(0, -mean_change / 5.0) if mean_change < 0 else 0,
                },
                uncertainty=distribution.uncertainty,
                metadata=summary,
            )

        else:
            raise ValueError(f"Unknown model type: {name}")

    def _compute_adaptive_weights(
        self, predictions: List[ModelPrediction], regime: Optional[str]
    ) -> Dict[str, float]:
        """Compute adaptive weights based on recent performance."""

        # If regime-specific weights exist, use them
        if regime and regime in self.regime_weights:
            logger.debug(f"[ENSEMBLE] Using regime-specific weights for: {regime}")
            return self.regime_weights[regime].copy()

        # Otherwise use recent performance-based weights
        weights = {}
        total_weight = 0.0

        for pred in predictions:
            name = pred.model_name

            # Base weight
            base_weight = self.weights.get(name, 1.0)

            # Adjust by recent performance (if available)
            if self.performance_history[name]:
                recent_accuracy = np.mean(list(self.performance_history[name]))
                adjusted_weight = base_weight * (0.5 + recent_accuracy)  # Scale: 0.5-1.5x
            else:
                adjusted_weight = base_weight

            weights[name] = adjusted_weight
            total_weight += adjusted_weight

        # Normalize to sum to 1.0
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def _combine_predictions(
        self, predictions: List[ModelPrediction], weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Combine predictions using weighted averaging."""

        # Accumulate weighted probabilities
        combined_probs = {"bullish": 0.0, "neutral": 0.0, "bearish": 0.0}

        for pred in predictions:
            weight = weights.get(pred.model_name, 0.0)
            for direction, prob in pred.probabilities.items():
                combined_probs[direction] += weight * prob

        # Determine final direction and confidence
        final_direction = max(combined_probs, key=combined_probs.get)
        final_confidence = combined_probs[final_direction]

        return {
            "direction": final_direction,
            "confidence": final_confidence,
            "probabilities": combined_probs,
        }

    def _calculate_ensemble_uncertainty(
        self, predictions: List[ModelPrediction], weights: Dict[str, float]
    ) -> float:
        """Calculate ensemble uncertainty from model disagreement."""

        # Method 1: Disagreement (variance in predictions)
        directions = [p.direction for p in predictions]
        unique_directions = set(directions)
        disagreement = len(unique_directions) / 3.0  # Normalized to 0-1

        # Method 2: Weighted average of individual uncertainties
        avg_uncertainty = 0.0
        total_weight = 0.0

        for pred in predictions:
            if pred.uncertainty is not None:
                weight = weights.get(pred.model_name, 0.0)
                avg_uncertainty += weight * pred.uncertainty
                total_weight += weight

        if total_weight > 0:
            avg_uncertainty /= total_weight
        else:
            avg_uncertainty = 0.5  # Default medium uncertainty

        # Combine both measures
        ensemble_uncertainty = (disagreement + avg_uncertainty) / 2.0

        return float(np.clip(ensemble_uncertainty, 0.0, 1.0))

    def update_performance(self, model_name: str, was_correct: bool):
        """Update performance history for a model.

        Call this after each prediction to track accuracy.

        Args:
            model_name: Name of the model
            was_correct: Whether the prediction was correct
        """
        if model_name in self.performance_history:
            self.performance_history[model_name].append(1.0 if was_correct else 0.0)

            # Update weights using online learning
            if len(self.performance_history[model_name]) >= 10:
                recent_acc = np.mean(list(self.performance_history[model_name])[-10:])
                self.weights[model_name] = max(0.1, recent_acc)  # Minimum weight of 0.1

                logger.debug(
                    f"[ENSEMBLE] {model_name} recent accuracy: {recent_acc:.2%}, "
                    f"weight: {self.weights[model_name]:.2f}"
                )

    def learn_regime_weights(self, regime: str):
        """Learn optimal weights for a specific regime based on performance."""
        if regime not in self.regime_weights:
            # Initialize with current weights
            self.regime_weights[regime] = self.weights.copy()
            logger.info(f"[ENSEMBLE] Created regime weights for: {regime}")

        # TODO: Implement regime-specific weight learning
        # This would analyze performance within this regime and adjust weights accordingly

    def get_model_rankings(self) -> List[tuple[str, float]]:
        """Get models ranked by recent performance."""
        rankings = []

        for name in self.models.keys():
            if self.performance_history[name]:
                accuracy = np.mean(list(self.performance_history[name]))
                rankings.append((name, accuracy))
            else:
                rankings.append((name, 0.5))  # Default

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_status(self) -> Dict[str, Any]:
        """Get ensemble status and statistics."""
        rankings = self.get_model_rankings()

        return {
            "num_models": len(self.models),
            "current_weights": self.weights.copy(),
            "model_rankings": rankings,
            "current_regime": self.current_regime,
            "num_regimes_learned": len(self.regime_weights),
        }
