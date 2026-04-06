"""Reinforcement Learning Agent for Strategy Optimization.

Wrapper around stable-baselines3 PPO for trading strategy optimization.
Learns optimal position sizing and timing through trial and error.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Any
from loguru import logger
from pathlib import Path


class RLAgent:
    """RL-based trading strategy optimizer.

    Usage:
        from atomicx.env import TradingEnvironment
        from atomicx.models import RLAgent

        # Create environment
        env = TradingEnvironment(historical_data=ohlcv)

        # Create and train agent
        agent = RLAgent(env)
        agent.train(total_timesteps=1_000_000)

        # Use trained agent
        action = agent.predict(current_state)
    """

    def __init__(
        self,
        env: Any,  # TradingEnvironment
        algorithm: str = "PPO",
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        device: str = "mps",
    ):
        self.env = env
        self.algorithm = algorithm
        self.device = device

        # Import stable-baselines3 (lazy import to avoid dependency issues)
        try:
            from stable_baselines3 import PPO, A2C, DQN

            algo_map = {
                "PPO": PPO,
                "A2C": A2C,
                "DQN": DQN,
            }

            AlgoClass = algo_map.get(algorithm, PPO)

            self.model = AlgoClass(
                policy,
                env,
                learning_rate=learning_rate,
                device=device,
                verbose=1,
            )

            logger.info(f"[RL] Initialized {algorithm} agent on {device}")

        except ImportError:
            logger.error("[RL] stable-baselines3 not installed. Run: pip install stable-baselines3")
            self.model = None

    def train(self, total_timesteps: int = 1_000_000, callback=None):
        """Train the RL agent.

        Args:
            total_timesteps: Number of training steps
            callback: Optional callback for monitoring
        """
        if self.model is None:
            logger.error("[RL] Model not initialized")
            return

        logger.info(f"[RL] Training for {total_timesteps:,} timesteps...")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
        )

        logger.success(f"[RL] Training complete!")

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """Predict action for given observation.

        Args:
            observation: Current state
            deterministic: Use deterministic policy (no exploration)

        Returns:
            Dictionary with action and info
        """
        if self.model is None:
            logger.error("[RL] Model not initialized")
            return {"action": 0, "direction": "hold"}

        action, _states = self.model.predict(observation, deterministic=deterministic)

        # Convert action to human-readable
        action_map = {
            0: "hold",
            1: "buy_small",
            2: "buy_large",
            3: "sell_small",
            4: "sell_large",
        }

        return {
            "action": int(action),
            "direction": action_map.get(int(action), "hold"),
        }

    def save(self, path: str):
        """Save trained model."""
        if self.model is None:
            logger.error("[RL] Model not initialized")
            return

        self.model.save(path)
        logger.info(f"[RL] Model saved to {path}")

    def load(self, path: str):
        """Load trained model."""
        if self.model is None:
            logger.error("[RL] Model not initialized")
            return

        # Import algorithm class
        from stable_baselines3 import PPO, A2C, DQN

        algo_map = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
        AlgoClass = algo_map.get(self.algorithm, PPO)

        self.model = AlgoClass.load(path, env=self.env, device=self.device)
        logger.info(f"[RL] Model loaded from {path}")

    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate agent performance.

        Args:
            n_episodes: Number of episodes to evaluate

        Returns:
            Dictionary with performance metrics
        """
        if self.model is None:
            logger.error("[RL] Model not initialized")
            return {}

        from stable_baselines3.common.evaluation import evaluate_policy

        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.env,
            n_eval_episodes=n_episodes,
            deterministic=True,
        )

        logger.info(f"[RL] Evaluation: Mean reward={mean_reward:.2f} ± {std_reward:.2f}")

        return {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
        }
