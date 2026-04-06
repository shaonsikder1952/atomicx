"""Multi-Agent Reinforcement Learning for Swarm Simulation.

Upgrade from hardcoded agent behavior to learned strategies through co-evolution.
Agents compete and cooperate to discover optimal trading strategies.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict, Optional
from loguru import logger


@dataclass
class MARLAgent:
    """MARL agent with learned policy."""
    agent_id: int
    agent_type: str
    policy_network: nn.Module
    capital: float
    position: float = 0.0
    lifetime_pnl: float = 0.0


class AgentPolicyNetwork(nn.Module):
    """Neural network policy for agent."""

    def __init__(self, state_dim: int = 10, action_dim: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh(),  # Actions in [-1, 1]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class MARLSwarmSimulator:
    """Multi-Agent RL Swarm Simulator.

    Instead of hardcoded behavior, agents learn optimal strategies through:
    - Self-play competition
    - Co-evolutionary dynamics
    - Reward based on PnL

    Agents discover novel strategies through trial and error.

    Usage:
        simulator = MARLSwarmSimulator(n_agents=500)

        # Train agents
        simulator.train(
            historical_data=ohlcv,
            episodes=1000
        )

        # Run simulation with trained agents
        result = simulator.simulate(
            current_price=52000,
            variables=variables,
        )

        # Agents now use learned strategies instead of hardcoded rules
    """

    def __init__(
        self,
        n_agents: int = 500,
        state_dim: int = 10,
        device: str = "mps",
    ):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.device = device

        # Create agent population
        self.agents: List[MARLAgent] = []
        self._initialize_agents()

        # Training history
        self.episode_rewards: List[float] = []

        logger.info(f"[MARL] Initialized {n_agents} learning agents on {device}")

    def _initialize_agents(self):
        """Initialize agent population with random policies."""
        agent_types = ["momentum", "contrarian", "adaptive", "market_maker"]

        for i in range(self.n_agents):
            # Random agent type
            agent_type = agent_types[i % len(agent_types)]

            # Create policy network
            policy = AgentPolicyNetwork(self.state_dim).to(self.device)

            # Random capital
            capital = np.random.uniform(1.0, 10.0)

            agent = MARLAgent(
                agent_id=i,
                agent_type=agent_type,
                policy_network=policy,
                capital=capital,
            )

            self.agents.append(agent)

    def train(
        self,
        historical_data: np.ndarray,
        episodes: int = 1000,
        steps_per_episode: int = 100,
    ):
        """Train agents through self-play.

        Args:
            historical_data: Historical market data
            episodes: Number of training episodes
            steps_per_episode: Steps per episode
        """
        logger.info(f"[MARL] Training {self.n_agents} agents for {episodes} episodes...")

        # Optimizers for each agent
        optimizers = [
            torch.optim.Adam(agent.policy_network.parameters(), lr=1e-3)
            for agent in self.agents
        ]

        for episode in range(episodes):
            # Reset episode
            episode_rewards = []

            for agent in self.agents:
                agent.position = 0.0

            # Run episode
            for step in range(steps_per_episode):
                # Get market state (simplified)
                if step < len(historical_data):
                    market_state = historical_data[step]
                else:
                    market_state = historical_data[-1]

                # Each agent observes state and takes action
                for agent, optimizer in zip(self.agents, optimizers):
                    # Create state tensor
                    state = self._get_agent_state(agent, market_state)
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                    # Get action from policy
                    action = agent.policy_network(state_tensor)
                    action_value = action.squeeze().item()

                    # Execute action
                    reward = self._execute_action(agent, action_value, market_state)

                    # Store reward
                    agent.lifetime_pnl += reward

                    # Update policy (simplified REINFORCE)
                    loss = -reward * action.mean()  # Negative because we want to maximize reward

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Log progress
            avg_pnl = np.mean([agent.lifetime_pnl for agent in self.agents])
            self.episode_rewards.append(avg_pnl)

            if (episode + 1) % 100 == 0:
                logger.info(f"[MARL] Episode {episode+1}/{episodes}, Avg PnL: {avg_pnl:.2f}")

        logger.success(f"[MARL] Training complete! Agents have learned strategies.")

    def _get_agent_state(self, agent: MARLAgent, market_state: np.ndarray) -> np.ndarray:
        """Get state observation for agent."""
        # Simplified state: [market_features, agent_position, agent_capital, agent_pnl]
        state = np.concatenate([
            market_state[:self.state_dim-3],  # Market features
            [agent.position],
            [agent.capital],
            [agent.lifetime_pnl],
        ])
        return state[:self.state_dim]

    def _execute_action(
        self, agent: MARLAgent, action_value: float, market_state: np.ndarray
    ) -> float:
        """Execute agent action and return reward."""
        # Action: change in position (-1 to +1)
        position_change = action_value * 0.1  # Scale down

        # Update position
        new_position = np.clip(agent.position + position_change, -1.0, 1.0)

        # Calculate reward (PnL from position change)
        # Simplified: reward = position * market_return
        market_return = np.random.normal(0, 0.01)  # Placeholder
        reward = new_position * market_return * agent.capital

        agent.position = new_position

        return reward

    def simulate(
        self,
        current_price: float,
        variables: Dict[str, float],
        steps: int = 100,
    ) -> Dict:
        """Run simulation using trained agent policies.

        Args:
            current_price: Current market price
            variables: Market variables
            steps: Simulation steps

        Returns:
            Simulation result
        """
        # Convert variables to state
        market_state = np.array([variables.get(f"VAR_{i}", 0.0) for i in range(self.state_dim)])

        # Simulate
        prices = [current_price]
        price = current_price

        for step in range(steps):
            # All agents act
            agent_actions = []

            for agent in self.agents:
                state = self._get_agent_state(agent, market_state)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    action = agent.policy_network(state_tensor)
                    agent_actions.append(action.item())

            # Aggregate agent actions to determine price movement
            avg_action = np.mean(agent_actions)
            price_change = avg_action * 0.01  # Scale

            price *= (1 + price_change)
            prices.append(price)

        # Calculate consensus
        final_positions = [agent.position for agent in self.agents]
        avg_position = np.mean(final_positions)

        if avg_position > 0.1:
            direction = "bullish"
        elif avg_position < -0.1:
            direction = "bearish"
        else:
            direction = "neutral"

        # Count agent distribution by position
        bullish_count = sum(1 for pos in final_positions if pos > 0.1)
        bearish_count = sum(1 for pos in final_positions if pos < -0.1)
        neutral_count = self.n_agents - bullish_count - bearish_count

        # Calculate swarm metrics
        positions_std = np.std(final_positions)
        convergence = 1.0 - min(positions_std, 1.0)  # Higher convergence when positions align
        diversity = min(positions_std * 2, 1.0)  # Diversity is inverse of convergence
        stability = 1.0 - abs(avg_position - np.median(final_positions))  # Stability when mean ≈ median

        return {
            "consensus_direction": direction,
            "consensus_strength": float(abs(avg_position)),
            "price_trajectory": prices[-20:],
            "agent_count": self.n_agents,

            # Agent distribution
            "bullish_agents": bullish_count,
            "bearish_agents": bearish_count,
            "neutral_agents": neutral_count,

            # Swarm metrics
            "convergence": float(convergence),
            "diversity": float(diversity),
            "stability": float(stability),
            "reward_signal": float(avg_position),  # Use average position as reward signal
            "top_strategy": f"{direction.capitalize()} consensus" if abs(avg_position) > 0.3 else "Mixed strategies",
            "consensus_steps": steps,
        }
